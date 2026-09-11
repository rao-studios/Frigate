//
//  Classifier.cpp
//  CVisionAX
//
//  WHAT: Session setup, name binding, the chunked run, argmax — and the split path's two
//        halves, which are the same code with the backbone taken out of the middle.
//  IN:   Classifier.hpp
//  OUT:  OrtApi.hpp
//  PIN:  Inputs are bound BY NAME, never by position. The head takes two tensors of
//        different rank and meaning; if an export ever reorders them, position-binding
//        would feed boxes to the feature port and ORT would report a shape error three
//        layers deep instead of "no input named boxes". The class count is checked
//        against the graph's own output width at load, so a spec that disagrees with
//        its model fails at create() rather than mislabelling every region.
//

#include "Classifier.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "ClassifierPreprocess.hpp"
#include "OrtApi.hpp"

namespace visionax {

namespace {

constexpr const char *kImageInput = "image";
constexpr const char *kFeatureOutput = "features";
constexpr const char *kFeatureInput = "features";
constexpr const char *kBoxesInput = "boxes";
constexpr const char *kProbsOutput = "probs";

/// RAII for the handful of ORT objects that outlive a statement.
template <typename T, void (*Release)(T *)>
struct OrtHandle {
    T *value = nullptr;
    OrtHandle() = default;
    explicit OrtHandle(T *v) : value(v) {}
    ~OrtHandle() { if (value != nullptr) Release(value); }
    OrtHandle(const OrtHandle &) = delete;
    OrtHandle &operator=(const OrtHandle &) = delete;
    OrtHandle(OrtHandle &&other) noexcept : value(other.value) { other.value = nullptr; }
    OrtHandle &operator=(OrtHandle &&other) noexcept {
        if (this != &other) {
            if (value != nullptr) Release(value);
            value = other.value;
            other.value = nullptr;
        }
        return *this;
    }
    T *get() const { return value; }
};

void releaseValue(OrtValue *value) { ort().ReleaseValue(value); }
using ValueHandle = OrtHandle<OrtValue, releaseValue>;

std::vector<std::string> sessionInputNames(OrtSession *session, OrtAllocator *allocator) {
    size_t count = 0;
    ortCheck(ort().SessionGetInputCount(session, &count));
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        char *name = nullptr;
        ortCheck(ort().SessionGetInputName(session, i, allocator, &name));
        names.emplace_back(name);
        ortDiscard(ort().AllocatorFree(allocator, name));
    }
    return names;
}

std::vector<std::string> sessionOutputNames(OrtSession *session, OrtAllocator *allocator) {
    size_t count = 0;
    ortCheck(ort().SessionGetOutputCount(session, &count));
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        char *name = nullptr;
        ortCheck(ort().SessionGetOutputName(session, i, allocator, &name));
        names.emplace_back(name);
        ortDiscard(ort().AllocatorFree(allocator, name));
    }
    return names;
}

void requireName(const std::vector<std::string> &names, const char *needed, const char *what) {
    if (std::find(names.begin(), names.end(), needed) == names.end()) {
        std::string available;
        for (const auto &name : names) {
            if (!available.empty()) available += ", ";
            available += name;
        }
        throw std::runtime_error(std::string("the ") + what + " graph has no tensor named \"" +
                                 needed + "\" (it has: " + available + ")");
    }
}

/// The dimensions of a tensor type, or empty when the type is not a tensor.
std::vector<int64_t> tensorShape(OrtTypeInfo *typeInfo) {
    const OrtTensorTypeAndShapeInfo *shapeInfo = nullptr;
    OrtStatus *status = ort().CastTypeInfoToTensorInfo(typeInfo, &shapeInfo);
    if (status != nullptr || shapeInfo == nullptr) {
        if (status != nullptr) ort().ReleaseStatus(status);
        return {};
    }
    size_t dims = 0;
    ortCheck(ort().GetDimensionsCount(shapeInfo, &dims));
    std::vector<int64_t> shape(dims, 0);
    ortCheck(ort().GetDimensions(shapeInfo, shape.data(), dims));
    return shape;
}

/// The head's output width — the number of classes the graph really predicts.
int64_t headClassCount(OrtSession *head) {
    OrtTypeInfo *typeInfo = nullptr;
    ortCheck(ort().SessionGetOutputTypeInfo(head, 0, &typeInfo));
    const std::vector<int64_t> shape = tensorShape(typeInfo);
    ort().ReleaseTypeInfo(typeInfo);
    return shape.size() >= 2 ? shape.back() : -1;
}

/// The channel count the head's `features` input was exported with; -1 when dynamic.
int64_t headFeatureChannels(OrtSession *head, OrtAllocator *allocator) {
    const std::vector<std::string> names = sessionInputNames(head, allocator);
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] != kFeatureInput) continue;
        OrtTypeInfo *typeInfo = nullptr;
        ortCheck(ort().SessionGetInputTypeInfo(head, i, &typeInfo));
        const std::vector<int64_t> shape = tensorShape(typeInfo);
        ort().ReleaseTypeInfo(typeInfo);
        return shape.size() == 4 ? shape[1] : -1;
    }
    return -1;
}

}  // namespace

struct Classifier::Impl {
    OrtSessionOptions *options = nullptr;
    OrtSession *backbone = nullptr;
    OrtSession *head = nullptr;
    OrtMemoryInfo *memory = nullptr;
    OrtAllocator *allocator = nullptr;

    ~Impl() {
        if (backbone != nullptr) ort().ReleaseSession(backbone);
        if (head != nullptr) ort().ReleaseSession(head);
        if (options != nullptr) ort().ReleaseSessionOptions(options);
        if (memory != nullptr) ort().ReleaseMemoryInfo(memory);
    }
};

Classifier::Classifier(const std::string &backbonePath,
                       const std::string &headPath,
                       const vx_classifier_spec &spec)
    : impl_(new Impl()), spec_(spec) {
    try {
        OrtEnv *env = ortEnv();
        ortCheck(ort().CreateSessionOptions(&impl_->options));
        if (spec.intra_op_threads > 0) {
            ortCheck(ort().SetIntraOpNumThreads(impl_->options, spec.intra_op_threads));
        }
        ortCheck(ort().SetSessionGraphOptimizationLevel(impl_->options, ORT_ENABLE_ALL));
        ortCheck(ort().SetSessionExecutionMode(impl_->options, ORT_SEQUENTIAL));

        ortCheck(ort().CreateSession(env, backbonePath.c_str(), impl_->options, &impl_->backbone));
        ortCheck(ort().CreateSession(env, headPath.c_str(), impl_->options, &impl_->head));
        ortCheck(ort().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &impl_->memory));
        ortCheck(ort().GetAllocatorWithDefaultOptions(&impl_->allocator));

        requireName(sessionInputNames(impl_->backbone, impl_->allocator), kImageInput, "backbone");
        requireName(sessionOutputNames(impl_->backbone, impl_->allocator), kFeatureOutput, "backbone");
        const auto headInputs = sessionInputNames(impl_->head, impl_->allocator);
        requireName(headInputs, kFeatureInput, "head");
        requireName(headInputs, kBoxesInput, "head");
        requireName(sessionOutputNames(impl_->head, impl_->allocator), kProbsOutput, "head");

        const int64_t classes = headClassCount(impl_->head);
        if (classes > 0 && spec.class_count > 0 && classes != spec.class_count) {
            throw std::runtime_error("the model predicts " + std::to_string(classes) +
                                     " classes but its spec declares " +
                                     std::to_string(spec.class_count));
        }
        featureChannels_ = headFeatureChannels(impl_->head, impl_->allocator);
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

Classifier::~Classifier() {
    delete impl_;
}

void Classifier::setLastError(const std::string &message) {
    std::lock_guard<std::mutex> guard(errorMutex_);
    lastError_ = message;
}

std::string Classifier::lastError() const {
    std::lock_guard<std::mutex> guard(errorMutex_);
    return lastError_;
}

std::vector<Classifier::Label> Classifier::classify(const cv::Mat &image,
                                                    const vx_region *regions,
                                                    int32_t count) const {
    if (count <= 0 || regions == nullptr) {
        return {};
    }
    const PreparedImage prepared = prepareImage(image, spec_);
    ValueHandle features(
        runBackbone(prepared.tensor.data(), prepared.paddedHeight, prepared.paddedWidth));
    return runHead(prepared, features.get(), regions, count);
}

PreparedImage Classifier::prepare(const cv::Mat &image) const {
    return prepareImage(image, spec_);
}

OrtValue *Classifier::runBackbone(const float *image, int64_t height, int64_t width) const {
    const int64_t imageShape[4] = {1, 3, height, width};
    ValueHandle imageValue;
    {
        OrtValue *raw = nullptr;
        ortCheck(ort().CreateTensorWithDataAsOrtValue(
            impl_->memory,
            const_cast<float *>(image),
            static_cast<size_t>(3 * height * width) * sizeof(float),
            imageShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw));
        imageValue = ValueHandle(raw);
    }

    const char *backboneInputs[] = {kImageInput};
    const char *backboneOutputs[] = {kFeatureOutput};
    OrtValue *featuresRaw = nullptr;
    const OrtValue *backboneInputValues[] = {imageValue.get()};
    ortCheck(ort().Run(impl_->backbone, nullptr, backboneInputs, backboneInputValues, 1,
                       backboneOutputs, 1, &featuresRaw));
    backboneRuns_.fetch_add(1);
    return featuresRaw;
}

std::vector<float> Classifier::backboneFeatures(const float *image,
                                                int64_t height,
                                                int64_t width,
                                                int64_t &channels,
                                                int64_t &featureHeight,
                                                int64_t &featureWidth) const {
    if (image == nullptr || height <= 0 || width <= 0) {
        throw std::runtime_error("the backbone needs a 3 x height x width tensor");
    }
    ValueHandle features(runBackbone(image, height, width));

    OrtTensorTypeAndShapeInfo *info = nullptr;
    ortCheck(ort().GetTensorTypeAndShape(features.get(), &info));
    size_t dims = 0;
    ortCheck(ort().GetDimensionsCount(info, &dims));
    std::vector<int64_t> shape(dims, 0);
    ortCheck(ort().GetDimensions(info, shape.data(), dims));
    ort().ReleaseTensorTypeAndShapeInfo(info);
    if (dims != 4 || shape[0] != 1) {
        throw std::runtime_error("the backbone returned a feature map that is not [1, C, H, W]");
    }
    channels = shape[1];
    featureHeight = shape[2];
    featureWidth = shape[3];

    float *data = nullptr;
    ortCheck(ort().GetTensorMutableData(features.get(), reinterpret_cast<void **>(&data)));
    return std::vector<float>(
        data, data + static_cast<size_t>(channels * featureHeight * featureWidth));
}

std::vector<Classifier::Label> Classifier::classifyFeatures(const PreparedImage &prepared,
                                                            const float *features,
                                                            int64_t channels,
                                                            int64_t height,
                                                            int64_t width,
                                                            const vx_region *regions,
                                                            int32_t count) const {
    if (count <= 0 || regions == nullptr) {
        return {};
    }
    // THE HEAD READS THE IMAGE EXTENT FROM THE MAP'S OWN SHAPE, so a map of the wrong size
    // would not fail — it would crop every box against the wrong extent, silently. Refused
    // here, naming both shapes, before anything is wrapped.
    const int32_t stride = spec_.feature_stride > 0 ? spec_.feature_stride : 8;
    const int64_t wantHeight = prepared.paddedHeight / stride;
    const int64_t wantWidth = prepared.paddedWidth / stride;
    if (height != wantHeight || width != wantWidth) {
        throw std::runtime_error(
            "the backbone returned a " + std::to_string(height) + "x" + std::to_string(width) +
            " feature map; the head needs " + std::to_string(wantHeight) + "x" +
            std::to_string(wantWidth) + " (the padded image divided by stride " +
            std::to_string(stride) + ")");
    }
    if (featureChannels_ > 0 && channels != featureChannels_) {
        throw std::runtime_error(
            "the backbone returned " + std::to_string(channels) +
            " channels; the head was exported for " + std::to_string(featureChannels_));
    }

    const int64_t shape[4] = {1, channels, height, width};
    const size_t elements = static_cast<size_t>(channels * height * width);
    ValueHandle value;
    {
        OrtValue *raw = nullptr;
        ortCheck(ort().CreateTensorWithDataAsOrtValue(
            impl_->memory, const_cast<float *>(features), elements * sizeof(float),
            shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw));
        value = ValueHandle(raw);
    }
    return runHead(prepared, value.get(), regions, count);
}

std::vector<Classifier::Label> Classifier::runHead(const PreparedImage &prepared,
                                                   const OrtValue *features,
                                                   const vx_region *regions,
                                                   int32_t count) const {
    std::vector<float> rois = prepareBoxes(regions, count, prepared);
    const int32_t chunkSize = spec_.max_boxes_per_run > 0 ? spec_.max_boxes_per_run : 512;
    std::vector<Label> labels(static_cast<size_t>(count));

    for (int32_t start = 0; start < count; start += chunkSize) {
        const int32_t rows = std::min(chunkSize, count - start);
        const int64_t boxShape[2] = {rows, 5};
        ValueHandle boxValue;
        {
            OrtValue *raw = nullptr;
            ortCheck(ort().CreateTensorWithDataAsOrtValue(
                impl_->memory,
                rois.data() + static_cast<size_t>(start) * 5,
                static_cast<size_t>(rows) * 5 * sizeof(float),
                boxShape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw));
            boxValue = ValueHandle(raw);
        }

        const char *headInputs[] = {kFeatureInput, kBoxesInput};
        const OrtValue *headInputValues[] = {features, boxValue.get()};
        const char *headOutputs[] = {kProbsOutput};
        OrtValue *probsRaw = nullptr;
        ortCheck(ort().Run(impl_->head, nullptr, headInputs, headInputValues, 2,
                           headOutputs, 1, &probsRaw));
        ValueHandle probs(probsRaw);
        headRuns_.fetch_add(1);

        float *data = nullptr;
        ortCheck(ort().GetTensorMutableData(probs.get(), reinterpret_cast<void **>(&data)));

        OrtTensorTypeAndShapeInfo *info = nullptr;
        ortCheck(ort().GetTensorTypeAndShape(probs.get(), &info));
        size_t dims = 0;
        ortCheck(ort().GetDimensionsCount(info, &dims));
        std::vector<int64_t> shape(dims, 0);
        ortCheck(ort().GetDimensions(info, shape.data(), dims));
        ort().ReleaseTensorTypeAndShapeInfo(info);

        const int64_t classes = dims >= 2 ? shape[dims - 1] : 0;
        if (classes <= 0 || (dims >= 1 && shape[0] != rows)) {
            throw std::runtime_error("the head returned an unexpected probability shape");
        }

        for (int32_t row = 0; row < rows; ++row) {
            const float *values = data + static_cast<size_t>(row) * classes;
            int32_t best = 0;
            float bestValue = values[0];
            for (int64_t klass = 1; klass < classes; ++klass) {
                if (values[klass] > bestValue) {
                    bestValue = values[klass];
                    best = static_cast<int32_t>(klass);
                }
            }
            labels[static_cast<size_t>(start + row)] = Label{best, bestValue};
        }
    }

    return labels;
}

}  // namespace visionax
