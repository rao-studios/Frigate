//
//  Classifier.hpp
//  CVisionAX
//
//  WHAT: Two ONNX Runtime sessions — image → features, (features + boxes) → probs.
//  IN:   Engine::classifyRegions, and the split path (vx_classifier_prepare_image → another
//        backbone → vx_classifier_classify_features)
//  OUT:  OrtApi.hpp, ClassifierPreprocess.hpp
//  PIN:  TWO GRAPHS, NOT ONE, and the reason is arithmetic. A 1080p screenshot yields
//        ~1,000 boxes; two 7x7xC RoIAlign crops each is hundreds of megabytes of
//        activations. Chunking is the only way to bound that, and chunking a FUSED
//        graph would re-run the backbone per chunk. So the backbone runs once, its
//        feature tensor is handed to the head untouched, and only the head repeats.
//        (It is also the seam a GPU backbone needs: RoiAlign has no CoreML kernel, so a
//        single graph would be partitioned in the middle regardless.)
//        THE SPLIT PATH REUSES BOTH HALVES OF THIS. `prepare` is the same preprocessing and
//        `classifyFeatures` the same chunked head; only the backbone between them is
//        someone else's, and its output is checked against the shape the head was exported
//        for before it is wrapped.
//        Sessions are stateless across Run, so one classifier serves many threads.
//

#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ClassifierPreprocess.hpp"
#include "visionax.h"

struct OrtValue;

namespace visionax {

class Classifier {
public:
    struct Label {
        int32_t classIndex = 0;
        float confidence = 0;
    };

    Classifier(const std::string &backbonePath,
               const std::string &headPath,
               const vx_classifier_spec &spec);
    ~Classifier();

    Classifier(const Classifier &) = delete;
    Classifier &operator=(const Classifier &) = delete;

    /// One backbone pass, then the head in chunks. `regions` may be empty.
    std::vector<Label> classify(const cv::Mat &image,
                                const vx_region *regions,
                                int32_t count) const;

    /// The preprocessing `classify` performs, kept for another backbone to read.
    PreparedImage prepare(const cv::Mat &image) const;

    /// This classifier's own ONNX backbone over a prepared tensor (3 x height x width, CHW),
    /// copied out as CHW floats — the reference another backbone is measured against.
    std::vector<float> backboneFeatures(const float *image,
                                        int64_t height,
                                        int64_t width,
                                        int64_t &channels,
                                        int64_t &featureHeight,
                                        int64_t &featureWidth) const;

    /// The head alone, over features another backbone computed from `prepared`.
    /// Throws, naming both shapes, when the features are not what the head was exported for.
    std::vector<Label> classifyFeatures(const PreparedImage &prepared,
                                        const float *features,
                                        int64_t channels,
                                        int64_t height,
                                        int64_t width,
                                        const vx_region *regions,
                                        int32_t count) const;

    const vx_classifier_spec &spec() const { return spec_; }

    void setLastError(const std::string &message);
    std::string lastError() const;

    /// ONNX backbone runs only — a run on another backbone does not count here.
    int64_t backboneRuns() const { return backboneRuns_.load(); }
    int64_t headRuns() const { return headRuns_.load(); }

private:
    /// The ONNX backbone over a 3 x height x width CHW tensor. The caller owns and
    /// releases the value.
    OrtValue *runBackbone(const float *image, int64_t height, int64_t width) const;

    /// The chunked head over `features`, whichever backbone produced them.
    std::vector<Label> runHead(const PreparedImage &prepared,
                               const OrtValue *features,
                               const vx_region *regions,
                               int32_t count) const;

    struct Impl;
    Impl *impl_;
    vx_classifier_spec spec_;
    /// The head's feature channel count, read from its graph at load; -1 when dynamic.
    int64_t featureChannels_ = -1;
    mutable std::atomic<int64_t> backboneRuns_{0};
    mutable std::atomic<int64_t> headRuns_{0};
    mutable std::mutex errorMutex_;
    std::string lastError_;
};

}  // namespace visionax
