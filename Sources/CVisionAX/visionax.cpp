//
//  visionax.cpp
//  CVisionAX
//
//  WHAT: The extern "C" facade — argument checks, Mat wrapping, malloc'd outputs.
//  IN:   include/visionax.h
//  OUT:  Engine.hpp
//  PIN:  No exception crosses this file. cv::Exception and std::exception both land
//        as VX_ERROR_INTERNAL with the outputs left zeroed.
//

#include "visionax.h"

#include "IconGlyphs.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>

#include "Classifier.hpp"
#include "Engine.hpp"
#include "MediaControls.hpp"
#include "OrtApi.hpp"

struct vx_engine {
    visionax::Engine engine;
};

struct vx_classifier {
    visionax::Classifier classifier;

    vx_classifier(const char *backbonePath, const char *headPath, const vx_classifier_spec &spec)
        : classifier(backbonePath, headPath, spec) {}
};

struct vx_prepared_image {
    visionax::PreparedImage prepared;
};

namespace {

/// Why the LAST vx_classifier_create failed. Create has no object to hang a message on,
/// and "the model would not load" is exactly when the reason matters most.
std::string &lastCreateError() {
    static std::string message;
    return message;
}

bool validImage(const vx_image_view *image) {
    if (image == nullptr || image->data == nullptr) return false;
    if (image->width <= 0 || image->height <= 0) return false;
    const int32_t bytesPerPixel = image->format == VX_PIXEL_GRAY8 ? 1 : 4;
    return image->bytes_per_row >= image->width * bytesPerPixel;
}

bool validOptions(const vx_canny_options *options) {
    if (options == nullptr) return false;
    if (options->aperture_size != 3 && options->aperture_size != 5 && options->aperture_size != 7) {
        return false;
    }
    if (options->low_threshold < 0 || options->high_threshold < 0) return false;
    if (options->blur_kernel < 0 || options->close_kernel < 0) return false;
    if (options->min_width < 0 || options->min_height < 0) return false;
    if (options->merge_iou < 0 || options->merge_iou > 1) return false;
    if (options->max_nodes < 1 || options->max_depth < 0) return false;
    return true;
}

/// A Mat header over the caller's pixels — no copy for GRAY8/BGRA8, one cvtColor
/// copy for RGBA8. Returns false for an unknown format.
bool wrap(const vx_image_view &image, cv::Mat &out) {
    void *data = const_cast<uint8_t *>(image.data);
    const size_t step = static_cast<size_t>(image.bytes_per_row);
    switch (image.format) {
    case VX_PIXEL_GRAY8:
        out = cv::Mat(image.height, image.width, CV_8UC1, data, step);
        return true;
    case VX_PIXEL_BGRA8:
        out = cv::Mat(image.height, image.width, CV_8UC4, data, step);
        return true;
    case VX_PIXEL_RGBA8: {
        const cv::Mat rgba(image.height, image.width, CV_8UC4, data, step);
        cv::cvtColor(rgba, out, cv::COLOR_RGBA2BGRA);
        return true;
    }
    }
    return false;
}

bool copyTree(const visionax::RegionTree &tree, vx_region_tree &out) {
    const size_t count = tree.regions.size();
    auto *regions = static_cast<vx_region *>(std::malloc(sizeof(vx_region) * (count == 0 ? 1 : count)));
    if (regions == nullptr) return false;
    for (size_t i = 0; i < count; ++i) {
        const visionax::Region &r = tree.regions[i];
        regions[i].id = r.id;
        regions[i].parent = r.parent;
        regions[i].depth = r.depth;
        regions[i].x = r.rect.x;
        regions[i].y = r.rect.y;
        regions[i].width = r.rect.width;
        regions[i].height = r.rect.height;
        regions[i].child_count = r.childCount;
    }
    out.regions = regions;
    out.count = static_cast<int32_t>(count);
    out.truncated = tree.truncated ? 1 : 0;
    out.contour_count = tree.contourCount;
    return true;
}

bool copyEdges(const cv::Mat &edges, vx_edge_map &out) {
    if (edges.empty() || edges.type() != CV_8UC1) return false;
    const size_t width = static_cast<size_t>(edges.cols);
    const size_t height = static_cast<size_t>(edges.rows);
    auto *data = static_cast<uint8_t *>(std::malloc(width * height));
    if (data == nullptr) return false;
    for (size_t row = 0; row < height; ++row) {
        std::memcpy(data + row * width, edges.ptr<uint8_t>(static_cast<int>(row)), width);
    }
    out.data = data;
    out.width = edges.cols;
    out.height = edges.rows;
    out.bytes_per_row = edges.cols;
    return true;
}

vx_rect toRect(const cv::Rect &rect) {
    return vx_rect{rect.x, rect.y, rect.width, rect.height};
}

vx_media_control toControl(const visionax::MediaControl &control) {
    vx_media_control out;
    out.frame = toRect(control.frame);
    out.glyph = control.glyph;
    out.confidence = control.confidence;
    return out;
}

}  // namespace

extern "C" {

const char *vx_status_message(vx_status status) {
    switch (status) {
    case VX_OK: return "ok";
    case VX_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case VX_ERROR_UNSUPPORTED_FORMAT: return "unsupported pixel format";
    case VX_ERROR_INTERNAL: return "internal failure";
    case VX_ERROR_MODEL: return "the model failed to load or run";
    }
    return "unknown status";
}

vx_canny_options vx_canny_options_default(void) {
    vx_canny_options options;
    // MEASURED, NOT INHERITED FROM A TUTORIAL. 50/150 is the canonical Canny example
    // and it is wrong for user interfaces: a 1px #374151 border on a #1f2937 dark
    // surface never reaches it, so tables, toolbars and cards produce no contour at
    // all while their text does. Scored against harvested ground truth, 20/60 lifts
    // proposal recall from 44% to 70% for +10% proposals and no extra time; 15/45
    // adds nothing further. On a real 1920x1080 desktop it finds 1320 regions where
    // 50/150 found 978, at the same 90ms.
    options.low_threshold = 20;
    options.high_threshold = 60;
    // 5, not the usual 3. A 3x3 Sobel accumulates too little gradient across a soft,
    // antialiased, low-contrast 1px border, so text FIELDS were never proposed at all
    // (0% recall) while the placeholder text inside them was. Widening the kernel took
    // text fields to 100% and buttons to 100% for the same 550 proposals and the same
    // 8ms. Lowering the thresholds instead does not fix it — the gradient is not weak,
    // it is spread out.
    options.aperture_size = 5;
    options.blur_kernel = 3;
    // 5, not 3, measured the same way as the thresholds: it merges the fragments of a
    // word into one blob, lifting AXStaticText recall from 81% to 91% while cutting
    // proposals by 30%. Larger keeps helping headings and lists — they are unions of
    // separated words — but 7 and above starts fusing adjacent links into each other
    // (Link recall 100% -> 60% -> 36%), which is a worse trade than the one it wins.
    options.close_kernel = 5;
    options.min_width = 8;
    options.min_height = 8;
    options.merge_iou = 0.90;
    options.merge_slack = 4;
    options.containment_slack = 2;
    options.reading_band = 24;
    options.max_depth = 24;
    options.max_nodes = 4000;
    return options;
}

vx_engine *vx_engine_create(void) {
    return new (std::nothrow) vx_engine();
}

void vx_engine_destroy(vx_engine *engine) {
    delete engine;
}

const char *vx_engine_opencv_version(void) {
    return CV_VERSION;
}

const char *vx_engine_version(void) {
#ifdef VISIONAX_VERSION
    return VISIONAX_VERSION;
#else
    return "0.0.0";
#endif
}

const char *vx_onnxruntime_version(void) {
    try {
        // Touch the environment too: GetVersionString alone would link without the
        // archive's real body, and this probe exists to prove the whole thing is there.
        (void)visionax::ortEnv();
        return OrtGetApiBase()->GetVersionString();
    } catch (const std::exception &) {
        return NULL;
    }
}

vx_status vx_engine_detect_regions(vx_engine *engine,
                                   const vx_image_view *image,
                                   const vx_canny_options *options,
                                   vx_region_tree *out_tree,
                                   vx_edge_map *out_edges) {
    if (out_tree != nullptr) std::memset(out_tree, 0, sizeof(*out_tree));
    if (out_edges != nullptr) std::memset(out_edges, 0, sizeof(*out_edges));
    if (engine == nullptr || out_tree == nullptr || !validImage(image) || !validOptions(options)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }

    try {
        cv::Mat mat;
        if (!wrap(*image, mat)) {
            return VX_ERROR_UNSUPPORTED_FORMAT;
        }
        cv::Mat edges;
        const visionax::RegionTree tree =
            engine->engine.detectRegions(mat, *options, out_edges != nullptr ? &edges : nullptr);

        if (!copyTree(tree, *out_tree)) {
            return VX_ERROR_INTERNAL;
        }
        if (out_edges != nullptr && !copyEdges(edges, *out_edges)) {
            vx_region_tree_free(out_tree);
            return VX_ERROR_INTERNAL;
        }
        return VX_OK;
    } catch (const std::exception &) {
        vx_region_tree_free(out_tree);
        if (out_edges != nullptr) vx_edge_map_free(out_edges);
        return VX_ERROR_INTERNAL;
    }
}

void vx_region_tree_free(vx_region_tree *tree) {
    if (tree == nullptr) return;
    std::free(tree->regions);
    std::memset(tree, 0, sizeof(*tree));
}

void vx_edge_map_free(vx_edge_map *edges) {
    if (edges == nullptr) return;
    std::free(edges->data);
    std::memset(edges, 0, sizeof(*edges));
}

// MARK: - Classifier

vx_classifier_spec vx_classifier_spec_default(void) {
    vx_classifier_spec spec;
    spec.long_side = 1600;
    spec.pad_multiple = 32;
    // ImageNet statistics — the backbone is pretrained, so these are not free choices.
    spec.mean[0] = 0.485f;
    spec.mean[1] = 0.456f;
    spec.mean[2] = 0.406f;
    spec.std[0] = 0.229f;
    spec.std[1] = 0.224f;
    spec.std[2] = 0.225f;
    spec.class_count = 0;
    spec.max_boxes_per_run = 512;
    spec.intra_op_threads = 0;
    spec.execution_provider = 0;
    spec.feature_stride = 8;
    return spec;
}

vx_status vx_classifier_create(const char *backbone_path,
                               const char *head_path,
                               const vx_classifier_spec *spec,
                               vx_classifier **out) {
    if (out != nullptr) *out = nullptr;
    if (backbone_path == nullptr || head_path == nullptr || spec == nullptr || out == nullptr) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    if (spec->execution_provider != 0) {
        // Reserved, not silently ignored: a caller asking for CoreML must be told it
        // did not happen rather than quietly getting CPU and wondering about the timing.
        return VX_ERROR_INVALID_ARGUMENT;
    }
    try {
        *out = new vx_classifier(backbone_path, head_path, *spec);
        return VX_OK;
    } catch (const std::exception &error) {
        lastCreateError() = error.what();
        return VX_ERROR_MODEL;
    }
}

void vx_classifier_destroy(vx_classifier *classifier) {
    delete classifier;
}

const char *vx_classifier_last_error(const vx_classifier *classifier) {
    if (classifier == nullptr) {
        return lastCreateError().c_str();
    }
    // Held on the object so the pointer stays valid until the next failing call.
    static thread_local std::string message;
    message = classifier->classifier.lastError();
    return message.c_str();
}

vx_classifier_stats vx_classifier_get_stats(const vx_classifier *classifier) {
    vx_classifier_stats stats;
    stats.backbone_runs = classifier != nullptr ? classifier->classifier.backboneRuns() : 0;
    stats.head_runs = classifier != nullptr ? classifier->classifier.headRuns() : 0;
    return stats;
}

vx_status vx_engine_classify_regions(vx_engine *engine,
                                     vx_classifier *classifier,
                                     const vx_image_view *image,
                                     const vx_region *regions,
                                     int32_t count,
                                     vx_region_labels *out_labels) {
    if (out_labels != nullptr) std::memset(out_labels, 0, sizeof(*out_labels));
    if (engine == nullptr || classifier == nullptr || out_labels == nullptr || !validImage(image)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    if (count < 0 || (count > 0 && regions == nullptr)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        // Nothing to name is not a failure: a blank screenshot proposes no regions.
        return VX_OK;
    }

    try {
        cv::Mat mat;
        if (!wrap(*image, mat)) {
            return VX_ERROR_UNSUPPORTED_FORMAT;
        }
        const std::vector<vx_region_label> labels =
            engine->engine.classifyRegions(mat, classifier->classifier, regions, count);

        auto *copied = static_cast<vx_region_label *>(
            std::malloc(sizeof(vx_region_label) * (labels.empty() ? 1 : labels.size())));
        if (copied == nullptr) {
            return VX_ERROR_INTERNAL;
        }
        std::memcpy(copied, labels.data(), sizeof(vx_region_label) * labels.size());
        out_labels->labels = copied;
        out_labels->count = static_cast<int32_t>(labels.size());
        return VX_OK;
    } catch (const std::exception &error) {
        classifier->classifier.setLastError(error.what());
        return VX_ERROR_MODEL;
    }
}

void vx_region_labels_free(vx_region_labels *labels) {
    if (labels == nullptr) return;
    std::free(labels->labels);
    std::memset(labels, 0, sizeof(*labels));
}

// MARK: - Classifier, split at the backbone

vx_status vx_classifier_prepare_image(vx_classifier *classifier,
                                      const vx_image_view *image,
                                      vx_prepared_image **out) {
    if (out != nullptr) *out = nullptr;
    if (classifier == nullptr || out == nullptr || !validImage(image)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    try {
        cv::Mat mat;
        if (!wrap(*image, mat)) {
            return VX_ERROR_UNSUPPORTED_FORMAT;
        }
        *out = new vx_prepared_image{classifier->classifier.prepare(mat)};
        return VX_OK;
    } catch (const std::exception &error) {
        classifier->classifier.setLastError(error.what());
        return VX_ERROR_MODEL;
    }
}

const float *vx_prepared_image_tensor(const vx_prepared_image *prepared,
                                      int32_t *out_height,
                                      int32_t *out_width) {
    if (out_height != nullptr) *out_height = 0;
    if (out_width != nullptr) *out_width = 0;
    if (prepared == nullptr) return nullptr;
    if (out_height != nullptr) *out_height = prepared->prepared.paddedHeight;
    if (out_width != nullptr) *out_width = prepared->prepared.paddedWidth;
    return prepared->prepared.tensor.data();
}

void vx_prepared_image_free(vx_prepared_image *prepared) {
    delete prepared;
}

vx_status vx_classifier_backbone_features(vx_classifier *classifier,
                                          const float *image,
                                          int32_t height,
                                          int32_t width,
                                          vx_feature_map *out_features) {
    if (out_features != nullptr) std::memset(out_features, 0, sizeof(*out_features));
    if (classifier == nullptr || image == nullptr || height <= 0 || width <= 0 ||
        out_features == nullptr) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    try {
        int64_t channels = 0, featureHeight = 0, featureWidth = 0;
        const std::vector<float> values = classifier->classifier.backboneFeatures(
            image, height, width, channels, featureHeight, featureWidth);
        auto *data = static_cast<float *>(
            std::malloc(sizeof(float) * (values.empty() ? 1 : values.size())));
        if (data == nullptr) {
            return VX_ERROR_INTERNAL;
        }
        std::memcpy(data, values.data(), sizeof(float) * values.size());
        out_features->data = data;
        out_features->channels = static_cast<int32_t>(channels);
        out_features->height = static_cast<int32_t>(featureHeight);
        out_features->width = static_cast<int32_t>(featureWidth);
        return VX_OK;
    } catch (const std::exception &error) {
        classifier->classifier.setLastError(error.what());
        return VX_ERROR_MODEL;
    }
}

void vx_feature_map_free(vx_feature_map *features) {
    if (features == nullptr) return;
    std::free(features->data);
    std::memset(features, 0, sizeof(*features));
}

vx_status vx_classifier_classify_features(vx_classifier *classifier,
                                          const vx_prepared_image *prepared,
                                          const float *features,
                                          int32_t channels,
                                          int32_t height,
                                          int32_t width,
                                          const vx_region *regions,
                                          int32_t count,
                                          vx_region_labels *out_labels) {
    if (out_labels != nullptr) std::memset(out_labels, 0, sizeof(*out_labels));
    if (classifier == nullptr || prepared == nullptr || features == nullptr ||
        out_labels == nullptr) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    if (channels <= 0 || height <= 0 || width <= 0) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    if (count < 0 || (count > 0 && regions == nullptr)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return VX_OK;
    }
    try {
        const std::vector<visionax::Classifier::Label> labels =
            classifier->classifier.classifyFeatures(
                prepared->prepared, features, channels, height, width, regions, count);
        auto *copied = static_cast<vx_region_label *>(
            std::malloc(sizeof(vx_region_label) * (labels.empty() ? 1 : labels.size())));
        if (copied == nullptr) {
            return VX_ERROR_INTERNAL;
        }
        for (size_t i = 0; i < labels.size(); ++i) {
            copied[i] = vx_region_label{labels[i].classIndex, labels[i].confidence};
        }
        out_labels->labels = copied;
        out_labels->count = static_cast<int32_t>(labels.size());
        return VX_OK;
    } catch (const std::exception &error) {
        classifier->classifier.setLastError(error.what());
        return VX_ERROR_MODEL;
    }
}

// MARK: - Media controls

vx_status vx_engine_read_media_controls(vx_engine *engine,
                                        const vx_image_view *frame,
                                        const vx_image_view *previous,
                                        vx_media_reading *out_reading) {
    if (out_reading != nullptr) std::memset(out_reading, 0, sizeof(*out_reading));
    if (engine == nullptr || out_reading == nullptr || !validImage(frame)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    // A previous frame that is present but malformed is an argument error rather than a
    // silently dropped witness: the caller believes it handed over two frames.
    if (previous != nullptr && !validImage(previous)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    out_reading->progress_fraction = -1;
    out_reading->motion = -1;
    out_reading->volume_fraction = -1;
    out_reading->center_glyph.glyph = VX_GLYPH_NONE;

    try {
        cv::Mat current;
        if (!wrap(*frame, current)) {
            return VX_ERROR_UNSUPPORTED_FORMAT;
        }
        cv::Mat before;
        if (previous != nullptr && !wrap(*previous, before)) {
            return VX_ERROR_UNSUPPORTED_FORMAT;
        }

        const visionax::MediaReading reading = engine->engine.readMediaControls(current, before);

        out_reading->controls_visible = reading.controlsVisible ? 1 : 0;
        out_reading->bar = toRect(reading.bar);
        out_reading->volume_track = toRect(reading.volumeTrack);
        out_reading->volume_fraction = reading.volumeFraction;
        out_reading->progress = toRect(reading.progress);
        out_reading->progress_fraction = reading.progressFraction;
        out_reading->motion = reading.motion;
        out_reading->center_glyph = toControl(reading.centerGlyph);

        const size_t count = reading.controls.size();
        if (count > 0) {
            auto *controls = static_cast<vx_media_control *>(
                std::malloc(sizeof(vx_media_control) * count));
            if (controls == nullptr) {
                return VX_ERROR_INTERNAL;
            }
            for (size_t i = 0; i < count; ++i) {
                controls[i] = toControl(reading.controls[i]);
            }
            out_reading->controls = controls;
            out_reading->control_count = static_cast<int32_t>(count);
        }
        return VX_OK;
    } catch (const std::exception &) {
        vx_media_reading_free(out_reading);
        return VX_ERROR_INTERNAL;
    }
}

void vx_media_reading_free(vx_media_reading *reading) {
    if (reading == nullptr) return;
    std::free(reading->controls);
    std::memset(reading, 0, sizeof(*reading));
}

vx_status vx_engine_match_icons(vx_engine *engine,
                                const vx_image_view *frame,
                                const vx_rect *rects,
                                int32_t count,
                                vx_icon_match *out_matches) {
    if (engine == nullptr || out_matches == nullptr || !validImage(frame)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    if (count < 0 || (count > 0 && rects == nullptr)) {
        return VX_ERROR_INVALID_ARGUMENT;
    }
    for (int32_t index = 0; index < count; ++index) {
        out_matches[index].glyph = VX_ICON_NONE;
        out_matches[index].confidence = 0;
    }
    if (count == 0) return VX_OK;

    try {
        cv::Mat image;
        if (!wrap(*frame, image)) return VX_ERROR_UNSUPPORTED_FORMAT;
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);

        const cv::Rect bounds(0, 0, gray.cols, gray.rows);
        for (int32_t index = 0; index < count; ++index) {
            const cv::Rect box = cv::Rect(
                rects[index].x, rects[index].y,
                rects[index].width, rects[index].height) & bounds;
            // A BOX TOO SMALL TO HOLD A SHAPE IS NOT AN ICON. Below this the
            // normalization is scaling a handful of pixels up to 32 and calling
            // whatever falls out a match.
            if (box.width < 16 || box.height < 16) continue;
            out_matches[index] = visionax::matchIcon(
                visionax::iconSilhouette(gray(box)));
        }
        return VX_OK;
    } catch (const std::exception &) {
        return VX_ERROR_INTERNAL;
    }
}

}  // extern "C"
