//
//  ClassifierPreprocess.hpp
//  CVisionAX
//
//  WHAT: Image and boxes into the exact tensors the exported graphs were trained on.
//  IN:   Classifier.cpp
//  OUT:  PreparedImage (CHW float) + RoIAlign rois [N,5]
//  PIN:  THE TWIN OF Training/visionax_train/preprocess.py. Read them side by side
//        before changing either. Divergence here does not crash and does not log — it
//        costs accuracy that looks like a bad model.
//

#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "visionax.h"

namespace visionax {

struct PreparedImage {
    /// [1, 3, paddedHeight, paddedWidth], normalized, row-major CHW.
    std::vector<float> tensor;
    int resizedWidth = 0;
    int resizedHeight = 0;
    int paddedWidth = 0;
    int paddedHeight = 0;
    /// Per axis, because width and height round independently.
    double scaleX = 1;
    double scaleY = 1;
};

/// `image` is CV_8UC1 or CV_8UC4 (BGRA). Throws on any other layout.
PreparedImage prepareImage(const cv::Mat &image, const vx_classifier_spec &spec);

/// Image-pixel boxes → [N,5] rois (batch index + xyxy) in RESIZED image space.
/// Neither rounded nor clipped: RoIAlign samples sub-pixel and the head clamps context.
std::vector<float> prepareBoxes(const vx_region *regions, int32_t count, const PreparedImage &prepared);

/// floor(x + 0.5). Spelled out because Python's round() is banker's rounding and
/// std::lround is not — and both sides must pick the same pixel.
int roundHalfUp(double value);

}  // namespace visionax
