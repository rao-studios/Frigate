//
//  ClassifierPreprocess.cpp
//  CVisionAX
//
//  WHAT: resize → pad → normalize → CHW, and boxes into roi rows.
//  IN:   Classifier.cpp
//  OUT:  PreparedImage
//  PIN:  Padding is the MEAN COLOUR, which normalizes to exactly 0 — so the buffer is
//        allocated zeroed and only the real pixels are written. Neither side computes a
//        pad value, so neither side can disagree about it.
//

#include "ClassifierPreprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace visionax {

namespace {

cv::Mat toRGB(const cv::Mat &image) {
    cv::Mat rgb;
    switch (image.channels()) {
    case 1:
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
        break;
    case 4:
        cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
        break;
    default:
        throw std::invalid_argument("classifier input must be GRAY8 or BGRA8");
    }
    return rgb;
}

int padTo(int value, int multiple) {
    if (multiple <= 1) return value;
    return static_cast<int>(std::ceil(static_cast<double>(value) / multiple)) * multiple;
}

}  // namespace

int roundHalfUp(double value) {
    return static_cast<int>(std::floor(value + 0.5));
}

PreparedImage prepareImage(const cv::Mat &image, const vx_classifier_spec &spec) {
    if (image.empty()) {
        throw std::invalid_argument("classifier input image is empty");
    }
    const cv::Mat rgb = toRGB(image);

    const int width = rgb.cols;
    const int height = rgb.rows;
    int newWidth = width;
    int newHeight = height;
    if (spec.long_side > 0 && std::max(width, height) > spec.long_side) {
        const double scale = static_cast<double>(spec.long_side) / std::max(width, height);
        newWidth = std::max(1, roundHalfUp(width * scale));
        newHeight = std::max(1, roundHalfUp(height * scale));
    }

    cv::Mat resized;
    if (newWidth != width || newHeight != height) {
        // INTER_AREA: the only interpolation that averages the pixels it drops, which
        // is what keeps a one-pixel border alive through a 2x downscale.
        cv::resize(rgb, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
    } else {
        resized = rgb;
    }

    PreparedImage prepared;
    prepared.resizedWidth = newWidth;
    prepared.resizedHeight = newHeight;
    prepared.paddedWidth = padTo(newWidth, spec.pad_multiple);
    prepared.paddedHeight = padTo(newHeight, spec.pad_multiple);
    prepared.scaleX = static_cast<double>(newWidth) / width;
    prepared.scaleY = static_cast<double>(newHeight) / height;

    const size_t plane =
        static_cast<size_t>(prepared.paddedWidth) * static_cast<size_t>(prepared.paddedHeight);
    prepared.tensor.assign(plane * 3, 0.0f);

    for (int channel = 0; channel < 3; ++channel) {
        const float mean = spec.mean[channel];
        const float deviation = spec.std[channel] != 0 ? spec.std[channel] : 1.0f;
        float *out = prepared.tensor.data() + plane * channel;
        for (int y = 0; y < newHeight; ++y) {
            const uint8_t *row = resized.ptr<uint8_t>(y);
            float *outRow = out + static_cast<size_t>(y) * prepared.paddedWidth;
            for (int x = 0; x < newWidth; ++x) {
                const float value = static_cast<float>(row[x * 3 + channel]) / 255.0f;
                outRow[x] = (value - mean) / deviation;
            }
        }
    }

    return prepared;
}

std::vector<float> prepareBoxes(const vx_region *regions, int32_t count,
                                const PreparedImage &prepared) {
    std::vector<float> rois(static_cast<size_t>(std::max(0, count)) * 5, 0.0f);
    for (int32_t i = 0; i < count; ++i) {
        const vx_region &region = regions[i];
        float *row = rois.data() + static_cast<size_t>(i) * 5;
        row[0] = 0.0f;  // one image per run, so every roi is in batch 0
        row[1] = static_cast<float>(region.x * prepared.scaleX);
        row[2] = static_cast<float>(region.y * prepared.scaleY);
        row[3] = static_cast<float>((region.x + region.width) * prepared.scaleX);
        row[4] = static_cast<float>((region.y + region.height) * prepared.scaleY);
    }
    return rois;
}

}  // namespace visionax
