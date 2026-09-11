//
//  CannyRegionDetector.cpp
//  CVisionAX
//
//  WHAT: The Canny pipeline, one stage per statement.
//  IN:   Engine.cpp
//  OUT:  CannyRegionResult
//  PIN:  Dedup keeps the LARGER box. A stroked outline yields two contours (the
//        stroke's outer and inner edge); merge_slack collapses them where IoU alone
//        would not for a small box.
//

#include "CannyRegionDetector.hpp"

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace visionax {

namespace {

int oddKernel(int32_t size) {
    return size | 1;
}

cv::Mat toGray(const cv::Mat &image) {
    cv::Mat gray;
    switch (image.channels()) {
    case 1:
        gray = image;
        break;
    case 3:
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        break;
    case 4:
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        break;
    default:
        throw std::invalid_argument("unsupported channel count");
    }
    return gray;
}

bool isDuplicate(const cv::Rect &candidate, const cv::Rect &kept, const vx_canny_options &options) {
    if (intersectionOverUnion(candidate, kept) >= options.merge_iou) {
        return true;
    }
    const int slack = std::max(0, options.merge_slack);
    return std::abs(candidate.x - kept.x) <= slack
        && std::abs(candidate.y - kept.y) <= slack
        && std::abs(candidate.br().x - kept.br().x) <= slack
        && std::abs(candidate.br().y - kept.br().y) <= slack;
}

}  // namespace

double intersectionOverUnion(const cv::Rect &a, const cv::Rect &b) {
    const double areaA = static_cast<double>(a.area());
    const double areaB = static_cast<double>(b.area());
    if (areaA <= 0 || areaB <= 0) {
        return 0;
    }
    const double intersection = static_cast<double>((a & b).area());
    const double unionArea = areaA + areaB - intersection;
    return unionArea > 0 ? intersection / unionArea : 0;
}

CannyRegionResult detectCannyRegions(const cv::Mat &image, const vx_canny_options &options) {
    CannyRegionResult result;

    cv::Mat gray = toGray(image);

    if (options.blur_kernel >= 3) {
        const int k = oddKernel(options.blur_kernel);
        cv::GaussianBlur(gray, gray, cv::Size(k, k), 0);
    }

    cv::Canny(gray, result.edges, options.low_threshold, options.high_threshold,
              options.aperture_size, false);

    cv::Mat closed = result.edges;
    if (options.close_kernel >= 2) {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(options.close_kernel, options.close_kernel));
        cv::morphologyEx(result.edges, closed, cv::MORPH_CLOSE, kernel);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    result.contourCount = static_cast<int32_t>(contours.size());

    std::vector<cv::Rect> candidates;
    candidates.reserve(contours.size());
    for (const auto &contour : contours) {
        const cv::Rect rect = cv::boundingRect(contour);
        if (rect.width < options.min_width || rect.height < options.min_height) {
            continue;
        }
        candidates.push_back(rect);
    }

    // Largest first; ties in reading order so the output is deterministic.
    std::sort(candidates.begin(), candidates.end(), [](const cv::Rect &a, const cv::Rect &b) {
        if (a.area() != b.area()) return a.area() > b.area();
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    // The image bounds count as already kept, so a full-bleed contour folds into the
    // root instead of becoming a second whole-image node.
    const cv::Rect bounds(0, 0, image.cols, image.rows);
    // The root occupies one of the budget's nodes, so this many boxes can join it.
    const size_t keepLimit =
        options.max_nodes > 1 ? static_cast<size_t>(options.max_nodes - 1) : 0;
    for (const auto &candidate : candidates) {
        if (result.rects.size() >= keepLimit) {
            // Candidates are largest-first, so what is dropped here is the smallest
            // of them — the specks, not the structure.
            result.capped = true;
            break;
        }
        if (isDuplicate(candidate, bounds, options)) {
            continue;
        }
        const bool duplicate = std::any_of(
            result.rects.begin(), result.rects.end(),
            [&](const cv::Rect &kept) { return isDuplicate(candidate, kept, options); });
        if (!duplicate) {
            result.rects.push_back(candidate);
        }
    }

    return result;
}

}  // namespace visionax
