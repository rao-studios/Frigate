//
//  CannyRegionDetector.hpp
//  CVisionAX
//
//  WHAT: gray → blur → Canny → close → contours → bounding boxes → filter → dedup.
//  IN:   Engine.cpp
//  OUT:  RegionTreeBuilder (rects, largest first)
//  PIN:  findContours' own hierarchy is NOT used — on an edge map it describes edge
//        loops, not UI containment. Nesting is geometric, in RegionTreeBuilder.
//        Keeping stops at `max_nodes` boxes. The tree could not emit more than that
//        anyway, and RegionTreeBuilder's parent search is quadratic in what it is
//        handed — a noisy photo yields tens of thousands of contours, and paying
//        O(n^2) for boxes the budget will throw away costs seconds.
//

#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "visionax.h"

namespace visionax {

struct CannyRegionResult {
    /// Kept boxes, sorted by area descending. Never includes the image bounds.
    std::vector<cv::Rect> rects;
    /// Raw Canny output, CV_8UC1, image-sized.
    cv::Mat edges;
    /// Contours found before size filtering and dedup.
    int32_t contourCount = 0;
    /// The `max_nodes` budget stopped the keep loop before the candidates ran out —
    /// the tree that follows is honest but incomplete.
    bool capped = false;
};

/// `image` is CV_8UC1 or CV_8UC4 (BGRA). Throws on an unsupported layout.
CannyRegionResult detectCannyRegions(const cv::Mat &image, const vx_canny_options &options);

/// Intersection over union of two rects; 0 when either is empty.
double intersectionOverUnion(const cv::Rect &a, const cv::Rect &b);

}  // namespace visionax
