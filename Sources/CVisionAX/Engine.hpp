//
//  Engine.hpp
//  CVisionAX
//
//  WHAT: The engine class. Three capabilities (detectRegions, classifyRegions,
//        readMediaControls); the seam for more.
//  IN:   visionax.cpp (extern "C" facade)
//  OUT:  CannyRegionDetector, RegionTreeBuilder, Classifier, MediaControls
//  PIN:  Takes a GRAY8 (CV_8UC1) or BGRA8 (CV_8UC4) cv::Mat — the facade normalizes
//        RGBA to BGRA before it gets here. Holds no per-call state.
//

#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "MediaControls.hpp"
#include "visionax.h"

namespace visionax {

class Classifier;

/// One node of the flat pre-order tree, before it is copied into vx_region.
struct Region {
    uint32_t id = 0;
    int32_t parent = -1;
    int32_t depth = 0;
    cv::Rect rect;
    int32_t childCount = 0;
};

struct RegionTree {
    std::vector<Region> regions;
    bool truncated = false;
    int32_t contourCount = 0;
};

class Engine {
public:
    Engine() = default;

    /// Canny edges → bounding boxes → containment tree. `edgesOut`, when given,
    /// receives the raw Canny map (CV_8UC1). Throws cv::Exception / std::exception
    /// on failure; the facade maps those to vx_status.
    RegionTree detectRegions(const cv::Mat &image,
                             const vx_canny_options &options,
                             cv::Mat *edgesOut) const;

    /// Names each of `regions` with a class index and a confidence, in order. The
    /// engine owns no model — the classifier is passed in, so one engine can serve
    /// several models and a model can outlive any single call.
    std::vector<vx_region_label> classifyRegions(const cv::Mat &image,
                                                 const Classifier &classifier,
                                                 const vx_region *regions,
                                                 int32_t count) const;

    /// A player's transport, out of the player's own pixels. `previous` may be empty.
    /// Needs no model: the shapes are drawn, not learned. See MediaControls.hpp.
    MediaReading readMediaControls(const cv::Mat &frame, const cv::Mat &previous) const;
};

}  // namespace visionax
