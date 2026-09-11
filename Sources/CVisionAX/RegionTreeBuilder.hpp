//
//  RegionTreeBuilder.hpp
//  CVisionAX
//
//  WHAT: Boxes → containment tree → flat pre-order array under a walk budget.
//  IN:   Engine.cpp (rects from CannyRegionDetector, largest first)
//  OUT:  RegionTree (index 0 = the image bounds root)
//  PIN:  Parent = the SMALLEST kept box that contains the child within
//        containment_slack. Partial overlaps become siblings, never a parent.
//        Budget semantics are Mary's AXTreeWalker: a node at max_depth is kept, its
//        children withheld; exactly max_nodes nodes are emitted before the cut.
//

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "Engine.hpp"
#include "visionax.h"

namespace visionax {

/// `rects` must be sorted by area descending and lie within `bounds`.
RegionTree buildRegionTree(const cv::Rect &bounds,
                           const std::vector<cv::Rect> &rects,
                           const vx_canny_options &options);

}  // namespace visionax
