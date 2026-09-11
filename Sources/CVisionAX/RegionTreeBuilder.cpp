//
//  RegionTreeBuilder.cpp
//  CVisionAX
//
//  WHAT: Containment nesting, reading-order siblings, budgeted pre-order flatten.
//  IN:   Engine.cpp
//  OUT:  RegionTree
//

#include "RegionTreeBuilder.hpp"

#include <algorithm>
#include <functional>

namespace visionax {

namespace {

bool contains(const cv::Rect &outer, const cv::Rect &inner, int slack) {
    return outer.x - slack <= inner.x
        && outer.y - slack <= inner.y
        && outer.br().x + slack >= inner.br().x
        && outer.br().y + slack >= inner.br().y;
}

/// Mary's roster rule: a row band by midY, then left to right. Area then y break ties
/// so the order is total.
bool readingPrecedes(const cv::Rect &a, const cv::Rect &b, int band) {
    const int bandA = (a.y + a.height / 2) / band;
    const int bandB = (b.y + b.height / 2) / band;
    if (bandA != bandB) return bandA < bandB;
    if (a.x != b.x) return a.x < b.x;
    if (a.area() != b.area()) return a.area() > b.area();
    return a.y < b.y;
}

}  // namespace

RegionTree buildRegionTree(const cv::Rect &bounds,
                           const std::vector<cv::Rect> &rects,
                           const vx_canny_options &options) {
    // Slot 0 is the root; every other slot is one kept rect.
    std::vector<cv::Rect> all;
    all.reserve(rects.size() + 1);
    all.push_back(bounds);
    all.insert(all.end(), rects.begin(), rects.end());

    const int slack = std::max(0, options.containment_slack);
    std::vector<std::vector<size_t>> children(all.size());
    for (size_t i = 1; i < all.size(); ++i) {
        size_t parent = 0;
        int parentArea = all[0].area();
        // Larger rects come first, so every candidate container precedes i.
        for (size_t j = 1; j < i; ++j) {
            if (all[j].area() < parentArea && contains(all[j], all[i], slack)) {
                parent = j;
                parentArea = all[j].area();
            }
        }
        children[parent].push_back(i);
    }

    const int band = std::max(1, options.reading_band);
    for (auto &siblings : children) {
        std::sort(siblings.begin(), siblings.end(), [&](size_t a, size_t b) {
            return readingPrecedes(all[a], all[b], band);
        });
    }

    RegionTree tree;
    const int32_t maxNodes = std::max<int32_t>(1, options.max_nodes);
    const int32_t maxDepth = std::max<int32_t>(0, options.max_depth);

    // Pre-order: emit, then children. The output index is the parent link the
    // children carry, so it is known before any child is visited.
    std::function<void(size_t, int32_t, int32_t)> visit = [&](size_t slot, int32_t parentIndex,
                                                               int32_t depth) {
        if (static_cast<int32_t>(tree.regions.size()) >= maxNodes) {
            tree.truncated = true;
            return;
        }
        const int32_t index = static_cast<int32_t>(tree.regions.size());
        Region region;
        region.id = static_cast<uint32_t>(index + 1);
        region.parent = parentIndex;
        region.depth = depth;
        region.rect = all[slot];
        tree.regions.push_back(region);

        if (children[slot].empty()) {
            return;
        }
        if (depth >= maxDepth) {
            tree.truncated = true;
            return;
        }
        for (size_t child : children[slot]) {
            const size_t before = tree.regions.size();
            visit(child, index, depth + 1);
            if (tree.regions.size() > before) {
                tree.regions[index].childCount += 1;
            }
        }
    };
    visit(0, -1, 0);

    return tree;
}

}  // namespace visionax
