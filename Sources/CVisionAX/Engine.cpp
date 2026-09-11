//
//  Engine.cpp
//  CVisionAX
//
//  WHAT: The engine's capabilities, each one wiring stages together and nothing else.
//  IN:   visionax.cpp
//  OUT:  CannyRegionDetector.hpp, RegionTreeBuilder.hpp, Classifier.hpp, MediaControls.hpp
//

#include "Engine.hpp"

#include "CannyRegionDetector.hpp"
#include "Classifier.hpp"
#include "MediaControls.hpp"
#include "RegionTreeBuilder.hpp"

namespace visionax {

RegionTree Engine::detectRegions(const cv::Mat &image,
                                 const vx_canny_options &options,
                                 cv::Mat *edgesOut) const {
    CannyRegionResult detected = detectCannyRegions(image, options);
    if (edgesOut != nullptr) {
        *edgesOut = detected.edges;
    }
    const cv::Rect bounds(0, 0, image.cols, image.rows);
    RegionTree tree = buildRegionTree(bounds, detected.rects, options);
    tree.contourCount = detected.contourCount;
    // Boxes dropped by the keep cap are the same budget the walk enforces, so they
    // are reported the same way.
    tree.truncated = tree.truncated || detected.capped;
    return tree;
}

std::vector<vx_region_label> Engine::classifyRegions(const cv::Mat &image,
                                                     const Classifier &classifier,
                                                     const vx_region *regions,
                                                     int32_t count) const {
    const std::vector<Classifier::Label> labels = classifier.classify(image, regions, count);
    std::vector<vx_region_label> out;
    out.reserve(labels.size());
    for (const Classifier::Label &label : labels) {
        out.push_back(vx_region_label{label.classIndex, label.confidence});
    }
    return out;
}

MediaReading Engine::readMediaControls(const cv::Mat &frame, const cv::Mat &previous) const {
    return visionax::readMediaControls(frame, previous);
}

}  // namespace visionax
