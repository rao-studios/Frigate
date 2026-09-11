//
//  IconGlyphs.hpp
//  CVisionAX
//
//  WHAT: The interface-icon vocabulary, drawn rather than loaded.
//  IN:   visionax.cpp (vx_engine_match_icons)
//  OUT:  a binary mask per vx_icon_glyph, at the media bank's canonical size
//  PIN:  DRAWN, LIKE THE TRANSPORT'S. A magnifier, a cross, three bars and a chevron
//        are the same shapes in every interface anybody has built, and drawing them
//        means the bank exists on a machine with no assets and no model at all — which
//        is exactly the machine that most needs a name for an icon-only button.
//        THE BOTTOM RUNG, NOT THE ONLY ONE. A named icon is a convenience; an unnamed
//        one is still a box a person can press by position. Nothing here is allowed to
//        decide whether a control exists.
//

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "visionax.h"

namespace visionax {

/// One icon's silhouette: CV_8UC1, 255 inside the shape and 0 outside.
const cv::Mat &iconMask(vx_icon_glyph glyph);

/// Every icon the bank can answer with, in vocabulary order. VX_ICON_NONE is not in it.
const std::vector<vx_icon_glyph> &iconVocabulary();

/// The best match for a candidate silhouette, by mask overlap.
///
/// Returns VX_ICON_NONE with confidence 0 for an empty candidate. The confidence IS the
/// overlap, so a caller's threshold is answering "how much of this shape is that shape".
vx_icon_match matchIcon(const cv::Mat &candidate);

/// The silhouette of whatever is drawn inside a box: the ink, separated from its
/// background by Otsu's threshold and polarity-corrected so a light glyph on a dark
/// button and a dark glyph on a light one both come out as ink.
cv::Mat iconSilhouette(const cv::Mat &grayCrop);

}  // namespace visionax
