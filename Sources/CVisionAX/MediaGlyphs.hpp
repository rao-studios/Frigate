//
//  MediaGlyphs.hpp
//  CVisionAX
//
//  WHAT: The transport glyph vocabulary, drawn rather than loaded.
//  IN:   MediaControls.cpp
//  OUT:  a binary mask per vx_media_glyph, at one canonical size
//  PIN:  DRAWN, NOT SHIPPED. A player's play button is a triangle and its pause button
//        is two bars in every player anyone has built; a bundle of PNGs would add a
//        resource, a git-lfs object and a "the model is missing" failure mode to
//        express the same twelve shapes. Drawing them also means the bank exists on a
//        machine with no assets at all, which is what makes the media lane work with
//        no classifier model present.
//        These are SILHOUETTES compared by overlap, not photographs compared by pixels:
//        a candidate is scaled to the same canonical box first, so stroke weight,
//        colour and size never enter the comparison.
//

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "visionax.h"

namespace visionax {

/// The side of every glyph mask, in pixels. Candidates are resized to this before
/// they are compared, so the number is a comparison resolution and nothing else.
constexpr int kGlyphMaskSide = 32;

/// One glyph's silhouette: CV_8UC1, 255 inside the shape and 0 outside.
const cv::Mat &glyphMask(vx_media_glyph glyph);

/// Every glyph the bank can answer with, in vocabulary order. VX_GLYPH_NONE is not in it.
const std::vector<vx_media_glyph> &glyphVocabulary();

/// The best match for a candidate silhouette, by mask overlap (intersection over union).
///
/// `candidate` is CV_8UC1, any size, 0 or 255. Returns VX_GLYPH_NONE with confidence 0
/// when the candidate is empty. The confidence IS the overlap — a caller comparing it
/// against a threshold is asking "how much of this shape is that shape".
vx_media_control matchGlyph(const cv::Mat &candidate);

/// A silhouette reduced to shape alone: cropped to its ink, scaled so its LONGER side
/// fills the canonical box, and centred.
///
/// PIN: SHARED WITH THE ICON BANK, so both compare shapes the same way. Aspect is
/// preserved and padding is not — stretching both sides to a square scores every shape
/// alike, which was measured as a pause pair matching `miniplayer`.
cv::Mat normalizedInk(const cv::Mat &silhouette);

}  // namespace visionax
