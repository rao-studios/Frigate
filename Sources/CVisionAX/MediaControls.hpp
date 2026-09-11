//
//  MediaControls.hpp
//  CVisionAX
//
//  WHAT: A player's transport, read out of the player's own pixels.
//  IN:   Engine.cpp
//  OUT:  MediaReading — bar, progress, controls, motion
//  PIN:  THE LAYOUT IS THE INVARIANT, NOT THE ARTWORK. Every video player built in the
//        last decade draws the same thing: a thin progress track across the bottom of
//        the picture, and a row of glyphs under it. This file finds THAT, which is why
//        it works on a player it has never seen. Nothing here knows the name of a site.
//        NOTHING HERE DECIDES "PLAYING". It measures motion, the fraction, and the
//        glyphs, and hands all three up. A detector that collapsed those into a verdict
//        would throw away the evidence the caller needs when they disagree.
//

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "visionax.h"

namespace visionax {

struct MediaControl {
    cv::Rect frame;
    vx_media_glyph glyph = VX_GLYPH_NONE;
    double confidence = 0;
};

struct MediaReading {
    bool controlsVisible = false;
    cv::Rect bar;
    cv::Rect progress;
    double progressFraction = -1;
    double motion = -1;
    std::vector<MediaControl> controls;
    MediaControl centerGlyph;
    /// The short track a player reveals beside its volume control, when one is showing.
    /// Zero-sized when none was found — which is the usual case, since it exists only
    /// while the pointer is over the control.
    cv::Rect volumeTrack;
    double volumeFraction = -1;
};

/// `frame` is CV_8UC1 or CV_8UC4 (BGRA), already cropped to the player. `previous`
/// may be empty, in which case `motion` stays -1.
MediaReading readMediaControls(const cv::Mat &frame, const cv::Mat &previous);

/// How much two views of the same region differ, 0..1, on a coarse gray grid.
///
/// Coarse ON PURPOSE: the question is whether the PICTURE moved, and at full
/// resolution antialiasing and video compression noise answer yes every time.
double frameMotion(const cv::Mat &a, const cv::Mat &b);

}  // namespace visionax
