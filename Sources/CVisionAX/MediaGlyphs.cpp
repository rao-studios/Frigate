//
//  MediaGlyphs.cpp
//  CVisionAX
//
//  WHAT: Each glyph drawn once into a static mask, and the overlap match against them.
//  IN:   MediaGlyphs.hpp
//  OUT:  MediaControls.cpp
//

#include "MediaGlyphs.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

namespace visionax {
namespace {

constexpr int S = kGlyphMaskSide;

cv::Mat blank() { return cv::Mat::zeros(S, S, CV_8UC1); }

void triangle(cv::Mat &mask, int left, int right, int top, int bottom) {
    const std::vector<cv::Point> points{
        cv::Point(left, top), cv::Point(left, bottom),
        cv::Point(right, (top + bottom) / 2)};
    cv::fillConvexPoly(mask, points, cv::Scalar(255));
}

void triangleLeft(cv::Mat &mask, int left, int right, int top, int bottom) {
    const std::vector<cv::Point> points{
        cv::Point(right, top), cv::Point(right, bottom),
        cv::Point(left, (top + bottom) / 2)};
    cv::fillConvexPoly(mask, points, cv::Scalar(255));
}

void box(cv::Mat &mask, int x, int y, int w, int h) {
    cv::rectangle(mask, cv::Rect(x, y, w, h), cv::Scalar(255), cv::FILLED);
}

void outline(cv::Mat &mask, cv::Rect rect, int thickness) {
    cv::rectangle(mask, rect, cv::Scalar(255), thickness);
}

/// A speaker cone: the small box at the pivot plus the flared triangle.
void speaker(cv::Mat &mask) {
    box(mask, 4, 13, 5, 6);
    const std::vector<cv::Point> points{
        cv::Point(9, 16), cv::Point(16, 6), cv::Point(16, 26), cv::Point(9, 19)};
    cv::fillConvexPoly(mask, points, cv::Scalar(255));
}

/// Four corner brackets. `inward` draws the exit-fullscreen arrangement, whose
/// corners point at the middle instead of away from it.
void corners(cv::Mat &mask, bool inward) {
    const int arm = 8;
    const int t = 3;
    const int lo = 5;
    const int hi = S - 5;
    if (!inward) {
        box(mask, lo, lo, arm, t);           box(mask, lo, lo, t, arm);
        box(mask, hi - arm, lo, arm, t);     box(mask, hi - t, lo, t, arm);
        box(mask, lo, hi - t, arm, t);       box(mask, lo, hi - arm, t, arm);
        box(mask, hi - arm, hi - t, arm, t); box(mask, hi - t, hi - arm, t, arm);
    } else {
        const int mid = S / 2;
        box(mask, mid - arm, mid - 5, arm, t);  box(mask, mid - t, mid - 5, t, arm - 2);
        box(mask, mid, mid - 5, arm, t);        box(mask, mid, mid - 5, t, arm - 2);
        box(mask, mid - arm, mid + 4, arm, t);  box(mask, mid - t, mid + 2, t, arm - 2);
        box(mask, mid, mid + 4, arm, t);        box(mask, mid, mid + 2, t, arm - 2);
    }
}

}  // namespace

/// A silhouette reduced to shape alone: cropped to its ink, scaled so its LONGER side
/// fills the canonical box, and centred.
///
/// PIN: ASPECT IS PRESERVED, PADDING IS NOT. A candidate arrives as a tight crop around
/// the ink while a drawn template has margins, so stretching both to a square would
/// compare a triangle against a triangle-shaped hole and score everything alike — that
/// was measured: a pause pair matched `miniplayer`. Scaling by the longer side keeps a
/// tall play triangle distinguishable from a square fullscreen bracket, which is the
/// discrimination the whole lane rests on.
cv::Mat normalizedInk(const cv::Mat &silhouette) {
    cv::Mat binary;
    if (silhouette.type() != CV_8UC1) return cv::Mat();
    cv::threshold(silhouette, binary, 96, 255, cv::THRESH_BINARY);
    const cv::Rect ink = cv::boundingRect(binary);
    if (ink.width < 1 || ink.height < 1) return cv::Mat();

    const double scale = static_cast<double>(kGlyphMaskSide) / std::max(ink.width, ink.height);
    const int width = std::max(1, static_cast<int>(std::lround(ink.width * scale)));
    const int height = std::max(1, static_cast<int>(std::lround(ink.height * scale)));
    cv::Mat scaled;
    cv::resize(binary(ink), scaled, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    cv::threshold(scaled, scaled, 96, 255, cv::THRESH_BINARY);

    cv::Mat canvas = cv::Mat::zeros(kGlyphMaskSide, kGlyphMaskSide, CV_8UC1);
    const int x = (kGlyphMaskSide - width) / 2;
    const int y = (kGlyphMaskSide - height) / 2;
    scaled.copyTo(canvas(cv::Rect(x, y, width, height)));
    return canvas;
}

namespace {

cv::Mat drawGlyph(vx_media_glyph glyph) {
    cv::Mat mask = blank();
    switch (glyph) {
    case VX_GLYPH_PLAY:
        triangle(mask, 9, 25, 4, 28);
        break;
    case VX_GLYPH_PAUSE:
        box(mask, 9, 5, 5, 22);
        box(mask, 18, 5, 5, 22);
        break;
    case VX_GLYPH_REPLAY:
        // A ring with a bite out of it, plus the arrow head that closes the loop.
        cv::ellipse(mask, cv::Point(S / 2, S / 2), cv::Size(11, 11), 0, 40, 340,
                    cv::Scalar(255), 4);
        triangle(mask, 18, 27, 2, 12);
        break;
    case VX_GLYPH_VOLUME:
        speaker(mask);
        cv::ellipse(mask, cv::Point(17, 16), cv::Size(5, 7), 0, -60, 60, cv::Scalar(255), 3);
        cv::ellipse(mask, cv::Point(17, 16), cv::Size(10, 13), 0, -60, 60, cv::Scalar(255), 3);
        break;
    case VX_GLYPH_MUTED:
        speaker(mask);
        cv::line(mask, cv::Point(20, 10), cv::Point(29, 22), cv::Scalar(255), 3);
        cv::line(mask, cv::Point(29, 10), cv::Point(20, 22), cv::Scalar(255), 3);
        break;
    case VX_GLYPH_FULLSCREEN:
        corners(mask, false);
        break;
    case VX_GLYPH_EXIT_FULLSCREEN:
        corners(mask, true);
        break;
    case VX_GLYPH_SETTINGS:
        // A gear reads, at this resolution, as a thick ring with a hollow middle.
        cv::circle(mask, cv::Point(S / 2, S / 2), 12, cv::Scalar(255), 6);
        cv::circle(mask, cv::Point(S / 2, S / 2), 3, cv::Scalar(255), cv::FILLED);
        break;
    case VX_GLYPH_CAPTIONS:
        outline(mask, cv::Rect(3, 8, 26, 17), 3);
        box(mask, 8, 15, 7, 3);
        box(mask, 18, 15, 7, 3);
        break;
    case VX_GLYPH_NEXT:
        triangle(mask, 6, 20, 5, 27);
        box(mask, 22, 5, 4, 22);
        break;
    case VX_GLYPH_PREVIOUS:
        triangleLeft(mask, 12, 26, 5, 27);
        box(mask, 6, 5, 4, 22);
        break;
    case VX_GLYPH_THEATER:
        outline(mask, cv::Rect(3, 9, 26, 15), 4);
        break;
    case VX_GLYPH_MINIPLAYER:
        outline(mask, cv::Rect(3, 6, 26, 21), 3);
        box(mask, 16, 16, 11, 9);
        break;
    case VX_GLYPH_NONE:
        break;
    }
    return mask;
}

struct Bank {
    std::vector<vx_media_glyph> vocabulary;
    std::unordered_map<int, cv::Mat> masks;
    std::unordered_map<int, cv::Mat> normalized;
};

const Bank &bank() {
    static Bank shared = [] {
        Bank built;
        built.vocabulary = {
            VX_GLYPH_PLAY, VX_GLYPH_PAUSE, VX_GLYPH_REPLAY, VX_GLYPH_VOLUME,
            VX_GLYPH_MUTED, VX_GLYPH_FULLSCREEN, VX_GLYPH_EXIT_FULLSCREEN,
            VX_GLYPH_SETTINGS, VX_GLYPH_CAPTIONS, VX_GLYPH_NEXT, VX_GLYPH_PREVIOUS,
            VX_GLYPH_THEATER, VX_GLYPH_MINIPLAYER};
        for (vx_media_glyph glyph : built.vocabulary) {
            const cv::Mat drawn = drawGlyph(glyph);
            built.masks[static_cast<int>(glyph)] = drawn;
            // Templates are normalized ONCE, the same way a candidate will be, so the
            // comparison is shape against shape with nothing left of how either was drawn.
            built.normalized[static_cast<int>(glyph)] = normalizedInk(drawn);
        }
        built.masks[static_cast<int>(VX_GLYPH_NONE)] = blank();
        built.normalized[static_cast<int>(VX_GLYPH_NONE)] = blank();
        return built;
    }();
    return shared;
}

}  // namespace

const cv::Mat &glyphMask(vx_media_glyph glyph) {
    const Bank &shared = bank();
    auto found = shared.masks.find(static_cast<int>(glyph));
    if (found != shared.masks.end()) return found->second;
    return shared.masks.at(static_cast<int>(VX_GLYPH_NONE));
}

const std::vector<vx_media_glyph> &glyphVocabulary() { return bank().vocabulary; }

namespace {
const cv::Mat &normalizedGlyphMask(vx_media_glyph glyph) {
    const Bank &shared = bank();
    auto found = shared.normalized.find(static_cast<int>(glyph));
    if (found != shared.normalized.end()) return found->second;
    return shared.normalized.at(static_cast<int>(VX_GLYPH_NONE));
}
}  // namespace

vx_media_control matchGlyph(const cv::Mat &candidate) {
    vx_media_control best{};
    best.glyph = VX_GLYPH_NONE;
    best.confidence = 0;
    if (candidate.empty() || candidate.type() != CV_8UC1) return best;

    const cv::Mat normalized = normalizedInk(candidate);
    if (normalized.empty()) return best;
    const double candidateArea = cv::countNonZero(normalized);
    if (candidateArea <= 0) return best;

    for (vx_media_glyph glyph : glyphVocabulary()) {
        const cv::Mat &mask = normalizedGlyphMask(glyph);
        cv::Mat intersection;
        cv::bitwise_and(normalized, mask, intersection);
        const double overlap = cv::countNonZero(intersection);
        const double maskArea = cv::countNonZero(mask);
        const double unionArea = candidateArea + maskArea - overlap;
        if (unionArea <= 0) continue;
        const double score = overlap / unionArea;
        if (score > best.confidence) {
            best.confidence = score;
            best.glyph = glyph;
        }
    }
    return best;
}

}  // namespace visionax
