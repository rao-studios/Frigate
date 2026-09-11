//
//  IconGlyphs.cpp
//  CVisionAX
//
//  WHAT: Twenty-two interface icons, drawn once and compared by silhouette overlap.
//  IN:   IconGlyphs.hpp
//  OUT:  vx_engine_match_icons
//  PIN:  THE SAME COMPARISON THE TRANSPORT USES, deliberately: crop to the ink, scale by
//        the longer side, overlap the masks. Stroke weight, colour, corner radius and
//        button size all fall out of the comparison, which is what lets one drawing
//        stand for every version of a magnifier anybody has shipped.
//        POLARITY IS DECIDED PER BOX. A toolbar draws dark icons on light chrome and a
//        player draws light ones on dark; the glyph is whichever side of Otsu's split
//        is the MINORITY, because an icon is a small mark on a large field.
//        A LOW SCORE IS AN HONEST "I DON'T KNOW". Every shape in here scores something
//        against every other, so the caller's threshold is the whole difference between
//        a name and a guess — see the floor in IconNamer.swift.
//

#include "IconGlyphs.hpp"
#include "MediaGlyphs.hpp"

#include <map>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace visionax {
namespace {

constexpr int S = kGlyphMaskSide;

cv::Mat blank() { return cv::Mat::zeros(S, S, CV_8UC1); }

void line(cv::Mat &mask, cv::Point a, cv::Point b, int thickness = 3) {
    cv::line(mask, a, b, cv::Scalar(255), thickness);
}

void bar(cv::Mat &mask, int x, int y, int w, int h) {
    cv::rectangle(mask, cv::Rect(x, y, w, h), cv::Scalar(255), cv::FILLED);
}

void ring(cv::Mat &mask, cv::Point centre, int radius, int thickness) {
    cv::circle(mask, centre, radius, cv::Scalar(255), thickness);
}

void dot(cv::Mat &mask, cv::Point centre, int radius) {
    cv::circle(mask, centre, radius, cv::Scalar(255), cv::FILLED);
}

/// A chevron pointing in one of four directions.
void chevron(cv::Mat &mask, int dx, int dy) {
    const int c = S / 2;
    const int arm = 9;
    if (dx != 0) {
        const int tip = c + dx * 5;
        line(mask, cv::Point(tip - dx * arm, c - arm), cv::Point(tip, c));
        line(mask, cv::Point(tip - dx * arm, c + arm), cv::Point(tip, c));
    } else {
        const int tip = c + dy * 5;
        line(mask, cv::Point(c - arm, tip - dy * arm), cv::Point(c, tip));
        line(mask, cv::Point(c + arm, tip - dy * arm), cv::Point(c, tip));
    }
}

void star(cv::Mat &mask) {
    std::vector<cv::Point> points;
    for (int index = 0; index < 10; ++index) {
        const double angle = -CV_PI / 2 + index * CV_PI / 5;
        const double radius = (index % 2 == 0) ? 14.0 : 6.0;
        points.emplace_back(
            static_cast<int>(S / 2 + radius * std::cos(angle)),
            static_cast<int>(S / 2 + radius * std::sin(angle)));
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{points}, cv::Scalar(255));
}

void heart(cv::Mat &mask) {
    dot(mask, cv::Point(11, 12), 6);
    dot(mask, cv::Point(21, 12), 6);
    const std::vector<cv::Point> points{
        cv::Point(5, 13), cv::Point(27, 13), cv::Point(16, 27)};
    cv::fillConvexPoly(mask, points, cv::Scalar(255));
}

cv::Mat drawIcon(vx_icon_glyph glyph) {
    cv::Mat mask = blank();
    switch (glyph) {
    case VX_ICON_SEARCH:
        ring(mask, cv::Point(13, 13), 8, 3);
        line(mask, cv::Point(19, 19), cv::Point(27, 27), 4);
        break;
    case VX_ICON_CLOSE:
        line(mask, cv::Point(6, 6), cv::Point(26, 26), 4);
        line(mask, cv::Point(26, 6), cv::Point(6, 26), 4);
        break;
    case VX_ICON_MENU:
        bar(mask, 5, 8, 22, 3);
        bar(mask, 5, 15, 22, 3);
        bar(mask, 5, 22, 22, 3);
        break;
    case VX_ICON_BACK:
        chevron(mask, -1, 0);
        break;
    case VX_ICON_FORWARD:
        chevron(mask, 1, 0);
        break;
    case VX_ICON_UP:
        chevron(mask, 0, -1);
        break;
    case VX_ICON_DOWN:
        chevron(mask, 0, 1);
        break;
    case VX_ICON_ADD:
        bar(mask, 14, 5, 4, 22);
        bar(mask, 5, 14, 22, 4);
        break;
    case VX_ICON_MORE:
        // Three dots. Vertical and horizontal arrangements normalize to the same
        // silhouette family, which is why one drawing answers for both.
        dot(mask, cv::Point(16, 6), 3);
        dot(mask, cv::Point(16, 16), 3);
        dot(mask, cv::Point(16, 26), 3);
        break;
    case VX_ICON_SHARE:
        dot(mask, cv::Point(23, 7), 4);
        dot(mask, cv::Point(9, 16), 4);
        dot(mask, cv::Point(23, 25), 4);
        line(mask, cv::Point(12, 14), cv::Point(21, 9), 2);
        line(mask, cv::Point(12, 18), cv::Point(21, 23), 2);
        break;
    case VX_ICON_MICROPHONE:
        cv::rectangle(mask, cv::Rect(12, 4, 8, 14), cv::Scalar(255), cv::FILLED);
        cv::ellipse(mask, cv::Point(16, 18), cv::Size(8, 7), 0, 0, 180, cv::Scalar(255), 3);
        bar(mask, 15, 24, 3, 5);
        break;
    case VX_ICON_NOTIFICATIONS:
        cv::ellipse(mask, cv::Point(16, 16), cv::Size(9, 10), 0, 180, 360, cv::Scalar(255), cv::FILLED);
        bar(mask, 6, 20, 20, 3);
        dot(mask, cv::Point(16, 26), 3);
        break;
    case VX_ICON_ACCOUNT:
        dot(mask, cv::Point(16, 11), 6);
        cv::ellipse(mask, cv::Point(16, 30), cv::Size(11, 10), 0, 180, 360, cv::Scalar(255), cv::FILLED);
        break;
    case VX_ICON_STAR:
        star(mask);
        break;
    case VX_ICON_HEART:
        heart(mask);
        break;
    case VX_ICON_DELETE:
        bar(mask, 8, 9, 16, 3);
        bar(mask, 13, 5, 6, 3);
        cv::rectangle(mask, cv::Rect(10, 12, 12, 15), cv::Scalar(255), 3);
        break;
    case VX_ICON_DONE:
        line(mask, cv::Point(6, 17), cv::Point(13, 24), 4);
        line(mask, cv::Point(13, 24), cv::Point(26, 8), 4);
        break;
    case VX_ICON_SETTINGS:
        // A GEAR, NOT TWO RINGS. Concentric circles are a target or a record; the shape
        // interfaces actually draw for settings has teeth, and the teeth are most of
        // what distinguishes it from every other round icon in the bank.
        {
            const int c = S / 2;
            for (int tooth = 0; tooth < 8; ++tooth) {
                const double angle = tooth * CV_PI / 4;
                const int x = static_cast<int>(c + 12 * std::cos(angle));
                const int y = static_cast<int>(c + 12 * std::sin(angle));
                dot(mask, cv::Point(x, y), 4);
            }
            dot(mask, cv::Point(c, c), 10);
            dot(mask, cv::Point(c, c), 4);
            cv::circle(mask, cv::Point(c, c), 4, cv::Scalar(0), cv::FILLED);
        }
        break;
    case VX_ICON_HOME:
        {
            const std::vector<cv::Point> roof{
                cv::Point(16, 4), cv::Point(28, 15), cv::Point(4, 15)};
            cv::fillConvexPoly(mask, roof, cv::Scalar(255));
            cv::rectangle(mask, cv::Rect(8, 15, 16, 13), cv::Scalar(255), 3);
        }
        break;
    case VX_ICON_FILTER:
        bar(mask, 4, 7, 24, 3);
        bar(mask, 8, 15, 16, 3);
        bar(mask, 12, 23, 8, 3);
        break;
    case VX_ICON_CART:
        cv::rectangle(mask, cv::Rect(7, 10, 18, 12), cv::Scalar(255), 3);
        dot(mask, cv::Point(12, 27), 3);
        dot(mask, cv::Point(21, 27), 3);
        line(mask, cv::Point(3, 5), cv::Point(8, 10), 3);
        break;
    case VX_ICON_DOWNLOAD:
        bar(mask, 14, 4, 4, 14);
        line(mask, cv::Point(8, 13), cv::Point(16, 21), 4);
        line(mask, cv::Point(24, 13), cv::Point(16, 21), 4);
        bar(mask, 5, 25, 22, 3);
        break;
    case VX_ICON_NONE:
        break;
    }
    return mask;
}

/// A silhouette softened into a field, so a comparison is about SHAPE rather than
/// about how thickly the shape was drawn.
///
/// PIN: OVERLAP ALONE CANNOT DO THIS JOB, and that was measured on this very bank: a
/// magnifier drawn with a 3px stroke scored 0.28 against a magnifier drawn with a 4px
/// stroke, below a cross at 0.30, and a checkmark read as a magnifier. Two icons of the
/// same shape barely overlap when their strokes differ by a pixel, because a stroke is
/// mostly edge. Blurring turns each mask into a field that falls off with distance, so
/// "the ink is nearly in the same places" scores high and "the ink is somewhere else"
/// still scores low — which is what the eye is doing when it calls both of them a
/// magnifier.
cv::Mat softened(const cv::Mat &normalized) {
    cv::Mat field;
    normalized.convertTo(field, CV_32F, 1.0 / 255.0);
    cv::GaussianBlur(field, field, cv::Size(0, 0), 2.5);
    // Zero-mean and unit-norm, so the score below is a correlation and not a brightness
    // contest between a dense glyph and a sparse one.
    cv::Scalar mean = cv::mean(field);
    field -= mean[0];
    const double norm = cv::norm(field);
    if (norm > 1e-6) field /= norm;
    return field;
}

struct Bank {
    std::map<int, cv::Mat> masks;
    std::map<int, cv::Mat> normalized;
    std::vector<vx_icon_glyph> vocabulary;
};

const Bank &bank() {
    static const Bank shared = [] {
        Bank built;
        built.vocabulary = {
            VX_ICON_SEARCH, VX_ICON_CLOSE, VX_ICON_MENU, VX_ICON_BACK, VX_ICON_FORWARD,
            VX_ICON_UP, VX_ICON_DOWN, VX_ICON_ADD, VX_ICON_MORE, VX_ICON_SHARE,
            VX_ICON_MICROPHONE, VX_ICON_NOTIFICATIONS, VX_ICON_ACCOUNT, VX_ICON_STAR,
            VX_ICON_HEART, VX_ICON_DELETE, VX_ICON_DONE, VX_ICON_SETTINGS, VX_ICON_HOME,
            VX_ICON_FILTER, VX_ICON_CART, VX_ICON_DOWNLOAD};
        for (vx_icon_glyph glyph : built.vocabulary) {
            const cv::Mat drawn = drawIcon(glyph);
            built.masks[static_cast<int>(glyph)] = drawn;
            built.normalized[static_cast<int>(glyph)] = softened(normalizedInk(drawn));
        }
        built.masks[static_cast<int>(VX_ICON_NONE)] = blank();
        built.normalized[static_cast<int>(VX_ICON_NONE)] = softened(blank());
        return built;
    }();
    return shared;
}

}  // namespace

const cv::Mat &iconMask(vx_icon_glyph glyph) {
    const Bank &shared = bank();
    auto found = shared.masks.find(static_cast<int>(glyph));
    if (found != shared.masks.end()) return found->second;
    return shared.masks.at(static_cast<int>(VX_ICON_NONE));
}

const std::vector<vx_icon_glyph> &iconVocabulary() { return bank().vocabulary; }

cv::Mat iconSilhouette(const cv::Mat &grayCrop) {
    if (grayCrop.empty() || grayCrop.type() != CV_8UC1) return cv::Mat();
    // A BORDER OF THE BUTTON IS NOT THE ICON. Buttons are drawn with rounded outlines
    // and focus rings; trimming a tenth off each side leaves the mark in the middle.
    const int inset = std::max(1, std::min(grayCrop.cols, grayCrop.rows) / 10);
    cv::Rect inner(inset, inset, grayCrop.cols - 2 * inset, grayCrop.rows - 2 * inset);
    if (inner.width < 4 || inner.height < 4) inner = cv::Rect(0, 0, grayCrop.cols, grayCrop.rows);

    cv::Mat binary;
    cv::threshold(grayCrop(inner), binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // WHICHEVER SIDE IS THE MINORITY IS THE MARK. An icon is a small shape on a large
    // field, so this is the same rule for a dark glyph on light chrome and a light one
    // on a dark player.
    const double lit = cv::countNonZero(binary);
    const double area = static_cast<double>(binary.rows) * binary.cols;
    if (lit > area / 2) cv::bitwise_not(binary, binary);
    return binary;
}

vx_icon_match matchIcon(const cv::Mat &candidate) {
    vx_icon_match best{};
    best.glyph = VX_ICON_NONE;
    best.confidence = 0;
    if (candidate.empty() || candidate.type() != CV_8UC1) return best;

    const cv::Mat normalized = normalizedInk(candidate);
    if (normalized.empty()) return best;
    if (cv::countNonZero(normalized) <= 0) return best;
    const cv::Mat field = softened(normalized);

    const Bank &shared = bank();
    for (vx_icon_glyph glyph : shared.vocabulary) {
        const cv::Mat &mask = shared.normalized.at(static_cast<int>(glyph));
        // Both are zero-mean and unit-norm, so the dot product IS the correlation:
        // 1 is the same shape, 0 is unrelated, and negative is the opposite.
        const double score = field.dot(mask);
        if (score > best.confidence) {
            best.confidence = score;
            best.glyph = glyph;
        }
    }
    return best;
}

}  // namespace visionax
