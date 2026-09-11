//
//  MediaControls.cpp
//  CVisionAX
//
//  WHAT: The four measurements — motion, the progress track, the control row, the
//        centered glyph.
//  IN:   MediaControls.hpp
//  OUT:  MediaGlyphs.hpp for naming what the shapes are
//

#include "MediaControls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cstdarg>

#include <opencv2/imgproc.hpp>

#include "MediaGlyphs.hpp"

namespace visionax {
namespace {

/// Why a candidate was refused, to stderr, when VISIONAX_MEDIA_TRACE is set.
///
/// PIN: A TUNING INSTRUMENT, OFF BY DEFAULT AND SILENT. Every real fix in this file came
/// from knowing which gate a plainly visible transport fell out of, and inferring that
/// from a nil result cost more time than printing it ever will.
bool tracing() {
    static const bool on = std::getenv("VISIONAX_MEDIA_TRACE") != nullptr;
    return on;
}

void trace(const char *format, ...) {
    if (!tracing()) return;
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(stderr, format, arguments);
    va_end(arguments);
    std::fputc('\n', stderr);
}

/// How tall a player's control row is, as a share of the track's width. A transport is
/// a thin track with a row of glyphs beneath it, and the row is a small fraction of the
/// player's width at every size anyone ships.
constexpr double kControlRowShare = 0.075;
/// The tallest control row worth searching, in pixels, whatever the arithmetic says.
constexpr int kMaxControlRow = 140;
/// A player's picture is 16:9 often enough to bound the search for a centered glyph and
/// for motion; it is a BOUND, not an assumption about the video.
constexpr double kPlayerAspect = 9.0 / 16.0;
/// Colour distance between neighbouring pixels that counts as an edge along a row.
constexpr int kRowStep = 26;
/// How far a pixel may sit from its run's mean and still belong to it, AFTER the row has
/// been smoothed. Larger than kRowStep because a translucent track carries the picture's
/// own variation even once the texture is gone.
constexpr int kRunTolerance = 72;
/// How well the leading control must match a play, pause or replay silhouette before a
/// candidate counts as a transport at all. Measured: real YouTube buttons score 0.57 to
/// 0.64 against the drawn templates, while the header icons that used to win scored
/// under 0.3 for any transport shape.
constexpr double kTransportFloor = 0.45;
/// How far a track's span may reach back over short runs to pick up a played nub or a
/// scrubber knob. Small: this is the width of a few percent of a bar, not of a picture.
constexpr int kNubReach = 34;
/// Horizontal-only smoothing applied before the row scan.
///
/// PIN: ALONG THE ROW, NEVER ACROSS IT. A progress track is three pixels tall, and
/// blurring vertically would mix it into the picture above and below and erase the very
/// thing being looked for. Along the row it removes the video's texture — measured, on
/// grass showing through a translucent unplayed track, whose per-pixel variation
/// straddled the run tolerance and split one bar into a dozen fragments that were each
/// too short to count.
constexpr int kRowSmoothing = 21;
/// A track row must be at least this share of the picture's width.
constexpr double kMinTrackShare = 0.45;
/// Coarse motion grid.
constexpr int kMotionColumns = 64;
constexpr int kMotionRows = 48;
constexpr int kMotionThreshold = 16;

cv::Mat toGray(const cv::Mat &image) {
    if (image.empty()) return cv::Mat();
    if (image.type() == CV_8UC1) return image;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    return gray;
}

cv::Mat toColor(const cv::Mat &image) {
    if (image.empty()) return cv::Mat();
    if (image.type() == CV_8UC3) return image;
    cv::Mat color;
    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    } else {
        cv::cvtColor(image, color, cv::COLOR_BGRA2BGR);
    }
    return color;
}

cv::Mat grayGrid(const cv::Mat &image, int columns, int rows) {
    cv::Mat gray = toGray(image);
    if (gray.empty()) return cv::Mat();
    cv::Mat grid;
    cv::resize(gray, grid, cv::Size(columns, rows), 0, 0, cv::INTER_AREA);
    return grid;
}

/// Colour distance between two BGR pixels, as a plain sum of channel differences —
/// enough to separate a red played track from a grey unplayed one, which luminance
/// alone does not (pure red and mid grey have nearly the same brightness).
int colorDistance(const cv::Vec3b &a, const cv::Vec3b &b) {
    return std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]) + std::abs(a[2] - b[2]);
}

struct TrackRun {
    int x0 = 0;
    int x1 = 0;
    cv::Vec3b color;
};

struct TrackRow {
    int y = -1;
    int x0 = 0;
    int x1 = 0;
    /// Where the filled part ends. Equal to x0 when nothing has played.
    int split = -1;
    std::vector<TrackRun> runs;
};

/// How much a colour reads as "the played part of a track".
///
/// Saturation first, brightness second, and that order is the measurement: every player
/// draws progress in an accent colour (YouTube red, Vimeo blue) against a desaturated
/// grey, and the ones that do not use white against grey. Both are covered; neither
/// needs the detector to know which site it is looking at.
double filledScore(const cv::Vec3b &color) {
    const int lowest = std::min({color[0], color[1], color[2]});
    const int highest = std::max({color[0], color[1], color[2]});
    const double saturation = highest - lowest;
    const double brightness = (color[0] + color[1] + color[2]) / 3.0;
    return saturation + brightness * 0.5;
}

/// A run's colour, sampled at its middle so an antialiased edge never speaks for it.
cv::Vec3b runColor(const cv::Vec3b *pixels, int x0, int x1) {
    return pixels[std::min(x1 - 1, x0 + (x1 - x0) / 2)];
}

/// One row's verdict: is this a progress track, and where does the filled part end.
///
/// A track row is a short sequence of LONG FLAT RUNS lying next to each other — two for
/// a plain played/unplayed bar, three when the player also draws a buffered region.
/// Everything else in the row is video, and video produces runs a few pixels long.
///
/// PIN: THE TRACK IS FOUND BY ITS RUNS, NOT BY TRIMMING THE ENDS. An earlier version
/// walked in from both edges while the colour matched pixel zero, which assumed the bar
/// was inset on a uniform scrim. Measured against YouTube, the bar overlays the PICTURE:
/// the row begins in grass and ends in grass, the edges contribute their own colour
/// steps, and the whole row was rejected for having too many of them.
TrackRow readTrackRow(const cv::Mat &color, int y, int minRun) {
    TrackRow row;
    row.y = y;
    const int width = color.cols;
    if (width < 32) return row;
    const cv::Vec3b *pixels = color.ptr<cv::Vec3b>(y);

    // Maximal flat spans across the whole row.
    //
    // PIN: EACH PIXEL IS COMPARED TO ITS RUN'S MEAN, NOT TO ITS NEIGHBOUR. A player's
    // unplayed track is TRANSLUCENT — YouTube draws white at about thirty percent over
    // the video — so the picture shows through it and no two neighbouring pixels are
    // identical. Comparing neighbours broke that track into hundreds of two-pixel runs
    // and the bar was never found; comparing against a running mean lets a run drift
    // with the video underneath while a real colour step still ends it.
    std::vector<std::pair<int, int>> runs;
    int start = 0;
    double meanB = pixels[0][0], meanG = pixels[0][1], meanR = pixels[0][2];
    int held = 1;
    for (int x = 1; x <= width; ++x) {
        const bool broke = x == width
            || (std::abs(pixels[x][0] - meanB) + std::abs(pixels[x][1] - meanG)
                    + std::abs(pixels[x][2] - meanR)) > kRunTolerance;
        if (broke) {
            runs.emplace_back(start, x);
            start = x;
            if (x < width) {
                meanB = pixels[x][0];
                meanG = pixels[x][1];
                meanR = pixels[x][2];
                held = 1;
            }
            continue;
        }
        // A slow average: the mean follows the picture under a translucent bar without
        // chasing a single bright blade of grass.
        const double weight = 1.0 / std::min(held + 1, 24);
        meanB += (pixels[x][0] - meanB) * weight;
        meanG += (pixels[x][1] - meanG) * weight;
        meanR += (pixels[x][2] - meanR) * weight;
        ++held;
    }

    // A run long enough to be part of a bar rather than a texture edge.
    const int longEnough = std::max(12, width / 90);
    // How much unexplained width may sit between two long runs.
    //
    // PIN: TIED TO THE SMOOTHING, because smoothing is what creates it. A crisp
    // played/unplayed boundary becomes a ramp as wide as the blur kernel, and that ramp
    // is neither run — measured, on the drawn fixtures, where a five-pixel bridge could
    // not cross an eleven-pixel ramp and every bar came apart into single runs.
    const int bridge = kRowSmoothing + 3;

    int bestStart = -1, bestEnd = -1;
    size_t bestCount = 0;
    size_t index = 0;
    while (index < runs.size()) {
        if (runs[index].second - runs[index].first < longEnough) {
            ++index;
            continue;
        }
        // Extend across adjacent long runs, tolerating short bridges between them.
        size_t last = index;
        size_t count = 1;
        size_t probe = index + 1;
        int gap = 0;
        while (probe < runs.size()) {
            const int length = runs[probe].second - runs[probe].first;
            if (length >= longEnough) {
                if (gap > bridge) break;
                last = probe;
                ++count;
                gap = 0;
            } else {
                gap += length;
                if (gap > bridge) break;
            }
            ++probe;
        }
        const int span = runs[last].second - runs[index].first;
        // TWO TO FOUR RUNS. One is a flat line; five is a picture.
        if (count >= 1 && count <= 4 && span > bestEnd - bestStart) {
            bestStart = runs[index].first;
            bestEnd = runs[last].second;
            bestCount = count;
        }
        index = last + 1;
    }

    // THE ACCENT RUN IS THE FALLBACK, AND OFTEN THE ONLY HONEST SIGNAL.
    //
    // An unplayed track is TRANSLUCENT: it carries whatever the video is doing
    // underneath, so over a bright, changing picture it breaks into fragments that no
    // tolerance fixes without merging the played boundary too. What never varies is the
    // PLAYED portion — it is painted in the site's accent colour, opaque and flat, at
    // every moment of every video. So when the flat-run scan cannot span the bar, the
    // accent run alone defines it, and the bar's extent is taken as symmetric about it:
    // every player insets its track equally at both ends (measured: 12px of 900 in
    // Safari, 12px of 1150 in Chrome).
    if (bestStart < 0 || bestEnd - bestStart < minRun) {
        // AND IT BEGINS AT THE BAR'S LEFT EDGE. Progress starts at the start; a run of
        // the site's accent colour that begins a third of the way across the picture is
        // something IN the video — a flower, a logo, a jersey — and admitting one was
        // measured reporting a bar in the middle of the trees.
        const int leftLimit = std::max(8, static_cast<int>(width * 0.06));
        int accentStart = -1, accentEnd = -1;
        for (const std::pair<int, int> &run : runs) {
            if (run.first > leftLimit) break;
            const int length = run.second - run.first;
            if (length < std::max(6, width / 60)) continue;
            const cv::Vec3b color = runColor(pixels, run.first, run.second);
            const int lowest = std::min({color[0], color[1], color[2]});
            const int highest = std::max({color[0], color[1], color[2]});
            if (highest - lowest < 90) continue;
            accentStart = run.first;
            accentEnd = run.second;
            break;
        }
        if (accentStart < 0) return row;
        const int inset = accentStart;
        const int mirrored = width - inset;
        if (mirrored - inset < minRun) return row;
        row.x0 = inset;
        row.x1 = mirrored;
        row.split = accentEnd;
        row.runs.push_back(TrackRun{accentStart, accentEnd, runColor(pixels, accentStart, accentEnd)});
        row.runs.push_back(TrackRun{accentEnd, mirrored, runColor(pixels, accentEnd, std::min(mirrored, accentEnd + 8))});
        return row;
    }

    row.x0 = bestStart;
    row.x1 = bestEnd;

    // Re-read the runs inside the winning span, so the fill test sees the bar's own
    // colours and not the texture either side of it.
    for (const std::pair<int, int> &run : runs) {
        if (run.first < bestStart || run.second > bestEnd) continue;
        if (run.second - run.first < 3) continue;
        row.runs.push_back(TrackRun{run.first, run.second, runColor(pixels, run.first, run.second)});
    }
    if (row.runs.empty()) return row;
    (void)bestCount;

    // TRIM THE SURROUND. A bar sits on something, and the something shows on BOTH sides
    // of it in the same colour — a dark scrim, a page background. Two end runs that match
    // each other are that surround, not part of the track, and leaving them in makes the
    // denominator too wide and the leading run the wrong one: an unstarted bar read as
    // 98% played. Runs that do NOT match are the picture either side of an overlaid bar,
    // and those stay, because there is no evidence about where the bar's own ends are.
    while (row.runs.size() >= 3
           && colorDistance(row.runs.front().color, row.runs.back().color) <= kRunTolerance) {
        row.runs.erase(row.runs.begin());
        row.runs.pop_back();
    }
    row.x0 = row.runs.front().x0;
    row.x1 = row.runs.back().x1;
    if (row.x1 - row.x0 < minRun) {
        row.runs.clear();
        return row;
    }

    // REACH BACK OVER THE NUB — AFTER the surround has been trimmed, never before.
    //
    // A video nine seconds into ten minutes has a played portion about twelve pixels
    // wide: shorter than this scan's idea of a run, so the span starts after it and the
    // bar reads as one flat unplayed line, which the divider rule then throws away.
    // Measured: a plainly visible YouTube transport reported as no transport at 0:09 of
    // 10:34. Doing this BEFORE the trim instead made the reached-over run look like the
    // left surround and get trimmed straight back off, taking the real accent with it.
    // Only a few pixels are reached over, and only contiguously — this admits a progress
    // nub and a scrubber knob, not the picture beside the bar.
    {
        int reach = row.x0;
        int spent = 0;
        bool grew = true;
        while (grew) {
            grew = false;
            for (const std::pair<int, int> &run : runs) {
                if (run.second != reach) continue;
                const int length = run.second - run.first;
                if (length >= longEnough || spent + length > kNubReach) break;
                row.runs.insert(
                    row.runs.begin(),
                    TrackRun{run.first, run.second, runColor(pixels, run.first, run.second)});
                reach = run.first;
                spent += length;
                grew = true;
                break;
            }
        }
        row.x0 = reach;
    }

    // WHERE THE FILL ENDS: THE FIRST RUN THAT IS VISIBLY FULLER THAN WHAT FOLLOWS IT.
    //
    // Not "the leading run" — a bar drawn over a picture picks up a few pixels of video
    // at its left edge, and calling that the played portion reports every video as
    // unstarted. Not "the brightest run" either — the BUFFERED region is bright grey and
    // sits to the RIGHT of the played one, so brightness alone over-reports. And not
    // "the best of the first three", which was measured breaking the moment the span
    // reached back over a progress nub and pushed the real accent past where the search
    // stopped looking.
    //
    // Scanning left to right for the first run that is fuller than its neighbour handles
    // all of them: the video margin is not fuller than the accent that follows it, the
    // accent IS fuller than the buffer or the empty track beyond it, and a bar with
    // nothing played has no such run at all and correctly reports zero.
    if (row.runs.size() == 1) {
        const cv::Vec3b &only = row.runs.front().color;
        const int lowest = std::min({only[0], only[1], only[2]});
        const int highest = std::max({only[0], only[1], only[2]});
        row.split = (highest - lowest) > 60 ? row.x1 : row.x0;
        return row;
    }
    row.split = row.x0;
    for (size_t i = 0; i + 1 < row.runs.size(); ++i) {
        const TrackRun &here = row.runs[i];
        // An antialiasing sliver is not a played portion.
        if (here.x1 - here.x0 < 4) continue;
        if (filledScore(here.color) > filledScore(row.runs[i + 1].color) + 8) {
            row.split = here.x1;
            break;
        }
    }
    return row;
}

struct TrackCandidate {
    cv::Rect rect;
    double fraction = -1;
    int runs = 0;
    double saturation = 0;
};

/// Every plausible progress track in the region, thickest-grouped first.
///
/// PIN: THE WHOLE REGION IS SCANNED, NOT THE BOTTOM OF IT. A watch page puts the player
/// at the top with a page of comments underneath, so a transport is not usually near
/// the bottom of what the caller handed over — measured against a real YouTube page,
/// where the bar sits at 60% of the page height. Restricting the scan to a bottom band
/// found nothing at all there, which is why this looks everywhere and SCORES instead.
std::vector<TrackCandidate> findTrackCandidates(const cv::Mat &color) {
    const int width = color.cols;
    const int minRun = static_cast<int>(width * kMinTrackShare);
    std::vector<TrackRow> rows;
    for (int y = 0; y < color.rows; ++y) {
        TrackRow row = readTrackRow(color, y, minRun);
        if (row.x1 - row.x0 < minRun || row.runs.empty()) continue;
        // A single flat run is admitted here and judged later. It is what an UNSTARTED
        // track looks like — nothing has played, so there is no second colour — and it
        // is also what every horizontal rule on the page looks like. Only the row of
        // buttons underneath tells those apart, and that test lives in the scoring.
        rows.push_back(row);
    }
    if (rows.empty()) return {};

    const int splitSlack = std::max(3, width / 100);
    std::vector<TrackCandidate> candidates;
    size_t start = 0;
    for (size_t i = 1; i <= rows.size(); ++i) {
        const bool contiguous = i < rows.size()
            && rows[i].y == rows[i - 1].y + 1
            && std::abs(rows[i].split - rows[start].split) <= splitSlack
            && std::abs(rows[i].x0 - rows[start].x0) <= splitSlack
            && std::abs(rows[i].x1 - rows[start].x1) <= splitSlack;
        if (contiguous) continue;

        const size_t length = i - start;
        const TrackRow &middle = rows[start + length / 2];
        const double span = static_cast<double>(middle.x1 - middle.x0);
        if (span > 0) {
            TrackCandidate candidate;
            candidate.rect = cv::Rect(middle.x0, rows[start].y, middle.x1 - middle.x0,
                                      static_cast<int>(length));
            candidate.fraction = std::clamp((middle.split - middle.x0) / span, 0.0, 1.0);
            candidate.runs = static_cast<int>(middle.runs.size());
            double best = 0;
            for (const TrackRun &run : middle.runs) {
                const int lowest = std::min({run.color[0], run.color[1], run.color[2]});
                const int highest = std::max({run.color[0], run.color[1], run.color[2]});
                best = std::max(best, static_cast<double>(highest - lowest));
            }
            candidate.saturation = best;
            candidates.push_back(candidate);
        }
        start = i;
    }
    return candidates;
}

/// The short track a player draws beside its volume control while the pointer is on it.
///
/// PIN: A SEPARATE SCAN, WITH A SEPARATE FLOOR. The progress track is found by requiring
/// a run across a large share of the frame — that requirement is what keeps every
/// horizontal rule on a page from being read as a transport. A volume track is a few
/// dozen pixels wide, so the same scan cannot see it and lowering the floor would make
/// the progress scan useless. Looking only in the band immediately trailing a control
/// already identified as the volume glyph is what makes a short run safe to admit: it is
/// bounded by something the glyph bank recognized.
TrackCandidate findVolumeTrack(const cv::Mat &color, const cv::Rect &control) {
    TrackCandidate none;
    const int reach = control.width * 6;
    const cv::Rect band(
        control.x + control.width,
        std::max(0, control.y - control.height / 4),
        std::min(reach, color.cols - (control.x + control.width)),
        std::min(control.height + control.height / 2, color.rows - control.y));
    if (band.width < control.width || band.height < 3) return none;

    const cv::Mat strip = color(band & cv::Rect(0, 0, color.cols, color.rows));
    const int minRun = std::max(8, control.width);
    std::vector<TrackRow> rows;
    for (int y = 0; y < strip.rows; ++y) {
        TrackRow row = readTrackRow(strip, y, minRun);
        if (row.x1 - row.x0 < minRun || row.runs.empty()) continue;
        rows.push_back(row);
    }
    if (rows.empty()) return none;

    // The thickest contiguous group, which is the track rather than its shadow.
    size_t start = 0, bestStart = 0, bestLength = 0;
    for (size_t i = 1; i <= rows.size(); ++i) {
        const bool contiguous = i < rows.size() && rows[i].y == rows[i - 1].y + 1;
        if (contiguous) continue;
        if (i - start > bestLength) {
            bestLength = i - start;
            bestStart = start;
        }
        start = i;
    }
    if (bestLength == 0) return none;

    const TrackRow &middle = rows[bestStart + bestLength / 2];
    const double span = static_cast<double>(middle.x1 - middle.x0);
    if (span <= 0) return none;
    TrackCandidate candidate;
    candidate.rect = cv::Rect(
        band.x + middle.x0, band.y + rows[bestStart].y,
        middle.x1 - middle.x0, static_cast<int>(bestLength));
    candidate.fraction = std::clamp((middle.split - middle.x0) / span, 0.0, 1.0);
    candidate.runs = static_cast<int>(middle.runs.size());
    return candidate;
}

/// Bright, glyph-shaped blobs inside a strip. Players draw their controls light on a
/// dark scrim, whichever way round the page's own theme runs.
std::vector<cv::Rect> controlBlobs(const cv::Mat &grayStrip) {
    std::vector<cv::Rect> boxes;
    if (grayStrip.empty() || grayStrip.rows < 6) return boxes;

    // BRIGHTER THAN WHAT IS AROUND IT, not brighter than a number.
    //
    // PIN: A player's scrim is a GRADIENT over whatever the video is showing, so one
    // threshold across the strip cannot work: set it for a dark picture and a bright one
    // turns the grass into blobs that swallow the buttons; set it for a bright picture
    // and the buttons over a dark one disappear. Measured on Chrome over a sunlit
    // meadow, where a global threshold merged the play triangle into the background and
    // the whole transport was refused. An adaptive threshold asks the only question that
    // holds in both cases — is this pixel lighter than its neighbourhood — and a glyph
    // always is.
    // Two conditions, both required: lighter than the neighbourhood (which survives the
    // gradient) AND light in absolute terms (which stops the adaptive pass turning every
    // slightly-less-dark patch of scrim into a blob).
    cv::Mat mask;
    const int block = std::max(15, (grayStrip.rows / 2) * 2 + 1);
    cv::adaptiveThreshold(
        grayStrip, mask, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, block, -18);
    cv::Scalar meanValue, stddevValue;
    cv::meanStdDev(grayStrip, meanValue, stddevValue);
    double minimum = 0, maximum = 0;
    cv::minMaxLoc(grayStrip, &minimum, &maximum);
    const double floorLevel = std::clamp(
        std::max(meanValue[0] + stddevValue[0], maximum * 0.55), 100.0, 210.0);
    cv::Mat bright;
    cv::threshold(grayStrip, bright, floorLevel, 255, cv::THRESH_BINARY);
    cv::bitwise_and(mask, bright, mask);

    // AND WHATEVER IS NEARLY WHITE, whether or not it beat its neighbourhood.
    //
    // PIN: A GLYPH OVER A SUNLIT FRAME IS STILL WHITE. The pair of rules above compares
    // a pixel with what surrounds it, which is exactly the comparison that fails when
    // the surroundings are bright too — measured on a white play triangle over the
    // brightest frame of a video, where the button was never found at all and a plainly
    // visible transport was refused. Players draw their glyphs at full white; almost
    // nothing in a photographed or rendered picture is.
    cv::Mat pureWhite;
    cv::threshold(grayStrip, pureWhite, 232, 255, cv::THRESH_BINARY);
    cv::bitwise_or(mask, pureWhite, mask);
    // Close across the gap inside a pause glyph and between a speaker and its waves,
    // so one control is one blob — without reaching the next button along.
    const int close = std::max(3, grayStrip.rows / 6);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close, close)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double minSide = std::max(6.0, grayStrip.rows * 0.18);
    const double maxSide = grayStrip.rows * 0.95;
    for (const std::vector<cv::Point> &contour : contours) {
        const cv::Rect box = cv::boundingRect(contour);
        if (box.height < minSide || box.height > maxSide) continue;
        if (box.width < minSide * 0.5) continue;
        const double aspect = static_cast<double>(box.width) / box.height;
        // Wider than three-to-one is a timestamp or a title, not a control.
        if (aspect > 3.0 || aspect < 0.28) continue;
        // A SHAPE CUT OFF BY THE EDGE IS NOT A SHAPE. Measured on a live watch page: the
        // crop's left edge sliced through the player's chrome and left a 31-pixel
        // fragment at x=0, which matched `pause` at 0.45 — enough to become the leftmost
        // control and therefore the transport, while the real play button sat 25 pixels
        // along matching `play` at 0.72. Everything downstream then reasoned from the
        // fragment: the reading said "playing" about a paused video, and the verification
        // of a press that had worked reported that nothing had happened.
        if (box.x <= 1 || box.x + box.width >= grayStrip.cols - 1) continue;
        boxes.push_back(box);
    }
    std::sort(boxes.begin(), boxes.end(),
              [](const cv::Rect &a, const cv::Rect &b) { return a.x < b.x; });
    return boxes;
}

}  // namespace

double frameMotion(const cv::Mat &a, const cv::Mat &b) {
    const cv::Mat left = grayGrid(a, kMotionColumns, kMotionRows);
    const cv::Mat right = grayGrid(b, kMotionColumns, kMotionRows);
    if (left.empty() || right.empty() || left.size() != right.size()) return -1;
    cv::Mat difference;
    cv::absdiff(left, right, difference);
    const int differing = cv::countNonZero(difference > kMotionThreshold);
    return static_cast<double>(differing) / (kMotionColumns * kMotionRows);
}

MediaReading readMediaControls(const cv::Mat &frame, const cv::Mat &previous) {
    MediaReading reading;
    if (frame.empty() || frame.cols < 48 || frame.rows < 48) return reading;

    const cv::Mat color = toColor(frame);
    const cv::Mat gray = toGray(frame);
    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    // The row scan reads the smoothed copy; everything else reads the real pixels.
    cv::Mat smoothed;
    cv::blur(color, smoothed, cv::Size(kRowSmoothing, 1));

    // THE TRANSPORT IS THE TRACK THAT HAS A ROW OF EVENLY SIZED BUTTONS SPANNING IT.
    //
    // Several things on a page look like a thin two-tone bar: a search field's border,
    // a section divider, a card's edge. What none of them has is a row of small bright
    // glyphs of similar size, reaching from one end of the bar to the other — measured
    // against a real YouTube watch page, where a bare "has a couple of blobs under it"
    // test picked the masthead instead of the player.
    const std::vector<TrackCandidate> candidates = findTrackCandidates(smoothed);
    double bestScore = 0;
    TrackCandidate best;
    std::vector<MediaControl> bestControls;
    cv::Rect bestStrip;

    for (const TrackCandidate &candidate : candidates) {
        // A progress track is THIN. A thick band is a panel, and a one-pixel hairline is
        // a rule; both are excluded here rather than scored down, because neither ever
        // becomes a transport with more evidence.
        if (candidate.rect.height < 2 || candidate.rect.height > 14) {
            trace("candidate y=%d h=%d rejected: thickness", candidate.rect.y, candidate.rect.height);
            continue;
        }

        // AND IT HAS TWO COLOURS IN IT, OR ONE VIVID ONE. A progress bar shows how far
        // along it is; that is what makes it a progress bar. A single flat grey run is a
        // DIVIDER — measured on a YouTube page, where the rule above a video's title row
        // paired with the buttons beneath it and won the scan whenever the real
        // transport happened to be hidden, so a click landed on "Subscribe".
        //
        // THE COST, STATED: a player drawing a completely grey bar with no accent
        // anywhere and nothing played is refused here. No player found so far does that
        // — every one shows an accent run or a scrubber knob — and the alternative is
        // reporting somebody's Subscribe button as a play button, which is worse than
        // saying nothing.
        if (candidate.runs < 2 && candidate.saturation <= 60) {
            trace("candidate y=%d rejected: one flat run, saturation %.0f",
                  candidate.rect.y, candidate.saturation);
            continue;
        }

        const int rowHeight = std::min(
            kMaxControlRow,
            std::max(28, static_cast<int>(candidate.rect.width * kControlRowShare)));
        // THE BUTTON ROW IS WIDER THAN THE TRACK IT BELONGS TO. A player insets its bar
        // and then puts play and full screen just outside those insets — and the track's
        // own measured ends drift inward anyway once the row has been smoothed. Taking
        // the strip at exactly the track's width cut the play button off the left end,
        // and the volume control became "the leftmost control". Measured on YouTube.
        const int margin = std::max(12, static_cast<int>(candidate.rect.width * 0.06));
        const int stripX = std::max(0, candidate.rect.x - margin);
        const cv::Rect strip = cv::Rect(
            stripX,
            candidate.rect.y + candidate.rect.height + 1,
            std::min(frame.cols - stripX, candidate.rect.width + margin * 2),
            rowHeight) & bounds;
        if (strip.width < 32 || strip.height < 12) {
            trace("candidate y=%d rejected: no room for a control row", candidate.rect.y);
            continue;
        }

        std::vector<cv::Rect> boxes = controlBlobs(gray(strip));
        // A TRANSPORT IS AT LEAST THREE BUTTONS. Two is a coincidence; every player ever
        // shipped has play, something in the middle, and full screen.
        if (boxes.size() < 3) {
            trace("candidate y=%d rejected: %zu blobs under it", candidate.rect.y, boxes.size());
            continue;
        }

        // The glyphs in a control row are all about the same size. A masthead's mix of
        // an avatar, a search field and a logo is not.
        std::vector<int> heights;
        heights.reserve(boxes.size());
        for (const cv::Rect &box : boxes) heights.push_back(box.height);
        std::sort(heights.begin(), heights.end());
        const double median = heights[heights.size() / 2];
        if (median <= 0) continue;
        size_t alike = 0;
        for (int height : heights) {
            if (height >= median * 0.45 && height <= median * 2.2) ++alike;
        }
        if (alike * 3 < boxes.size() * 2) {
            trace("candidate y=%d rejected: blob sizes disagree (%zu of %zu alike)",
                  candidate.rect.y, alike, boxes.size());
            continue;
        }

        // AND THE ROW SPANS THE BAR. A player puts play at one end and full screen at
        // the other, so the buttons reach across the track they belong to; a cluster of
        // icons sitting under an unrelated line does not.
        int leftmost = strip.width, rightmost = 0;
        for (const cv::Rect &box : boxes) {
            leftmost = std::min(leftmost, box.x);
            rightmost = std::max(rightmost, box.x + box.width);
        }
        const double span = static_cast<double>(rightmost - leftmost) / strip.width;
        if (span < 0.55) {
            trace("candidate y=%d rejected: buttons span only %.2f of the bar",
                  candidate.rect.y, span);
            continue;
        }

        std::vector<MediaControl> controls;
        controls.reserve(boxes.size());
        for (const cv::Rect &box : boxes) {
            MediaControl control;
            control.frame = cv::Rect(box.x + strip.x, box.y + strip.y, box.width, box.height);
            controls.push_back(control);
        }

        // AND THE FIRST BUTTON IS A TRANSPORT BUTTON.
        //
        // This is the test that makes a false positive impossible rather than merely
        // unlikely. A page has plenty of thin lines with a row of evenly sized icons
        // beneath them — a site's masthead is exactly that — and every structural rule
        // above admits them. What no masthead has is a PLAY OR PAUSE TRIANGLE at the
        // left end. Measured: without this, a search bar's underline and the row of
        // header icons beneath it were reported as a video's transport, and the click
        // that followed went into somebody's navigation.
        // THE FIRST CONTROL, NOT ONE OF THE FIRST FEW. Play sits at the left end of every
        // player's bar; that is not a tendency, it is the convention every one of them
        // follows. Accepting a transport shape ANYWHERE near the left was measured
        // admitting a video's metadata row, whose thumbs-up matched a play triangle well
        // enough while the avatar beside it sat first — and a click meant for play landed
        // on somebody's like button.
        // IN THE LEADING FIFTH OF THE BAR, not literally the first blob. A letterbox
        // edge or a scrollbar sliver can land left of the play button and take its place
        // in the list — measured on Chrome, where a plainly visible transport was
        // refused because the first blob was not a triangle. A fifth is narrow enough
        // that a metadata row's like button, two thirds across, still cannot qualify.
        double transportConfidence = 0;
        const int leadingEdge = strip.x + std::max(24, strip.width / 5);
        for (const MediaControl &control : controls) {
            if (control.frame.x > leadingEdge) break;
            cv::Mat glyph;
            cv::threshold(gray(control.frame), glyph, 0, 255,
                          cv::THRESH_BINARY | cv::THRESH_OTSU);
            const vx_media_control named = matchGlyph(glyph);
            trace("  leading blob x=%d w=%d h=%d -> glyph %d at %.2f",
                  control.frame.x, control.frame.width, control.frame.height,
                  static_cast<int>(named.glyph), named.confidence);
            if (named.glyph == VX_GLYPH_PLAY || named.glyph == VX_GLYPH_PAUSE
                || named.glyph == VX_GLYPH_REPLAY) {
                transportConfidence = std::max(transportConfidence, named.confidence);
            }
        }
        if (transportConfidence < kTransportFloor) {
            trace("candidate y=%d rejected: first button is not a transport (%.2f)",
                  candidate.rect.y, transportConfidence);
            continue;
        }

        // COLOUR AND STRUCTURE DECIDE BETWEEN TWO BARS THAT BOTH HAVE BUTTONS UNDER
        // THEM. A player's control row has more than one horizontal edge above it — the
        // scrim's top, a pill's outline, the row's own boundary — and every one of them
        // passes the structural tests. Only one of them is a PROGRESS bar, and what makes
        // it one is that it is painted in an accent colour and divided into a played part
        // and an unplayed part. Measured on Chrome's newer YouTube layout, where a line
        // seventeen pixels below the real track won on structure alone and reported 88%
        // through on a video thirty seconds in.
        double score = 100.0;
        score += 10.0 * candidate.rect.width / frame.cols;
        score += 10.0 * span;
        score += std::min(6.0, static_cast<double>(controls.size()));
        score += std::min(6.0, 3.0 * (candidate.runs - 1));
        score += 15.0 * candidate.saturation / 255.0;
        // The clearest transport wins when two candidates both look like players.
        score += 20.0 * transportConfidence;
        trace("candidate y=%d x=%d w=%d runs=%d sat=%.0f score=%.1f",
              candidate.rect.y, candidate.rect.x, candidate.rect.width,
              candidate.runs, candidate.saturation, score);
        if (score > bestScore) {
            bestScore = score;
            best = candidate;
            bestControls = controls;
            bestStrip = strip;
        }
    }

    const bool hasTrack = bestScore >= 100 || (bestScore > 0 && best.saturation > 60);
    if (hasTrack) {
        reading.progress = best.rect;
        reading.progressFraction = best.fraction;
        reading.bar = cv::Rect(best.rect.x, best.rect.y, best.rect.width,
                               (bestStrip.y + bestStrip.height) - best.rect.y) & bounds;
        // Naming happens only for the winner: classifying every candidate's blobs would
        // pay for template matching over a page's worth of false tracks.
        for (MediaControl &control : bestControls) {
            cv::Mat candidate;
            cv::threshold(gray(control.frame), candidate, 0, 255,
                          cv::THRESH_BINARY | cv::THRESH_OTSU);
            const vx_media_control matched = matchGlyph(candidate);
            control.glyph = matched.glyph;
            control.confidence = matched.confidence;
        }
        reading.controls = bestControls;

        // AND THE VOLUME TRACK, when the player is showing one. Looked for only beside a
        // control the bank actually named, so a short run is bounded by evidence.
        for (const MediaControl &control : reading.controls) {
            if (control.glyph != VX_GLYPH_VOLUME && control.glyph != VX_GLYPH_MUTED) {
                continue;
            }
            const TrackCandidate volume = findVolumeTrack(color, control.frame);
            if (volume.rect.width > 0 && volume.runs >= 1) {
                reading.volumeTrack = volume.rect;
                reading.volumeFraction = volume.fraction;
            }
            break;
        }
    }
    reading.controlsVisible = hasTrack && reading.controls.size() >= 2;

    // THE PICTURE IS THE BOX ABOVE THE BAR, not everything the caller handed over. A
    // watch page is mostly comments and thumbnails; measuring motion over all of it
    // would report an animated advert as a playing video. 16:9 bounds the box — it is a
    // bound, not a claim about the video's real shape.
    cv::Rect picture = bounds;
    if (hasTrack) {
        const int height = std::min(
            static_cast<int>(best.rect.width * kPlayerAspect), reading.bar.y);
        picture = cv::Rect(best.rect.x, std::max(0, reading.bar.y - height),
                           best.rect.width, std::max(1, height)) & bounds;
    }
    if (!previous.empty() && previous.size() == frame.size() && picture.area() > 64) {
        reading.motion = frameMotion(frame(picture), previous(picture));
    }

    // A paused player often draws one big glyph over the middle of its picture.
    // Corroboration only — plenty of players draw nothing at all.
    const cv::Rect center(
        picture.x + picture.width * 3 / 10, picture.y + picture.height * 3 / 10,
        picture.width * 4 / 10, picture.height * 4 / 10);
    const cv::Rect centerClipped = center & bounds;
    if (centerClipped.width > 16 && centerClipped.height > 16) {
        const std::vector<cv::Rect> blobs = controlBlobs(gray(centerClipped));
        const auto largest = std::max_element(
            blobs.begin(), blobs.end(), [](const cv::Rect &a, const cv::Rect &b) {
                return a.area() < b.area();
            });
        if (largest != blobs.end() && largest->area() > centerClipped.area() / 64) {
            const cv::Rect placed(largest->x + centerClipped.x, largest->y + centerClipped.y,
                                  largest->width, largest->height);
            cv::Mat candidate;
            cv::threshold(gray(placed), candidate, 0, 255,
                          cv::THRESH_BINARY | cv::THRESH_OTSU);
            const vx_media_control matched = matchGlyph(candidate);
            reading.centerGlyph.frame = placed;
            reading.centerGlyph.glyph = matched.glyph;
            reading.centerGlyph.confidence = matched.confidence;
        }
    }

    return reading;
}

}  // namespace visionax
