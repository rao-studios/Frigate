//
//  visionax.h
//  CVisionAX
//
//  WHAT: The engine's only public surface. Pure C — Swift imports this, nothing else.
//  IN:   VisionAX/Engine/VisionEngine.swift
//  OUT:  visionax.cpp (the extern "C" facade over visionax::Engine)
//  PIN:  Every rect is in INPUT-IMAGE PIXEL SPACE, TOP-LEFT ORIGIN — the same
//        orientation as AX. Output arrays are malloc'd by the engine and returned to
//        it through the matching vx_*_free; the caller never frees them itself.
//

#ifndef VISIONAX_H
#define VISIONAX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: - Status

typedef enum vx_status {
    VX_OK = 0,
    VX_ERROR_INVALID_ARGUMENT = 1,
    VX_ERROR_UNSUPPORTED_FORMAT = 2,
    VX_ERROR_INTERNAL = 3,
    /// A model would not load, or a session failed mid-run. The text is in
    /// vx_classifier_last_error — ONNX Runtime's own message, which names the tensor.
    VX_ERROR_MODEL = 4,
} vx_status;

/// Static string for a status; never NULL.
const char *vx_status_message(vx_status status);

// MARK: - Input

typedef enum vx_pixel_format {
    /// One byte per pixel.
    VX_PIXEL_GRAY8 = 0,
    /// Four bytes per pixel, B G R A in memory — what a little-endian premultipliedFirst
    /// CGBitmapContext produces.
    VX_PIXEL_BGRA8 = 1,
    /// Four bytes per pixel, R G B A in memory.
    VX_PIXEL_RGBA8 = 2,
} vx_pixel_format;

/// A borrowed view of caller-owned pixels. Row 0 is the TOP row.
typedef struct vx_image_view {
    const uint8_t *data;
    int32_t width;
    int32_t height;
    int32_t bytes_per_row;
    vx_pixel_format format;
} vx_image_view;

// MARK: - Options

/// Canny → contours → boxes → containment tree. Defaults from vx_canny_options_default().
typedef struct vx_canny_options {
    /// Canny hysteresis thresholds. Default 20 / 60 — low, because UI borders are
    /// low-contrast by design and the textbook 50/150 misses them entirely.
    double low_threshold;
    double high_threshold;
    /// Sobel aperture for Canny: 3, 5, or 7. Default 5 — a soft 1px UI border spreads
    /// its gradient wider than a 3x3 kernel can see.
    int32_t aperture_size;
    /// Gaussian blur kernel before Canny; 0 = none, otherwise odd. Default 3.
    int32_t blur_kernel;
    /// Morphological close kernel after Canny, sealing gaps so a box's edge reads as
    /// one loop and a word's strokes read as one blob; 0 = none. Default 5.
    int32_t close_kernel;
    /// Boxes narrower or shorter than this are dropped. Default 8 / 8.
    int32_t min_width;
    int32_t min_height;
    /// A box whose IoU with a larger kept box reaches this is a duplicate. Default 0.90.
    double merge_iou;
    /// A box whose four edges all lie within this many pixels of a larger kept box is a
    /// duplicate — the inner and outer contour of one stroked outline. Default 4.
    int32_t merge_slack;
    /// Tolerance, in pixels, when deciding "A contains B". Default 2.
    int32_t containment_slack;
    /// Siblings are ordered by (floor(midY / reading_band), minX) — Mary's roster
    /// reading rule. Default 24.
    int32_t reading_band;
    /// Walk budget, Mary AXTreeWalker semantics: a node AT max_depth is kept and its
    /// children withheld; the walk stops after max_nodes. Default 24 / 4000.
    int32_t max_depth;
    int32_t max_nodes;
} vx_canny_options;

vx_canny_options vx_canny_options_default(void);

// MARK: - Output

/// One node of the flat, PRE-ORDER tree. Index 0 is the root — the whole image — with
/// parent == -1 and depth == 0. Children follow their parent, in reading order.
typedef struct vx_region {
    /// Monotonic from 1 in pre-order.
    uint32_t id;
    /// Index into the same array, or -1 for the root.
    int32_t parent;
    int32_t depth;
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    /// Children actually emitted (a budget can withhold some).
    int32_t child_count;
} vx_region;

typedef struct vx_region_tree {
    vx_region *regions;
    int32_t count;
    /// 1 when max_depth or max_nodes cut the walk short.
    int32_t truncated;
    /// Contours found before size filtering and dedup — a tuning signal.
    int32_t contour_count;
} vx_region_tree;

/// The raw Canny output — 8-bit, one channel, 255 on an edge.
typedef struct vx_edge_map {
    uint8_t *data;
    int32_t width;
    int32_t height;
    int32_t bytes_per_row;
} vx_edge_map;

// MARK: - Engine

/// Opaque. Stateless per call today, so one instance may serve many threads.
typedef struct vx_engine vx_engine;

/// NULL only on allocation failure.
vx_engine *vx_engine_create(void);
void vx_engine_destroy(vx_engine *engine);

/// The linked OpenCV's version string, e.g. "4.13.0". Static; never NULL.
const char *vx_engine_opencv_version(void);

/// This engine's own version, e.g. "0.1.0" — stamped into every dataset sample so a
/// training run can tell which detector produced its proposals. Static; never NULL.
const char *vx_engine_version(void);

/// The linked ONNX Runtime's version, e.g. "1.24.2", or NULL if the runtime could not
/// be reached at all. Creating the shared environment is part of answering, so a
/// non-NULL result proves the static archive linked, not merely that a header parsed.
const char *vx_onnxruntime_version(void);

/// Canny edges → bounding boxes → containment tree.
/// `out_tree` is required; `out_edges` may be NULL when the caller does not want the
/// edge map. On any status other than VX_OK both outputs are left zeroed.
vx_status vx_engine_detect_regions(
    vx_engine *engine,
    const vx_image_view *image,
    const vx_canny_options *options,
    vx_region_tree *out_tree,
    vx_edge_map *out_edges);

/// Release what vx_engine_detect_regions handed out. Safe on a zeroed struct.
void vx_region_tree_free(vx_region_tree *tree);
void vx_edge_map_free(vx_edge_map *edges);

// MARK: - Classifier

/// Everything the runtime needs to reproduce training-time pixels. Swift fills this
/// from the model's JSON sidecar; C never parses JSON and never sees a class NAME —
/// only how many there are, so an index can be range-checked.
typedef struct vx_classifier_spec {
    /// Resize so max(width, height) == long_side. NEVER upscales; 0 disables.
    int32_t long_side;
    /// Pad right/bottom out to a multiple of this. Default 32.
    int32_t pad_multiple;
    /// Per-channel RGB normalization, values in 0..1.
    float mean[3];
    float std[3];
    /// The head's output width. Validated against the loaded graph.
    int32_t class_count;
    /// How many boxes go through the head in one Run. Default 512.
    int32_t max_boxes_per_run;
    /// 0 lets ONNX Runtime choose.
    int32_t intra_op_threads;
    /// 0 = CPU. Reserved: the CoreML provider cannot run RoiAlign, so a CoreML build
    /// has to split the graph deliberately rather than flip a flag here.
    int32_t execution_provider;
    /// The backbone's output stride. Features handed to vx_classifier_classify_features
    /// must measure the padded image divided by this, on both axes. Default 8.
    int32_t feature_stride;
} vx_classifier_spec;

vx_classifier_spec vx_classifier_spec_default(void);

/// Opaque. Holds two ONNX Runtime sessions; safe to use from many threads at once.
typedef struct vx_classifier vx_classifier;

/// Loads the two graphs. On failure returns VX_ERROR_MODEL and leaves *out NULL.
vx_status vx_classifier_create(
    const char *backbone_path,
    const char *head_path,
    const vx_classifier_spec *spec,
    vx_classifier **out);

void vx_classifier_destroy(vx_classifier *classifier);

/// Why the last call failed, in ONNX Runtime's own words. "" when nothing has failed.
/// Valid until the next call on the same classifier; never NULL.
const char *vx_classifier_last_error(const vx_classifier *classifier);

/// Run counters — proof, for a test, that one image costs one backbone pass however
/// many boxes it carries.
typedef struct vx_classifier_stats {
    int64_t backbone_runs;
    int64_t head_runs;
} vx_classifier_stats;

vx_classifier_stats vx_classifier_get_stats(const vx_classifier *classifier);

/// One region's answer: an index into the model's role table, and how sure it is.
typedef struct vx_region_label {
    int32_t class_index;
    float confidence;
} vx_region_label;

typedef struct vx_region_labels {
    vx_region_label *labels;
    int32_t count;
} vx_region_labels;

/// Labels `regions[0..count)`, in order — `out_labels->labels[i]` answers `regions[i]`.
/// The caller passes the boxes it wants named; the root is normally excluded, since the
/// whole image is not an element. A count of 0 succeeds and returns an empty result.
vx_status vx_engine_classify_regions(
    vx_engine *engine,
    vx_classifier *classifier,
    const vx_image_view *image,
    const vx_region *regions,
    int32_t count,
    vx_region_labels *out_labels);

void vx_region_labels_free(vx_region_labels *labels);

// MARK: - Classifier, split at the backbone

/// Opaque: one image, preprocessed exactly the way vx_engine_classify_regions does it.
/// PIN: THE SEAM A GPU BACKBONE PLUGS INTO. Preprocessing and the RoiAlign head stay here,
///      on the path Python's preprocessing is proven against; only the backbone moves.
typedef struct vx_prepared_image vx_prepared_image;

/// Resize, pad and normalize `image` for this classifier. On failure returns VX_ERROR_MODEL
/// (the reason in vx_classifier_last_error) and leaves *out NULL.
vx_status vx_classifier_prepare_image(
    vx_classifier *classifier,
    const vx_image_view *image,
    vx_prepared_image **out);

/// The prepared tensor: 3 x height x width floats, CHW, normalized, padded with the mean
/// colour (which normalizes to 0). Valid until vx_prepared_image_free.
const float *vx_prepared_image_tensor(
    const vx_prepared_image *prepared,
    int32_t *out_height,
    int32_t *out_width);

void vx_prepared_image_free(vx_prepared_image *prepared);

/// A backbone's output: channels x height x width floats, CHW, malloc'd.
typedef struct vx_feature_map {
    float *data;
    int32_t channels;
    int32_t height;
    int32_t width;
} vx_feature_map;

/// The classifier's own ONNX backbone over a prepared tensor (3 x height x width, CHW — what
/// vx_prepared_image_tensor returns): the same features the single call hands its head, and
/// the reference any other backbone is measured against. Free with vx_feature_map_free.
vx_status vx_classifier_backbone_features(
    vx_classifier *classifier,
    const float *image,
    int32_t height,
    int32_t width,
    vx_feature_map *out_features);

void vx_feature_map_free(vx_feature_map *features);

/// The head alone, over features another backbone computed from `prepared`. `features` is
/// channels x height x width, CHW and contiguous; a shape the head was not exported for is
/// VX_ERROR_MODEL, with both shapes named in vx_classifier_last_error. Labels exactly as
/// vx_engine_classify_regions returns them; free with vx_region_labels_free.
vx_status vx_classifier_classify_features(
    vx_classifier *classifier,
    const vx_prepared_image *prepared,
    const float *features,
    int32_t channels,
    int32_t height,
    int32_t width,
    const vx_region *regions,
    int32_t count,
    vx_region_labels *out_labels);

// MARK: - Media controls

/// What a player's control glyph depicts. A CLOSED vocabulary: the detector may answer
/// VX_GLYPH_NONE, and a caller that cannot name a control still knows where it is.
typedef enum vx_media_glyph {
    VX_GLYPH_NONE = 0,
    VX_GLYPH_PLAY = 1,
    VX_GLYPH_PAUSE = 2,
    VX_GLYPH_REPLAY = 3,
    VX_GLYPH_VOLUME = 4,
    VX_GLYPH_MUTED = 5,
    VX_GLYPH_FULLSCREEN = 6,
    VX_GLYPH_EXIT_FULLSCREEN = 7,
    VX_GLYPH_SETTINGS = 8,
    VX_GLYPH_CAPTIONS = 9,
    VX_GLYPH_NEXT = 10,
    VX_GLYPH_PREVIOUS = 11,
    VX_GLYPH_THEATER = 12,
    VX_GLYPH_MINIPLAYER = 13,
} vx_media_glyph;

// MARK: - Interface icons

/// What a wordless control depicts. A CLOSED vocabulary, and a SMALL one: these are the
/// shapes that mean the same thing everywhere, so a name from this list is a name a
/// person would use. Anything else answers VX_ICON_NONE and keeps its position.
typedef enum vx_icon_glyph {
    VX_ICON_NONE = 0,
    VX_ICON_SEARCH = 1,
    VX_ICON_CLOSE = 2,
    VX_ICON_MENU = 3,
    VX_ICON_BACK = 4,
    VX_ICON_FORWARD = 5,
    VX_ICON_UP = 6,
    VX_ICON_DOWN = 7,
    VX_ICON_ADD = 8,
    VX_ICON_MORE = 9,
    VX_ICON_SHARE = 10,
    VX_ICON_MICROPHONE = 11,
    VX_ICON_NOTIFICATIONS = 12,
    VX_ICON_ACCOUNT = 13,
    VX_ICON_STAR = 14,
    VX_ICON_HEART = 15,
    VX_ICON_DELETE = 16,
    VX_ICON_DONE = 17,
    VX_ICON_SETTINGS = 18,
    VX_ICON_HOME = 19,
    VX_ICON_FILTER = 20,
    VX_ICON_CART = 21,
    VX_ICON_DOWNLOAD = 22,
} vx_icon_glyph;

/// An integer rect in the same input-image pixel space as vx_region.
typedef struct vx_rect {
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
} vx_rect;

/// One control found in the transport, with what it looks like and how sure that is.
typedef struct vx_media_control {
    vx_rect frame;
    vx_media_glyph glyph;
    /// Normalized template correlation, 0..1. Below the caller's threshold the frame is
    /// still good — position is geometry, identity is a guess.
    double confidence;
} vx_media_control;

/// One look at a media player's on-screen transport.
///
/// EVERY FIELD IS AN OBSERVATION, NOT A CONCLUSION. Whether the video is PLAYING is not
/// decided here: this reports motion, the progress fraction, and what the glyphs look
/// like, and the caller weighs those witnesses. A detector that answered "playing" would
/// be hiding which evidence it had.
typedef struct vx_media_reading {
    /// 1 when a transport was found at all — a progress bar, or enough control blobs.
    int32_t controls_visible;
    /// The control bar's band, when one was found; zero-sized otherwise.
    vx_rect bar;
    /// The progress track. Zero-sized when no bar row was found.
    vx_rect progress;
    /// How far along the track the filled part reaches, 0..1, or -1 when unknown.
    double progress_fraction;
    /// How much the picture changed against `previous`, 0..1, measured OVER THE VIDEO
    /// AREA ONLY (the bar band is excluded, because a moving progress bar is not a
    /// moving picture). -1 when no previous frame was given.
    double motion;
    /// Controls found in the bar, left to right. Freed by vx_media_reading_free.
    vx_media_control *controls;
    int32_t control_count;
    /// The big centered glyph a paused player draws over the picture, when there is one.
    /// glyph == VX_GLYPH_NONE and a zero frame means none was found.
    vx_media_control center_glyph;
    /// The short track beside the volume control, when the player is showing one. It
    /// exists only while the pointer is over that control, so a zero rect usually means
    /// "not revealed" rather than "not there".
    vx_rect volume_track;
    /// How far along that track, 0..1, or -1 when there is none.
    double volume_fraction;
} vx_media_reading;

/// Read a player's transport out of one frame, optionally against the frame before it.
///
/// `frame` is the PLAYER's pixels — the caller crops to the page or video region first,
/// which is both the region of interest and the reason this needs no ROI argument.
/// `previous` may be NULL; without it `motion` is -1 and the caller has one fewer witness.
vx_status vx_engine_read_media_controls(
    vx_engine *engine,
    const vx_image_view *frame,
    const vx_image_view *previous,
    vx_media_reading *out_reading);

void vx_media_reading_free(vx_media_reading *reading);

/// What one wordless control looks like.
typedef struct vx_icon_match {
    vx_icon_glyph glyph;
    /// Silhouette overlap, 0..1. The caller's threshold decides whether it is a name.
    double confidence;
} vx_icon_match;

/// Name the wordless controls in a frame, one answer per rect.
///
/// `rects` are boxes the caller already believes are controls — small, near-square, and
/// with no words inside them. `out_matches` must have room for `count` answers. A rect
/// the bank cannot name comes back VX_ICON_NONE, which is not a failure: position is
/// geometry and identity is a guess.
vx_status vx_engine_match_icons(
    vx_engine *engine,
    const vx_image_view *frame,
    const vx_rect *rects,
    int32_t count,
    vx_icon_match *out_matches);

#ifdef __cplusplus
}
#endif

#endif /* VISIONAX_H */
