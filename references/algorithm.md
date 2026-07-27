# Algorithm Reference

This document explains *why* the skill works, for debugging or curiosity. You don't need to read it to use the skill.

## The problem

AI image generators (ChatGPT, Gemini, DALL·E, Midjourney) are sometimes asked for "transparent background". They often misunderstand: instead of producing a PNG with an alpha channel (true transparency), they **draw the checkerboard pattern** — the universal UI symbol for transparency — as actual colored pixels in the RGB channels.

The result: an opaque image where transparent areas are filled with a gray-white alternating grid. When used anywhere, the checkerboard shows up as a real background.

## Why standard tools fail

### Background removal tools (rembg, remove.bg)

These expect a photographic background (sky, wall, solid color). A checkerboard is "busy" — full of edges and color variation. Salient object detectors get confused: is the grid the background, or is it texture on the subject? Results are inconsistent.

### Color thresholding

Stack Overflow's most common answer: threshold pixels in the light-gray range (e.g. `cv2.inRange(img, (230,230,230), (255,255,255))`) and make them transparent.

**Fails on**: any light-colored subject. A white card, light text, pale icons — all fall in the threshold range and get removed along with the checkerboard.

### Frequency-domain filtering (DFT notch filter)

Another SO answer: treat the checkerboard as a periodic signal, find its frequency peaks in the DFT, and notch them out.

**Fails because**: the original SO author admits "the checkerboard is mitigated, but still there." Frequency filtering reduces but doesn't eliminate the pattern. It also assumes a fixed grid size and doesn't produce a clean alpha matte.

### Contour + area filtering

Find connected regions matching the checker colors, delete small ones (assume noise).

**Fails on**: large light-colored areas inside the subject. The SO author notes "large white areas in the foreground may not work."

## Our approach: two-engine fusion

### Engine 1 — Checkerboard residual method

Exploits the *defining mathematical property* of a checkerboard: it's a **two-color periodic pattern**. Subject pixels break this periodicity.

**Steps:**

1. **Detect period** (`detect_period`): scan rows/columns, measure "same-color run lengths", find the dominant period T via histogram. Works on any grid size — no hardcoding.
2. **Detect two colors** (`detect_two_colors`): sample the image border (confident background region), build a brightness histogram, find the two peaks via local maxima. This is robust because it doesn't depend on phase alignment.
3. **Build ideal template** (`build_template`): reconstruct what a "perfect checkerboard" would look like at every pixel, given T, colors, and phase.
4. **Compute residual** (`compute_alpha`): for each pixel, compute distance to the *nearest* checkerboard color. Small distance → likely background (alpha → 0). Large distance → likely subject (alpha → 1). The transition is linear with a dead-zone to avoid noise sensitivity.

**What it's good at**: confidently identifying background pixels. The periodic pattern is a strong, unambiguous signal.

**What it fails at**: subject pixels whose color happens to be close to a checkerboard color (e.g. white card ≈ white square). These get misclassified as background.

### Engine 2 — Semantic segmentation (BiRefNet / rembg)

A neural network that recognizes "what is a subject" regardless of color. It understands that a white card is an object, even though its color matches the checkerboard.

**What it's good at**: subject boundary detection, especially for light-colored subjects.

**What it fails at**: thin strokes (sometimes), non-photographic subjects (pixel art, geometric shapes), and it can't remove the checkerboard texture itself — it just finds the subject boundary.

### Fusion logic

```
alpha_engine = semantic_segmentation(image)    # subject map
alpha_cb     = residual_method(image)          # background map

# Engine is authoritative for subject detection
# Residual method is authoritative for background detection
# Only override the engine where BOTH agree it's background:
sure_bg = (alpha_cb < 0.1) AND (alpha_engine < 0.3)
alpha   = max(alpha_engine, alpha_cb)          # union of both
alpha[sure_bg] = 0                              # force transparent
```

**Why this works**: the residual method never *removes* something the engine considers a subject. It only reinforces transparency where the engine already agrees. This prevents the "white card eaten" problem while still getting cleaner background removal than the engine alone.

**Why not just use the engine alone**: the engine finds the subject, but it doesn't know the checkerboard exists. On its own output, you'd see faint checkerboard residue at edges, and the subject boundary might include or exclude partial checker squares inconsistently. The residual method gives precise background detection; the engine gives accurate subject detection.

## Post-processing

### Morphological cleanup (`clean_alpha`)

After fusion, a 3×3 morphological open + close removes salt-and-pepper noise, followed by a light Gaussian blur.

### Feathering

Optional larger-radius Gaussian blur on the alpha channel, applied only to the transition zone. Because Gaussian blur of a binary-ish signal only affects values near 0.5, the core subject (alpha ≈ 1) and core background (alpha ≈ 0) are unaffected — only edges soften.

### RGB erasure

Pixels with alpha < 10 have their RGB set to (0,0,0). This prevents "ghost" checkerboard pixels from lingering in fully-transparent areas (which would be invisible but are dirty data). It also helps file compression.

## Limitations

1. **Non-checkerboard backgrounds**: if the AI drew a solid color or gradient instead of a checkerboard, period detection returns confidence ≈ 0. The script then enters **solid-background mode** (see below), not a failure.

2. **Very small subjects**: if the subject covers <5% of the image and the checkerboard dominates, two-color detection is robust. If the subject covers >70%, border sampling may fail (no confident background region). In this case, colors default to 254/243 (typical AI checkerboard) — usually still correct.

3. **Multiple checker scales in one image**: not supported. Each image is assumed to have one consistent grid size.

4. **JPEG artifacts**: if the source is JPEG (not PNG), checkerboard edges are blurry and colors spread. Detection still works but edges are less clean. Always prefer PNG sources.

## Solid-background fallback mode

When `_verify_checker_pattern` rejects the detected period (the "period" was actually a subject edge, not a real grid), the script switches to solid-background removal:

1. **Detect solid bg** (`detect_solid_background`): sample four edge strips. If color std < 25 across all strips, classify as solid background (white/light/color).
2. **Subject isolation**: use the semantic engine for boundary, plus `floodfill_background` from four image corners with the detected bg color and a tolerance (default 12). Take the union — the engine catches subjects the flood fill might merge with bg (light-colored subjects), and the flood fill catches connected bg regions the engine might miss.
3. **No-engine fallback**: pure flood fill. Works for well-separated subjects. Fails on subjects with colors very close to the background (gap < tolerance).

The checkerboard verification step is what makes the two modes switch cleanly. It counts same-color run lengths in edge strips: real checkerboards have many short runs clustered at period T; solid backgrounds have few long runs.
