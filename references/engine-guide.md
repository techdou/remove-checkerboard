# Engine Selection Guide

## How the fusion works

All engines are fused with the **checkerboard residual method** — they don't work alone. The residual method handles background detection (it's excellent at finding the periodic pattern), the engine handles subject boundary detection. Read `algorithm.md` for the fusion logic.

This means engine choice mainly affects **edge quality on the subject boundary**, not background removal.

## Engine comparison

Tested on 8 AI-generated images (art text, info cards, banners, vertical strips) with checkerboard backgrounds baked into pixels.

### rembg (default)

- **Model**: U²-Net via rembg
- **Strengths**: CPU-friendly, no GPU needed. Good subject boundary on photos and illustrations. Handles light-colored subjects near the bg color better than color keying, because it segments the subject semantically.
- **Weaknesses**: Edges slightly less clean on flat geometric shapes compared to BiRefNet.
- **Download**: ~176MB, first-run only, cached to `~/.u2net/`.
- **Speed**: ~0.3s/image on CPU.

### BiRefNet (best quality, GPU)

- **Model**: ZhengPeng7/BiRefNet_dynamic (2025.3, SOTA dichotomous image segmentation)
- **Architecture**: Bilateral Reference framework, trained on dynamic resolutions 256×256 to 2304×2304
- **Strengths**: Cleanest edges on geometric shapes (cards, frames, icons). Handles light-colored subjects (white card backgrounds) well — doesn't confuse them with white checkerboard squares.
- **Weaknesses**: Slightly less sensitive to thin strokes (art text, fine lines) than rembg in some cases. Requires GPU.
- **Download**: ~900MB, first-run only, cached to HuggingFace cache.
- **Speed**: ~0.5s/image on RTX 5070 Ti.

### rembg (U²-Net)

- **Model**: U²-Net (2020, salient object detection)
- **Strengths**: Good sensitivity to any "non-background" region — works well on thin strokes, text, hairlines. Runs on CPU. Small model (~176MB).
- **Weaknesses**: Older model, edges slightly less clean on flat geometric shapes compared to BiRefNet.
- **Speed**: ~0.3s/image (CPU or GPU).

### none (residual-only)

- **How**: Pure checkerboard detection, no neural network.
- **Strengths**: Instant, no download, fully offline.
- **Weaknesses**: Cannot distinguish subject's light-colored areas from white checkerboard squares. A white card on a white/gray checkerboard will have its white interior treated as background. Only use for quick previews or subjects with dark/distinct colors.

## When the fusion matters

The key insight: **pure color/texture methods fail on light-colored subjects**. A white card background (RGB ~251) and a white checkerboard square (RGB ~254) are too close in color space to separate by thresholding alone.

This is why every public solution (Stack Overflow answers, existing scripts) has the same acknowledged limitation: "large white areas in the foreground may not work."

The neural network engine solves this by *semantically* recognizing "this is a card/object" vs "this is a background pattern". The residual method then handles the actual checkerboard removal. Together they cover each other's blind spots.

### corner (fast path for uniform AI asset packs)

- **How**: samples four corners + top-center, keys out pixels close to those colors (Manhattan distance, tol 26/channel). No model, no download.
- **Strengths**: ~0.1s/image; excellent on asset packs where the subject colors differ from the fake-transparent background (character sprites, envelopes, cards, frames). Skips images that already have real alpha.
- **Weaknesses**: no subject understanding — light subjects close to the bg color get partially eaten; semi-transparent subjects (glass) and full-bleed scenes trip the safety guard (more than 2 sampled colors) and auto-fall back to the checkerboard/solid-bg pipeline.
- **When**: batch-processing AI asset packs; try it first, check `mode` in output — `corner_key` means clean keying, anything else means fallback happened and you may want `rembg` for that image.

## GPU vs CPU

BiRefNet requires CUDA. If you get a CUDA error or are on CPU-only:

```bash
# Fall back to rembg (runs on CPU)
python scripts/remove_checkerboard.py input.png --engine rembg --feather 4
```

rembg uses ONNX Runtime and runs acceptably on CPU (~0.3-0.5s/image).

## Two background modes

The skill handles two distinct background types, auto-detected per image:

### Checkerboard mode (`mode=checkerboard`)

For AI-generated "fake transparency" — the gray-white grid is baked into pixels. Uses the residual method + engine fusion described above. This is the primary mode.

### Solid-background mode (`mode=solid_bg`)

Activated when no checkerboard pattern is detected (the `_verify_checker_pattern` check fails). Handles white/flat-color backgrounds:

1. `detect_solid_background`: samples image edge strips. If pixel colors are tightly clustered (std < 25), it's a solid background. Classifies as white/light/color.
2. Background removal: engine (rembg/BiRefNet) for subject boundary + `floodfill_background` from four corners to catch any connected background region the engine missed. Union of both.
3. `none` engine fallback: pure flood fill with color tolerance — no neural network needed, works offline.

**Known limitation of solid-bg mode**: flood fill with `none` engine can't recover subjects whose color is very close to the background (e.g. a pale cream card on white, gap < 12 in RGB). The neural engine handles this fine; flood fill alone may merge them into background. For such cases, use `--engine rembg` or `--engine birefnet`.

## Environment requirements

- Python 3.10+
- `pillow`, `numpy`, `scipy` (core)
- `rembg`, `onnxruntime` (for rembg engine)
- `torch`, `torchvision`, `transformers`, `einops`, `kornia` (for birefnet engine)
- Optional: `opencv-python` (faster morphology + flood fill; falls back to PIL if absent)
