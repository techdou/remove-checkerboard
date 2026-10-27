---
name: remove-checkerboard
description: Remove backgrounds from images to produce real transparent PNGs, in three ways - (1) fake checkerboard patterns baked into AI-generated image pixels (ChatGPT/Gemini/DALL-E "fake transparency"), (2) solid or white backgrounds, (3) fast batch processing of AI-generated asset packs (sprites, icons, envelopes) via corner color keying. Use whenever the user mentions checkerboard in image pixels, fake transparency, transparent background not working, white background removal, cutout or extract subject, batch asset-pack background removal, 棋盘格背景, 假透明, 白底变透明, 去白底, 抠图, 去背景, 素材包批量抠底, or shows images where a grid or solid color blocks the subject. Auto-detects background type and engine capability - user does not need to specify. For asset packs prefer --engine corner (zero-model, instant); for subjects with light areas close to the background color use the default rembg engine.
---

# Remove Checkerboard & White Background

Produce real transparent PNGs from three common scenarios:

1. **Fake checkerboard**: AI generators (ChatGPT, Gemini, DALL·E) sometimes render the *visual symbol* of transparency (gray-white grid) as actual pixels instead of a real alpha channel.
2. **Solid/white background**: images with white or light-colored backgrounds that need to become transparent.
3. **Batch asset packs**: many AI-generated sprites/icons/envelopes at once — use the `corner` engine for instant, zero-model processing.

The script **auto-detects** which type it is — you don't choose. It checks for a checkerboard pattern first; if none found, it falls back to solid-background mode.

**Quick check**: open the image in an editor. If "transparent" areas show a checkerboard and there's no alpha channel → fake checkerboard. If the background is a flat white/solid color → solid background. Either way, use this skill.

## Quick start

```bash
# Default engine (rembg) - CPU works out of the box, first run downloads ~176MB model
python scripts/remove_checkerboard.py input.png -o ./output --feather 4

# Batch a folder (mixed checkerboard + white-bg images is fine)
python scripts/remove_checkerboard.py ./images -o ./output --feather 4

# AI asset pack, batch (zero model download, instant; auto-falls back if unsuitable)
python scripts/remove_checkerboard.py ./sprites -o ./out --engine corner --feather 4

# Best edges, needs NVIDIA GPU
python scripts/remove_checkerboard.py input.png -o ./output --engine birefnet --feather 4
```

## How background detection works

The script auto-detects the background type per image — no flag needed:

1. **Checkerboard mode**: detects a two-color periodic grid, removes it via residual method + semantic segmentation fusion. For AI-generated "fake transparency".
2. **Solid-background mode**: if no checkerboard found, detects the solid background color (white, light gray, etc.) from image edges, then removes it via semantic segmentation + flood fill. For white-bg photos, product shots, scanned documents.

The mode is printed in the output. You can mix image types in the same batch — each is handled automatically. The full set of `mode` values and what they mean:

| mode | Meaning |
|---|---|
| `checkerboard` | Fake-transparency grid detected, residual+engine fusion applied |
| `solid_bg` | Flat background detected, engine+flood-fill applied |
| `corner_key` | corner engine succeeded (uniform bg, subject kept) |
| `corner_fallback` | corner engine skipped (corners on content), routed to standard pipeline |

For batch runs, a line is printed per image; a line with `[!]` prefix means the image hit an error path — check that image individually.

## Choose your engine

| Engine | Quality | Speed | Needs GPU | When to use |
|---|---|---|---|---|
| `birefnet` | Best | ~0.5s/img | Yes (CUDA) | You have an NVIDIA GPU and want the best edges |
| `rembg` (default) | Good | ~0.3s/img | No | No GPU, or CPU-only machine. Best for subjects with light/white areas close to the bg color |
| `corner` | Subject-dependent | ~0.1s/img | No | **AI-generated asset packs with uniform fake-transparent backgrounds** (character sprites, icons, envelopes) where the subject's colors differ from the bg. Zero-model, zero-download. Auto-falls back to the standard pipeline when corner points land on content (samples > 2) |
| `none` | Basic | ~0.1s/img | No | Quick preview, offline. Checkerboard uses residual-only; solid-bg uses flood fill |

## corner engine: scope & limits

`--engine corner` samples the four corners (+ top-center) of the image and
keys out every pixel close to those colors (Manhattan distance, tol 26/channel).
Real-world validated on a 52-image watercolor asset pack (dual-universe):

- **Great** for character sprites, solid-color props (envelopes, cards, frames) —
  clean cutout at ~0.1s/image, no model download.
- **Auto-fallback** when the corners land on content (semi-transparent glass,
  full-bleed scenes → more than 2 sampled colors): the image is routed through
  the standard checkerboard/solid-bg pipeline instead. The output `mode` field
  says `corner_key` vs `corner_fallback`/`checkerboard`/`solid_bg`.
- **Known limits**: light subjects whose color is close to the bg color
  (white flowers on white, pale pink on pale pink) get partially eaten — use
  `rembg` for those. Corner engine skips images that already have a real
  alpha channel.

Read `references/engine-guide.md` for the full comparison and edge cases.

## Feather (anti-aliasing)

`--feather N` controls edge softness. Works the same in both modes.

- `0` — sharp but jagged on diagonals/curves
- `4` — **recommended**, smooth edges without losing detail
- `8` — very soft, may make thin strokes (art text) look slightly fuzzy

For per-image-type tuning, read `references/tuning.md`.

## Install dependencies

```bash
# Core (enough for --engine corner and --engine none)
pip install pillow numpy scipy

# Default rembg engine
pip install rembg onnxruntime

# birefnet engine (GPU)
pip install torch torchvision transformers einops kornia
```

Optional: `opencv-python` speeds up morphology/flood-fill (falls back to PIL if absent).

## First run

- BiRefNet downloads a ~900MB model on first run (cached to HuggingFace cache).
- rembg downloads ~176MB (cached to `~/.u2net/`).
- Subsequent runs are fast.

## Verify results

After running, check three things:

1. **Background transparency**: open output in a browser or Photopea — transparent areas should show through, no checkerboard residue.
2. **Subject integrity**: the main content (text, card, icon) should be fully opaque, nothing eaten away.
3. **Edge quality**: zoom into edges — no stair-step jaggies (increase `--feather` if seen), no checkerboard bleed.

If edges have residual checkerboard, or the subject's light-colored areas (e.g. white card background) are being incorrectly removed, read `references/tuning.md`. For a deeper understanding of why this happens and how the algorithm works, read `references/algorithm.md`.

## Test the skill

```bash
python scripts/test_remove_checkerboard.py
# or
python -m pytest scripts/test_remove_checkerboard.py -v
```

37 tests covering period detection, color detection, alpha computation, feather, engine fallback, and edge cases (solid color, gradient, dark images, idempotency).
