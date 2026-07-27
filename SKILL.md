---
name: remove-checkerboard
description: Remove backgrounds from images to produce transparent PNGs. Handles two background types automatically — (1) fake checkerboard patterns baked into AI-generated image pixels (ChatGPT/Gemini/DALL-E "fake transparency"), and (2) solid/white backgrounds. Use whenever the user mentions checkerboard patterns in image pixels, fake transparency from AI generators, transparent background not working, 棋盘格背景, 假透明, 白底变透明, 去白底, 抠图, or wants to extract subjects from images. Triggers on "remove the checkered background", "real transparent PNG", "make white background transparent", "去背景", and images where a grid or solid color blocks the subject. The skill auto-detects background type (checkerboard vs solid) — no need to specify.
---

# Remove Checkerboard & White Background

Produce real transparent PNGs from two common background problems:

1. **Fake checkerboard**: AI generators (ChatGPT, Gemini, DALL·E) sometimes render the *visual symbol* of transparency (gray-white grid) as actual pixels instead of a real alpha channel.
2. **Solid/white background**: images with white or light-colored backgrounds that need to become transparent.

The script **auto-detects** which type it is — you don't choose. It checks for a checkerboard pattern first; if none found, it falls back to solid-background mode.

**Quick check**: open the image in an editor. If "transparent" areas show a checkerboard and there's no alpha channel → fake checkerboard. If the background is a flat white/solid color → solid background. Either way, use this skill.

## Quick start

```bash
# Default: BiRefNet engine + feather 4 (best quality, needs GPU)
python scripts/remove_checkerboard.py input.png -o ./output --engine birefnet --feather 4

# Batch a folder (mixed checkerboard + white-bg images is fine)
python scripts/remove_checkerboard.py ./images -o ./output --engine birefnet --feather 4

# CPU-only (rembg, no GPU needed)
python scripts/remove_checkerboard.py input.png --engine rembg --feather 4
```

## How background detection works

The script auto-detects the background type per image — no flag needed:

1. **Checkerboard mode**: detects a two-color periodic grid, removes it via residual method + semantic segmentation fusion. For AI-generated "fake transparency".
2. **Solid-background mode**: if no checkerboard found, detects the solid background color (white, light gray, etc.) from image edges, then removes it via semantic segmentation + flood fill. For white-bg photos, product shots, scanned documents.

The mode is printed in the output (`mode=checkerboard` or `mode=solid_bg`). You can mix checkerboard and solid-background images in the same batch — each is handled automatically.

## Choose your engine

| Engine | Quality | Speed | Needs GPU | When to use |
|---|---|---|---|---|
| `birefnet` (default) | Best | ~0.5s/img | Yes (CUDA) | You have an NVIDIA GPU and want the best edges |
| `rembg` | Good | ~0.3s/img | No | No GPU, or CPU-only machine |
| `none` | Basic | ~0.1s/img | No | Quick preview, offline. Checkerboard uses residual-only; solid-bg uses flood fill |

Read `references/engine-guide.md` for the full comparison and edge cases.

## Feather (anti-aliasing)

`--feather N` controls edge softness. Works the same in both modes.

- `0` — sharp but jagged on diagonals/curves
- `4` — **recommended**, smooth edges without losing detail
- `8` — very soft, may make thin strokes (art text) look slightly fuzzy

For per-image-type tuning, read `references/tuning.md`.

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

24 tests covering period detection, color detection, alpha computation, feather, engine fallback, and edge cases (solid color, gradient, dark images, idempotency).
