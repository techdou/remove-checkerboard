# Tuning Guide

## Feather (anti-aliasing)

`--feather N` applies a Gaussian blur to the alpha channel edges after subject/background separation. It softens hard pixel boundaries into gradual transitions.

### What problem it solves

Checkerboard detection naturally produces hard alpha edges (0 or 255, little in between). On diagonal lines, curves, and rounded corners, this creates visible stair-step jaggies — the "staircase effect" of pixel-aligned rendering.

### Measured impact

Tested on an info card image (golden frame edge, 80 sample points along the left border):

| feather | Edge transition width (median) | Visual effect |
|---|---|---|
| 0 | 6px | Jaggy stair-steps on frame edges |
| 2 | 8px | Slightly improved |
| **4** | **10px** | **Sweet spot — jaggies gone, detail preserved** |
| 6 | 12px | Soft, starting to lose fine detail |
| 8 | 14px | Very soft — thin strokes look fuzzy |

### Choosing feather by content type

Different content types tolerate different feather values:

- **Thick frames, cards, solid shapes** (info cards, banners): `--feather 6` or `8` — edges look rounder and more natural. These subjects have no fine detail to lose.
- **Thin strokes, art text, icons, hairlines**: `--feather 2` or `4` — higher values make strokes look fat/fuzzy because the soft edge eats into the stroke body.
- **Mixed content** (a card with text inside): `--feather 4` is the safe compromise.

### Batch processing mixed content

If processing a folder with mixed image types, you can't set per-image feather from the CLI. Two options:

1. **Use `--feather 4`** for the whole batch — it's the best single value across all types.
2. **Run twice with different output dirs**, then manually pick:
   ```bash
   python scripts/remove_checkerboard.py ./images -o ./out_sharp --engine birefnet --feather 2
   python scripts/remove_checkerboard.py ./images -o ./out_soft --engine birefnet --feather 6
   ```

### The trade-off

Feathering isn't free. Soft edges mean: when compositing onto a new background with very different color, you may see a faint "halo" of the original checkerboard color blended in at the edge. For most uses (white/solid backgrounds in PPT, web), this is invisible. For dark backgrounds with high contrast, use lower feather.

## Common issues and fixes

### Checkerboard residue on edges

**Symptom**: faint grid pattern visible at subject edges after processing.

**Cause**: the residual method's `bg_thresh` is too strict, leaving partial checkerboard pixels near the subject boundary.

**Fix**: this is rare with the fusion approach (the engine mask overrides it). If it happens, try a different engine, or slightly increase feather to blur out the residue.

### Subject's white/light area removed

**Symptom**: a white card's background gets cut out (becomes transparent), only text/borders remain.

**Cause**: using `--engine none` on a light-colored subject. Pure residual method can't distinguish white card from white checkerboard.

**Fix**: use `--engine birefnet` or `--engine rembg`. The neural network recognizes the subject semantically.

### Subject has "holes"

**Symptom**: solid areas inside the subject show partial transparency.

**Cause**: engine under-segmentation (common with thin/light elements) or feather too high on small features.

**Fix**: lower feather, or try the other engine. BiRefNet and rembg have different blind spots — what one misses, the other often catches.

### Art text strokes partially missing

**Symptom**: some brush strokes or thin character parts are faded or absent.

**Cause**: BiRefNet sometimes treats very thin strokes as non-subject. This is a known limitation of dichotomous segmentation models trained on natural objects.

**Fix**: try `--engine rembg` — U²-Net's salient object detection is more sensitive to any non-background region, including thin strokes. If both miss strokes, the original image may have had those areas blend into the checkerboard (low contrast) — no algorithm can recover what isn't visible.

## Debug output

The script prints detected parameters per image:

```
04_card.png → 04_card_nobg.png  {'Tx': 28, 'Ty': 28, 'cx': 0, 'cy': 0, 'conf': '0.68', 'c_white': 254, 'c_gray': 244, 'engine': 'birefnet/fused'}
```

- `Tx, Ty`: detected grid cell size in pixels. Should match the visible checker.
- `cx, cy`: phase offset. Can be non-zero.
- `conf`: detection confidence (0-1). Below 0.2 means the period detection is uncertain — the image may not have a checkerboard, or it's heavily obscured by the subject.
- `c_white, c_gray`: the two detected checkerboard colors. If these are nearly equal (gap < 3), the image probably isn't a checkerboard.

If `conf` is very low or `c_white ≈ c_gray`, the image may not actually have a fake checkerboard — check the input.
