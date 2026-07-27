# Remove Checkerboard | 去除棋盘格背景

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

A portable Agent Skill that removes backgrounds from images to produce **real transparent PNGs**. Auto-detects and handles two background types:

1. **Fake checkerboard**: AI generators (ChatGPT, Gemini, DALL·E) sometimes render the *visual symbol* of transparency (gray-white grid) as actual pixels instead of a real alpha channel.
2. **Solid/white background**: flat white or light-colored backgrounds that need to become transparent.

The script **auto-detects** which type — you don't choose.

### Quick start

```bash
# Default: BiRefNet engine + feather 4 (best quality, needs GPU)
python scripts/remove_checkerboard.py input.png -o ./output --engine birefnet --feather 4

# Batch a folder (mixed checkerboard + white-bg images is fine)
python scripts/remove_checkerboard.py ./images -o ./output --engine birefnet --feather 4

# CPU-only (rembg, no GPU needed)
python scripts/remove_checkerboard.py input.png --engine rembg --feather 4
```

### Engines

| Engine | Quality | GPU | Speed |
|---|---|---|---|
| `birefnet` (default) | Best | Required | Slower |
| `rembg` | Good | Not required | Faster |

### When to trigger

- "Remove the checkered background"
- "Make white background transparent"
- "Extract subject from image"
- "去背景" / "去白底" / "棋盘格背景" / "假透明" / "抠图"
- AI-generated images where transparency looks broken

### Project structure

```text
remove-checkerboard/
├── SKILL.md                            Routing rules and detection logic
├── scripts/
│   ├── remove_checkerboard.py          Core: auto-detect + remove background
│   └── test_remove_checkerboard.py     Unit tests
└── references/
    ├── algorithm.md                    Detection algorithm explanation
    ├── engine-guide.md                 BiRefNet vs rembg engine comparison
    └── tuning.md                       Feather / threshold tuning guide
```

---

<a id="中文"></a>
## 中文

可移植 Agent Skill，从图片中去除背景，生成**真正的透明 PNG**。自动识别处理两类背景：

1. **假棋盘格**：AI 生成器（ChatGPT、Gemini、DALL·E）有时会把透明的视觉符号（灰白网格）渲染成真实像素，而不是真正的 alpha 通道。
2. **纯色/白底**：需要变透明的纯白或浅色背景。

脚本**自动识别**类型——无需手动选择。

### 快速开始

```bash
# 默认：BiRefNet 引擎 + 羽化 4（最佳质量，需 GPU）
python scripts/remove_checkerboard.py input.png -o ./output --engine birefnet --feather 4

# 批量处理（棋盘格 + 白底混合图片可一起处理）
python scripts/remove_checkerboard.py ./images -o ./output --engine birefnet --feather 4

# 仅 CPU（rembg，无需 GPU）
python scripts/remove_checkerboard.py input.png --engine rembg --feather 4
```

### 引擎对比

| 引擎 | 质量 | GPU | 速度 |
|---|---|---|---|
| `birefnet`（默认） | 最佳 | 必需 | 较慢 |
| `rembg` | 良好 | 不需要 | 较快 |

### 触发场景

- "去除棋盘格背景"
- "把白底变成透明"
- "从图片中抠出主体"
- AI 生成图片透明度看起来坏了

---

## License | 协议

MIT
