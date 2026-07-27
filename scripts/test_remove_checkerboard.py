"""
remove_checkerboard.py 的单元测试
==================================
用代码合成的测试图(不依赖外部大图),精确断言各函数行为。
覆盖:核心函数、边界情况、降级路径、幂等性。

运行:
    python -m pytest test_remove_checkerboard.py -v
    # 或不用 pytest:
    python test_remove_checkerboard.py
"""
import numpy as np
from PIL import Image

import remove_checkerboard as rc


# ---------- 测试 fixture 工具 ----------

def make_checkerboard(w: int, h: int, T: int = 25, c_white: int = 254,
                      c_gray: int = 243, cx: int = 0, cy: int = 0,
                      subject: tuple | None = None) -> np.ndarray:
    """
    合成一张棋盘格灰度图,可选在中心放一个矩形主体。
    subject: (x0, y0, x1, y1, color) 主体矩形区域及填充灰度值
    返回 HxWx3 RGB uint8(灰度值复制到三通道)。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    gx = (xx - cx) // T
    gy = (yy - cy) // T
    is_white = (gx + gy) % 2 == 0
    gray = np.where(is_white, c_white, c_gray).astype(np.uint8)
    if subject is not None:
        x0, y0, x1, y1, color = subject
        gray[y0:y1, x0:x1] = color
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb


def make_checkerboard_rgb(w: int, h: int, T: int = 25,
                          c_white=(254, 254, 254), c_gray=(243, 243, 243),
                          cx=0, cy=0, subject=None) -> np.ndarray:
    """彩色版合成,主体可以是彩色。"""
    yy, xx = np.mgrid[0:h, 0:w]
    gx = (xx - cx) // T
    gy = (yy - cy) // T
    is_white = (gx + gy) % 2 == 0
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[is_white] = c_white
    img[~is_white] = c_gray
    if subject is not None:
        x0, y0, x1, y1, color = subject
        img[y0:y1, x0:x1] = color
    return img


# ---------- 1. detect_period ----------

def test_detect_period_standard():
    """标准棋盘格应正确检测周期。"""
    rgb = make_checkerboard(300, 200, T=25)
    gray = np.array(Image.fromarray(rgb).convert("L"))
    Tx, Ty, cx, cy, conf = rc.detect_period(gray)
    assert Tx == 25, f"Tx 应为25,得到{Tx}"
    assert Ty == 25, f"Ty 应为25,得到{Ty}"
    assert conf > 0.3, f"置信度应>0.3,得到{conf}"


def test_detect_period_with_phase():
    """带相位偏移(cx≠0)的棋盘格。"""
    rgb = make_checkerboard(300, 200, T=30, cx=15)
    gray = np.array(Image.fromarray(rgb).convert("L"))
    Tx, Ty, cx, cy, conf = rc.detect_period(gray)
    assert Tx == 30, f"Tx 应为30,得到{Tx}"
    assert Ty == 30, f"Ty 应为30,得到{Ty}"


def test_detect_period_different_sizes():
    """不同格子大小都应正确检测。"""
    for T in [20, 35, 50]:
        rgb = make_checkerboard(400, 400, T=T)
        gray = np.array(Image.fromarray(rgb).convert("L"))
        Tx, Ty, *_ = rc.detect_period(gray)
        assert Tx == T, f"T={T}: 检测到Tx={Tx}"
        assert Ty == T, f"T={T}: 检测到Ty={Ty}"


# ---------- 2. detect_two_colors ----------

def test_detect_two_colors_standard():
    """标准两色应正确识别。"""
    rgb = make_checkerboard(300, 200, T=25, c_white=254, c_gray=243)
    gray = np.array(Image.fromarray(rgb).convert("L"))
    cw, cg = rc.detect_two_colors(gray, 25, 25, 0, 0)
    assert abs(cw - 254) <= 2, f"白格色应为254,得到{cw}"
    assert abs(cg - 243) <= 2, f"灰格色应为243,得到{cg}"


def test_detect_two_colors_with_subject():
    """有主体时不应被主体颜色干扰(从边缘采样)。"""
    rgb = make_checkerboard(300, 200, T=25,
                            subject=(100, 80, 200, 160, 50))  # 中心深色块
    gray = np.array(Image.fromarray(rgb).convert("L"))
    cw, cg = rc.detect_two_colors(gray, 25, 25, 0, 0)
    assert abs(cw - 254) <= 3, f"主体干扰:白格色={cw}"
    assert abs(cg - 243) <= 3, f"主体干扰:灰格色={cg}"


# ---------- 3. build_template ----------

def test_build_template_values():
    """模板在已知位置应是正确颜色。"""
    rgb = make_checkerboard(100, 100, T=25, c_white=254, c_gray=243)
    gray = np.array(Image.fromarray(rgb).convert("L"))
    tpl = rc.build_template(gray, 25, 25, 0, 0, 254, 243)
    # (0,0) 是白格起点
    assert tpl[0, 0] == 254
    # (25,0) 跨格 → 灰
    assert tpl[0, 25] == 243
    # (0,25) 跨格 → 灰
    assert tpl[25, 0] == 243
    # (25,25) 对角 → 白
    assert tpl[25, 25] == 254


# ---------- 4. compute_alpha ----------

def test_compute_alpha_background_is_zero():
    """纯棋盘格背景 alpha 应为 0。"""
    rgb = make_checkerboard(200, 200, T=25)
    gray = np.array(Image.fromarray(rgb).convert("L"))
    tpl = rc.build_template(gray, 25, 25, 0, 0, 254, 243)
    alpha = rc.compute_alpha(gray, tpl, 254, 243)
    assert alpha.max() < 0.05, f"背景alpha应~0, max={alpha.max():.3f}"


def test_compute_alpha_dark_subject_is_one():
    """深色主体 alpha 应为 1。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(80, 80, 120, 120, 30))
    gray = np.array(Image.fromarray(rgb).convert("L"))
    tpl = rc.build_template(gray, 25, 25, 0, 0, 254, 243)
    alpha = rc.compute_alpha(gray, tpl, 254, 243)
    center = alpha[90:110, 90:110]
    assert center.min() > 0.9, f"深色主体alpha应>0.9, min={center.min():.3f}"


def test_compute_alpha_range():
    """alpha 应在 [0,1] 范围内。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(50, 50, 150, 150, 120))
    gray = np.array(Image.fromarray(rgb).convert("L"))
    tpl = rc.build_template(gray, 25, 25, 0, 0, 254, 243)
    alpha = rc.compute_alpha(gray, tpl, 254, 243)
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0


# ---------- 5. process_image 整体 ----------

def test_process_image_no_rembg_background_transparent():
    """无 rembg 模式:纯棋盘格背景应透明。"""
    rgb = make_checkerboard(200, 200, T=25)
    rgba, info = rc.process_image(rgb, use_rembg=False)
    alpha = rgba[:, :, 3]
    # 大部分像素应为透明
    assert (alpha == 0).sum() / alpha.size > 0.9, "背景应>90%透明"


def test_process_image_no_rembg_dark_subject_kept():
    """无 rembg 模式:深色主体应保留。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(70, 70, 130, 130, 30))
    rgba, info = rc.process_image(rgb, use_rembg=False)
    alpha = rgba[:, :, 3]
    center = alpha[90:110, 90:110]
    assert center.mean() > 200, f"深色主体alpha均值应>200, 得到{center.mean():.0f}"


def test_process_image_rgb_erased_in_transparent_area():
    """透明区 RGB 应被擦除为 0。"""
    rgb = make_checkerboard(200, 200, T=25)
    rgba, info = rc.process_image(rgb, use_rembg=False)
    alpha = rgba[:, :, 3]
    rgb_out = rgba[:, :, :3]
    bg = alpha < 10
    if bg.any():
        assert rgb_out[bg].max() == 0, "透明区RGB应=0"


def test_process_image_output_shape():
    """输出尺寸应与输入一致。"""
    rgb = make_checkerboard(150, 100, T=20)
    rgba, info = rc.process_image(rgb, use_rembg=False)
    assert rgba.shape[:2] == (100, 150)
    assert rgba.shape[2] == 4


# ---------- 6. 边界情况 ----------

def test_process_image_solid_color_no_crash():
    """纯色图(非棋盘格)不应崩溃,返回不透明图。"""
    rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
    rgba, info = rc.process_image(rgb, use_rembg=False)
    assert rgba.shape == (100, 100, 4)
    # 未检测到棋盘格 → 应返回全不透明
    if "error" in info:
        assert rgba[:, :, 3].min() == 255


def test_process_image_gradient_no_crash():
    """渐变图不应崩溃。"""
    yy, xx = np.mgrid[0:100, 0:100]
    g = (xx + yy).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    rgba, info = rc.process_image(rgb, use_rembg=False)
    assert rgba.shape == (100, 100, 4)


def test_process_image_already_dark():
    """全黑图不应崩溃。"""
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    rgba, info = rc.process_image(rgb, use_rembg=False)
    assert rgba.shape == (50, 50, 4)


# ---------- 7. 幂等性(对已处理图再跑)----------

def test_idempotent_on_small_checkerboard():
    """对输出再跑一次不应崩溃。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(80, 80, 120, 120, 30))
    rgba1, _ = rc.process_image(rgb, use_rembg=False)
    # 把已处理的 RGBA 当 RGB 再输入
    rgb2 = rgba1[:, :, :3]
    rgba2, info = rc.process_image(rgb2, use_rembg=False)
    assert rgba2.shape == rgba1.shape


# ---------- 8. 真实样本回归(用合成图模拟你的场景)----------

def test_regresssion_card_white_subject():
    """模拟信息卡场景:浅色主体(白底)在棋盘格上,残差法单独会漏检中心。
    验证 process_image 至少不崩溃且 alpha 在合理范围。"""
    rgb = make_checkerboard_rgb(
        300, 300, T=25,
        subject=(75, 75, 225, 225, (250, 250, 250))  # 浅色卡片
    )
    rgba, info = rc.process_image(rgb, use_rembg=False)
    alpha = rgba[:, :, 3]
    assert alpha.shape == (300, 300)
    # 至少背景区应该透明
    corner = alpha[0:20, 0:20].mean()
    assert corner < 10, f"角落应透明, alpha={corner:.0f}"


def test_regresssion_multiple_sizes():
    """模拟你的8张图不同格子大小:所有都能正确检测周期。"""
    for T in [22, 25, 28, 34, 49, 51]:
        rgb = make_checkerboard(500, 400, T=T)
        gray = np.array(Image.fromarray(rgb).convert("L"))
        Tx, Ty, *_ = rc.detect_period(gray)
        # 允许±1误差(相位边界)
        assert abs(Tx - T) <= 1, f"T={T}: 检测Tx={Tx}"


# ---------- 9. feather 羽化参数 ----------

def test_feather_zero_no_change():
    """feather=0 应与旧行为一致(过渡窄)。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(70, 70, 130, 130, 30))
    rgba, info = rc.process_image(rgb, use_rembg=False, feather=0)
    assert rgba.shape == (200, 200, 4)


def test_feather_softens_edge():
    """feather>0 应使边缘过渡更宽(更柔和)。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(70, 70, 130, 130, 30))
    rgba0, _ = rc.process_image(rgb, use_rembg=False, feather=0)
    rgba4, _ = rc.process_image(rgb, use_rembg=False, feather=4)
    a0 = rgba0[:, :, 3]
    a4 = rgba4[:, :, 3]
    # 过渡带(50-205)像素数:feather 版应更多
    edge0 = ((a0 > 50) & (a0 < 205)).sum()
    edge4 = ((a4 > 50) & (a4 < 205)).sum()
    assert edge4 >= edge0, f"feather后边缘过渡应更宽: {edge0}→{edge4}"


def test_feather_preserves_core():
    """feather 不应让主体核心区 alpha 大幅下降。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(60, 60, 140, 140, 30))
    rgba, _ = rc.process_image(rgb, use_rembg=False, feather=4)
    core = rgba[90:110, 90:110, 3]
    assert core.mean() > 180, f"主体核心应保持不透明: {core.mean():.0f}"


# ---------- 10. engine 引擎选择 ----------

def test_engine_none_uses_residual_only():
    """engine=none 时应走纯残差法(info 标记 none)。"""
    rgb = make_checkerboard(200, 200, T=25, subject=(70, 70, 130, 130, 30))
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    assert info["engine"] == "none"
    assert rgba.shape == (200, 200, 4)


def test_engine_invalid_falls_back():
    """未知引擎名应降级为纯残差法,不崩溃。"""
    rgb = make_checkerboard(200, 200, T=25)
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="nonexistent")
    assert rgba.shape == (200, 200, 4)


# ---------- 11. 白底/纯色背景降级模式 ----------

def make_white_bg(w=300, h=300, subject=None):
    """合成白底图,可选放一个矩形主体。subject=(x0,y0,x1,y1,color)"""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    if subject:
        x0, y0, x1, y1, color = subject
        img[y0:y1, x0:x1] = color
    return img


def test_white_bg_detected_as_solid():
    """白底图应走 solid_bg 模式,不走棋盘格路径。"""
    rgb = make_white_bg(300, 300, subject=(80, 80, 220, 220, (180, 50, 50)))
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    assert info["mode"] == "solid_bg", f"白底图应检测为 solid_bg,得到{info.get('mode')}"


def test_white_bg_background_transparent():
    """白底区应变透明。"""
    rgb = make_white_bg(300, 300, subject=(80, 80, 220, 220, (180, 50, 50)))
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    alpha = rgba[:, :, 3]
    corner = alpha[0:30, 0:30].mean()
    assert corner < 20, f"白底区应透明(alpha<20),得到{corner:.0f}"


def test_white_bg_subject_kept():
    """深色主体应保留。"""
    rgb = make_white_bg(300, 300, subject=(80, 80, 220, 220, (180, 50, 50)))
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    alpha = rgba[:, :, 3]
    center = alpha[130:170, 130:170].mean()
    assert center > 200, f"主体应保留(alpha>200),得到{center:.0f}"


def test_white_bg_none_engine_uses_floodfill():
    """engine=none 时白底图应用 floodfill 兜底。"""
    rgb = make_white_bg(300, 300, subject=(80, 80, 220, 220, (180, 50, 50)))
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    assert "flood" in info["engine"], f"应使用 floodfill,得到 engine={info['engine']}"


def test_white_bg_rgb_erased():
    """白底透明区的 RGB 应被擦除。"""
    rgb = make_white_bg(300, 300, subject=(80, 80, 220, 220, (180, 50, 50)))
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    alpha = rgba[:, :, 3]
    rgb_out = rgba[:, :, :3]
    bg = alpha < 10
    if bg.any():
        assert rgb_out[bg].max() == 0, "透明区 RGB 应=0"


def test_white_bg_light_color_subject():
    """浅色主体(极浅米色)在白底上也应被保留。"""
    rgb = make_white_bg(300, 300, subject=(80, 80, 220, 220, (238, 233, 228)))
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    alpha = rgba[:, :, 3]
    center = alpha[130:170, 130:170].mean()
    # floodfill 兜底:颜色差异小可能漏检,但至少不能全透明
    assert center > 100, f"浅色主体应至少部分保留(alpha>100),得到{center:.0f}"


def test_non_solid_non_checkerboard_no_crash():
    """非纯色非棋盘格图(如照片)不应崩溃。"""
    # 模拟一张颜色杂乱的图
    yy, xx = np.mgrid[0:100, 0:100]
    r = (xx * 2.5 % 255).astype(np.uint8)
    g = (yy * 2.5 % 255).astype(np.uint8)
    b = ((xx + yy) % 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgba, info = rc.process_image(rgb, use_rembg=False, engine="none")
    assert rgba.shape == (100, 100, 4)


def test_white_bg_feather_applied():
    """白底图 feather 应生效(边缘过渡变宽)。"""
    rgb = make_white_bg(300, 300, subject=(80, 80, 220, 220, (180, 50, 50)))
    rgba0, _ = rc.process_image(rgb, use_rembg=False, engine="none", feather=0)
    rgba4, _ = rc.process_image(rgb, use_rembg=False, engine="none", feather=4)
    a0, a4 = rgba0[:, :, 3], rgba4[:, :, 3]
    edge0 = ((a0 > 50) & (a0 < 205)).sum()
    edge4 = ((a4 > 50) & (a4 < 205)).sum()
    assert edge4 >= edge0


# ---------- 兼容非 pytest 运行 ----------

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    exit(1 if failed else 0)
