"""
棋盘格假透明 → 真透明背景 PNG
=============================
适用场景:AI 生成的图片把编辑器的棋盘格背景"画"进了 RGB 像素里(非真 Alpha 透明)。
本脚本检测这种规则周期纹理并移除,生成真正的透明 PNG。普适于任意主体
(艺术字、人像、信息卡、图标、横幅等)。

双引擎融合算法:
1. 棋盘格残差法(擅长背景):
   - 自动测周期 T、两色 c1/c2、相位 (cx, cy)
   - 重建理想棋盘格模板,算每像素偏离度 → 确信背景(alpha→0)
   - 对深色主体也能直接识别(alpha→1)
2. rembg / U2Net 语义分割(擅长主体):
   - 区分"主体浅色区"(如卡片白底)与"棋盘格白格"——这是纯颜色法分不开的
3. 融合:rembg 为主体判断主权威,残差法只在 rembg 也认可处强化背景透明

依赖:pillow, numpy, scipy, rembg, onnxruntime(首次自动下载 U2Net 模型 ~176MB)
无 rembg 时自动降级为纯残差法(对浅色主体效果会打折)。

用法:
    python remove_checkerboard.py <输入图或目录> [-o 输出目录]
    python remove_checkerboard.py 01_xxx.png            # 同目录生成 _nobg.png
    python remove_checkerboard.py . -o ./out            # 批量
    python remove_checkerboard.py . --no-rembg          # 不用 rembg(纯残差法)
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image


# ---------- 1. 自动检测棋盘格参数 ----------

def detect_period(gray: np.ndarray) -> tuple[int, int, int, int, int]:
    """
    返回 (Tx, Ty, cx, cy, confidence)
    Tx/Ty: 水平/竖直方向单格边长
    cx/cy: 相位(第一个白格左上角的坐标)
    confidence: 检测置信度 0~1
    """
    h, w = gray.shape

    # 在背景区采样测周期:对每一行/列做差分,统计"同色 run 长度"直方图
    def best_period(axis: int) -> tuple[int, float]:
        cnt: Counter = Counter()
        dim = h if axis else w
        n_samples = min(40, dim)
        step = max(1, dim // n_samples)
        indices = list(range(0, dim, step))
        for i in indices:
            if axis == 0:
                if i >= h:
                    continue
                line = gray[i, :] >= 249
            else:
                if i >= w:
                    continue
                line = gray[:, i] >= 249
            line = (gray[i, :] if axis == 0 else gray[:, i]) >= 249  # 白=True
            cur = line[0]; ln = 1
            for v in line[1:]:
                if v == cur:
                    ln += 1
                else:
                    if 8 <= ln <= 80:  # 合理的格子边长范围
                        cnt[ln] += 1
                    cur = v; ln = 1
            if 8 <= ln <= 80:
                cnt[ln] += 1
        if not cnt:
            return 0, 0.0
        # 取出现最多的长度,但棋盘格相邻同色会合并(当主体穿过),所以多数票即周期
        best, votes = cnt.most_common(1)[0]
        total = sum(cnt.values())
        # 同色 run 可能是 1T 或 2T(两格同色偶然连一起),取众数最稳
        return int(best), votes / total if total else 0.0

    Tx, conf_x = best_period(0)
    Ty, conf_y = best_period(1)
    if Tx == 0 or Ty == 0:
        return 0, 0, 0, 0, 0.0

    # 测相位:找一个确信的背景小窗口,对齐到周期
    # 扫角落找方差最低的 Tx*Ty 窗口(纯背景方差小)
    best_var = float("inf")
    best_cx, best_cy = 0, 0
    for cy in range(0, min(h, 60), 3):
        for cx in range(0, min(w, 60), 3):
            win = gray[cy:cy+Ty, cx:cx+Tx]
            if win.size == 0:
                continue
            v = win.var()
            if v < best_var:
                best_var = v
                best_cx, best_cy = cx, cy
    # 把相位规整到周期内
    cx = best_cx % Tx if Tx else 0
    cy = best_cy % Ty if Ty else 0
    conf = (conf_x + conf_y) / 2

    # 验证:边缘区是否真的呈现棋盘格交替(而非主体边缘误判)
    # 在边缘条带采样多个 Tx 宽的块,看块内是否有两色交替
    if not _verify_checker_pattern(gray, Tx, Ty, cx, cy):
        return 0, 0, 0, 0, 0.0
    return Tx, Ty, cx, cy, conf


def _verify_checker_pattern(gray: np.ndarray, Tx: int, Ty: int,
                            cx: int, cy: int) -> bool:
    """
    验证检测到的周期确实是棋盘格图案(而非主体边界误判)。
    方法:在边缘条带(确信背景区)里,沿水平方向数连续同色 run 的长度。
    真棋盘格:run 长度高度集中在 Tx 附近,且有多次交替。
    非棋盘格(纯色背景):run 极长(整条边都是一色)或无交替。
    """
    h, w = gray.shape
    margin = max(Tx, Ty, 20)
    # 二值化阈值:用边缘条带的均值(在两色中间,比中位数更稳)
    band = gray[0:margin].flatten()
    mid = float(band.mean())
    run_lengths = []
    # 扫描上边缘条带的多行
    for y in range(2, min(margin, h - 2), max(1, margin // 5)):
        row = gray[y, :] >= mid + 3
        cur = row[0]; ln = 1
        for v in row[1:]:
            if v == cur:
                ln += 1
            else:
                run_lengths.append(ln)
                cur = v; ln = 1
        run_lengths.append(ln)
    # 同样扫左边缘
    for x in range(2, min(margin, w - 2), max(1, margin // 5)):
        col = gray[:, x] >= mid + 3
        cur = col[0]; ln = 1
        for v in col[1:]:
            if v == cur:
                ln += 1
            else:
                run_lengths.append(ln)
                cur = v; ln = 1
        run_lengths.append(ln)

    if not run_lengths:
        return True
    rl = np.array(run_lengths)
    # 真棋盘格:run 长度应集中在 Tx 附近
    near_T = np.sum((rl >= Tx * 0.7) & (rl <= Tx * 1.3))
    total = len(rl)
    # 要求:接近 Tx 的 run 占比 > 30%,且总交替次数够多(至少 6 次)
    near_ratio = near_T / total if total else 0
    return near_ratio > 0.3 and total >= 6


def detect_two_colors(gray: np.ndarray, Tx: int, Ty: int, cx: int, cy: int) -> tuple[int, int]:
    """
    从边缘条带(确信背景区)用亮度直方图双峰法估两个棋盘格颜色。
    双峰法不依赖相位分类,即使边缘恰好只落在一种格子上也能工作。
    """
    h, w = gray.shape
    margin = max(Tx, Ty, 20)
    band = np.concatenate([
        gray[0:margin, :].flatten(),
        gray[h-margin:h, :].flatten(),
        gray[:, 0:margin].flatten(),
        gray[:, w-margin:w].flatten(),
    ])
    # 只看高亮度区(棋盘格两色都很亮)
    bright = band[band >= 235]
    if len(bright) < 50:
        return 254, 243
    # 直方图,找两个峰
    hist = np.bincount(bright.astype(int), minlength=256)
    # 平滑
    kernel = np.ones(3) / 3
    hist_s = np.convolve(hist, kernel, mode="same")
    # 在 235-255 找局部极大
    peaks = []
    for i in range(236, 255):
        if hist_s[i] > hist_s[i-1] and hist_s[i] >= hist_s[i+1] and hist_s[i] > len(bright) * 0.02:
            peaks.append((i, hist_s[i]))
    if len(peaks) >= 2:
        peaks.sort(key=lambda p: -p[1])
        top2 = sorted([peaks[0][0], peaks[1][0]])
        return top2[1], top2[0]  # (白, 灰)
    # 兜底:用分位数
    hi = int(np.percentile(bright, 90))
    lo = int(np.percentile(bright, 40))
    return hi if hi != lo else 254, lo if hi != lo else 243


# ---------- 2. 重建模板 + 残差 ----------

def build_template(gray: np.ndarray, Tx: int, Ty: int, cx: int, cy: int,
                   c_white: int, c_gray: int) -> np.ndarray:
    """构建与图像同尺寸的理想棋盘格模板。"""
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gx = ((xx - cx) // Tx).astype(int)
    gy = ((yy - cy) // Ty).astype(int)
    is_white = ((gx + gy) % 2 == 0)
    tpl = np.where(is_white, c_white, c_gray).astype(np.float32)
    return tpl


def compute_alpha(gray: np.ndarray, template: np.ndarray,
                  c_white: int, c_gray: int,
                  Tx: int = 25, Ty: int = 25, cx: int = 0, cy: int = 0) -> np.ndarray:
    """
    棋盘格残差法:只做它擅长的事——识别"确定性背景"。
    像素到最近棋盘格色的距离 d_min:
      d_min <= bg_thresh  → 确信背景,alpha=0
      d_min >= fg_thresh  → 确信主体(深色),alpha=1
      中间过渡
    主体内部的浅色区(颜色接近棋盘格)交给 rembg 语义分割处理。
    """
    g = gray.astype(np.float32)
    d_white = np.abs(g - c_white)
    d_gray = np.abs(g - c_gray)
    d_min = np.minimum(d_white, d_gray)

    gap = max(1.0, float(c_white - c_gray))
    bg_thresh = gap * 0.6 + 2.0
    fg_thresh = gap * 1.5 + 8.0

    alpha = np.zeros_like(g)
    mask_mid = (d_min > bg_thresh) & (d_min < fg_thresh)
    alpha[mask_mid] = (d_min[mask_mid] - bg_thresh) / (fg_thresh - bg_thresh)
    alpha[d_min >= fg_thresh] = 1.0
    return alpha


# ---------- 3. 形态学清理 ----------

# ---------- 2b. 白底检测(无棋盘格时降级)----------

def detect_solid_background(rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
    """
    检测纯色/近纯色背景(常见:白底、浅灰底)。
    返回 (背景mask True=背景, 背景色名)。无纯色背景返回 (None, "")。
    用边缘条带采样:若边缘像素颜色高度集中(方差小),判定为纯色背景。
    """
    h, w = rgb.shape[:2]
    margin = max(10, min(h, w) // 20)
    # 四周边缘条带
    band = np.concatenate([
        rgb[0:margin].reshape(-1, 3),
        rgb[h-margin:h].reshape(-1, 3),
        rgb[:, 0:margin].reshape(-1, 3),
        rgb[:, w-margin:w].reshape(-1, 3),
    ]).astype(np.float32)
    # 聚类中心(均值)和离散度
    center = band.mean(axis=0)
    spread = band.std(axis=0).mean()  # 颜色离散度
    if spread > 25:  # 边缘颜色太杂,不是纯色背景
        return None, ""
    bg_color = tuple(center.astype(int))
    # 判定背景色名
    r, g, b = bg_color
    if min(r, g, b) > 200 and max(r, g, b) - min(r, g, b) < 30:
        name = "white"
    elif min(r, g, b) > 180:
        name = "light"
    else:
        name = "color"
    return bg_color, name


def floodfill_background(rgb: np.ndarray, bg_color: tuple,
                         tol: int = 12) -> np.ndarray:
    """
    从图像四角 floodfill,标记与背景色相近的连通区域。
    返回 0~1 的 alpha(1=主体,0=背景)。tol 为颜色容差。
    用于无神经网络时的白底降级。
    """
    try:
        import cv2
        h, w = rgb.shape[:2]
        # cv2.floodFill 需要彩色图
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # 从四角开始填充
        for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
            cv2.floodFill(bgr, mask, seed, (0, 0, 0),
                          (tol, tol, tol), (tol, tol, tol),
                          cv2.FLOODFILL_MASK_ONLY)
        # mask: 1=被填充(背景), 0=未填充(主体)。偏移 1 像素。
        bg_mask = mask[1:h+1, 1:w+1] == 1
        alpha = (~bg_mask).astype(np.float32)
        return alpha
    except ImportError:
        # 无 cv2,用简单的颜色距离法
        diff = np.abs(rgb.astype(np.float32) - np.array(bg_color, np.float32))
        dist = diff.max(axis=2)
        alpha = (dist > tol).astype(np.float32)
        return alpha


def clean_alpha(alpha: np.ndarray, feather: int = 0) -> np.ndarray:
    """
    形态学清理 + 边缘羽化(抗锯齿)。
    feather>0 时对 alpha 做较大半径高斯模糊,消除楼梯台阶锯齿。
    只影响过渡带(主体内部和背景区不受影响,因为模糊后仍趋近 0/1)。
    """
    try:
        import cv2
        a = (alpha * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
        a = cv2.GaussianBlur(a, (3, 3), 0)
        if feather > 0:
            # 羽化:大半径高斯模糊边缘,过渡从~6px扩到~10-15px
            a = cv2.GaussianBlur(a, (feather * 2 + 1, feather * 2 + 1), feather * 0.8)
        return a.astype(np.float32) / 255.0
    except ImportError:
        from PIL import ImageFilter
        img = (alpha * 255).astype(np.uint8)
        pil = Image.fromarray(img).filter(ImageFilter.MedianFilter(3))
        if feather > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(feather))
        return np.array(pil).astype(np.float32) / 255.0


# ---------- 4. rembg 语义分割(兜底) ----------

_rembg_session = None

def get_rembg_alpha(rgb: np.ndarray) -> np.ndarray | None:
    """用 rembg(U2Net)做语义分割,返回 0~1 alpha。失败返回 None。"""
    global _rembg_session
    try:
        from rembg import remove, new_session
        if _rembg_session is None:
            _rembg_session = new_session("u2net")
        out = remove(Image.fromarray(rgb), session=_rembg_session, only_mask=True)
        a = np.array(out).astype(np.float32) / 255.0
        return a
    except Exception as e:
        print(f"  [rembg 不可用: {e}]")
        return None


_birefnet_model = None

def get_birefnet_alpha(rgb: np.ndarray) -> np.ndarray | None:
    """用 BiRefNet_dynamic 做语义分割,返回 0~1 alpha。需要 GPU + torch。"""
    global _birefnet_model
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        if _birefnet_model is None:
            torch.set_float32_matmul_precision('high')
            _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                'ZhengPeng7/BiRefNet_dynamic', trust_remote_code=True
            ).to('cuda').eval().half()
        pil = Image.fromarray(rgb)
        w, h = pil.size
        # 对齐到 32 的倍数(BiRefNet patch 切分要求)
        nw, nh = max(32, (w // 32) * 32), max(32, (h // 32) * 32)
        resized = pil.resize((nw, nh), Image.BILINEAR)
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        inp = t(resized).unsqueeze(0).to('cuda').half()
        with torch.no_grad():
            preds = _birefnet_model(inp)[-1].sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(preds).resize(pil.size)
        return np.array(mask).astype(np.float32) / 255.0
    except Exception as e:
        print(f"  [BiRefNet 不可用: {e}]")
        return None


def get_engine_alpha(rgb: np.ndarray, engine: str) -> np.ndarray | None:
    """按引擎名分发。rembg=U²-Net,birefnet=SOTA,none=纯残差法。"""
    if engine == "rembg":
        return get_rembg_alpha(rgb)
    elif engine == "birefnet":
        return get_birefnet_alpha(rgb)
    return None


# ---------- 5. 主流程(融合) ----------

def process_image(rgb: np.ndarray, use_rembg: bool = True,
                  feather: int = 0, engine: str = "rembg") -> tuple[np.ndarray, dict]:
    """
    输入 HxWx3 RGB uint8,返回 (RGBA uint8, info)。
    融合策略:
      - 棋盘格残差法识别确定性背景(alpha→0)
      - 语义分割引擎(rembg/birefnet)识别主体(覆盖残差法在浅色区的误判)
      - feather>0 时对 alpha 边缘羽化,消除楼梯台阶锯齿
    """
    gray = np.array(Image.fromarray(rgb).convert("L"))
    Tx, Ty, cx, cy, conf = detect_period(gray)
    info = {"Tx": Tx, "Ty": Ty, "cx": cx, "cy": cy, "conf": f"{conf:.2f}"}

    # === 降级路径:检测不到棋盘格 → 白底/纯色背景模式 ===
    if Tx == 0:
        info["mode"] = "solid_bg"
        bg_color, bg_name = detect_solid_background(rgb)
        info["bg"] = bg_name
        if bg_color is None:
            # 既不是棋盘格,也不是纯色背景 → 尝试纯引擎分割
            alpha_engine = get_engine_alpha(rgb, engine) if (use_rembg or engine != "rembg") else None
            if alpha_engine is not None:
                if alpha_engine.shape != gray.shape:
                    alpha_engine = np.array(
                        Image.fromarray((alpha_engine * 255).astype(np.uint8)).resize(
                            (gray.shape[1], gray.shape[0]), Image.BILINEAR)
                    ).astype(np.float32) / 255.0
                alpha = alpha_engine
                info["engine"] = engine
            else:
                # 无引擎可用,无法处理
                info["engine"] = "unavailable"
                rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
                return rgba, info
        else:
            # 纯色背景:引擎优先,floodfill 兜底
            alpha_engine = get_engine_alpha(rgb, engine) if (use_rembg or engine != "rembg") else None
            if alpha_engine is not None:
                if alpha_engine.shape != gray.shape:
                    alpha_engine = np.array(
                        Image.fromarray((alpha_engine * 255).astype(np.uint8)).resize(
                            (gray.shape[1], gray.shape[0]), Image.BILINEAR)
                    ).astype(np.float32) / 255.0
                # 引擎 + floodfill 取并集(引擎可能漏掉浅色区,floodfill 补上)
                alpha_flood = floodfill_background(rgb, bg_color)
                alpha = np.maximum(alpha_engine, alpha_flood)
                info["engine"] = engine + "+flood"
            else:
                # 无引擎:纯 floodfill
                alpha = floodfill_background(rgb, bg_color)
                info["engine"] = "flood"

        alpha = clean_alpha(alpha, feather=feather)
        alpha_u8 = (alpha * 255).astype(np.uint8)
        rgb_clean = rgb.copy()
        rgb_clean[alpha_u8 < 10] = 0
        rgba = np.dstack([rgb_clean, alpha_u8])
        return rgba, info

    # === 主路径:棋盘格模式 ===
    info["mode"] = "checkerboard"

    c_white, c_gray = detect_two_colors(gray, Tx, Ty, cx, cy)
    info["c_white"], info["c_gray"] = c_white, c_gray

    template = build_template(gray, Tx, Ty, cx, cy, c_white, c_gray)
    alpha_cb = compute_alpha(gray, template, c_white, c_gray, Tx, Ty, cx, cy)

    # 融合语义分割引擎
    alpha_engine = None
    if use_rembg or engine != "rembg":
        alpha_engine = get_engine_alpha(rgb, engine)
    if alpha_engine is not None:
        if alpha_engine.shape != gray.shape:
            alpha_engine = np.array(
                Image.fromarray((alpha_engine * 255).astype(np.uint8)).resize(
                    (gray.shape[1], gray.shape[0]), Image.BILINEAR)
            ).astype(np.float32) / 255.0
        # 融合策略:
        # - 引擎是主体判断的主权威(能区分卡片白底 vs 棋盘格白格)
        # - 残差法只在引擎也认为非主体的区域强化透明(消除棋盘格残留)
        #   ——不能单独用残差法压引擎,否则主体浅色区会被误杀
        # - 残差法确信主体(深色区)的,补充引擎可能漏检的部分
        sure_bg_cb = alpha_cb < 0.1           # 残差法判为背景
        engine_bg = alpha_engine < 0.3         # 引擎也判为背景
        true_bg = sure_bg_cb & engine_bg       # 两者都判背景 → 确信透明
        alpha = np.maximum(alpha_engine, alpha_cb)
        alpha[true_bg] = 0.0
        info["engine"] = engine + "/fused"
    else:
        alpha = alpha_cb
        info["engine"] = "none"

    alpha = clean_alpha(alpha, feather=feather)

    # 擦除透明区 RGB
    alpha_u8 = (alpha * 255).astype(np.uint8)
    rgb_clean = rgb.copy()
    rgb_clean[alpha_u8 < 10] = 0
    rgba = np.dstack([rgb_clean, alpha_u8])
    return rgba, info


def main():
    ap = argparse.ArgumentParser(description="移除 AI 生成的棋盘格假透明背景")
    ap.add_argument("input", help="输入 PNG/JPG 文件或目录")
    ap.add_argument("-o", "--output", help="输出目录(默认与输入同级)")
    ap.add_argument("--engine", choices=["rembg", "birefnet", "none"], default="rembg",
                    help="语义分割引擎:rembg(U²-Net,默认), birefnet(SOTA,需GPU), none(纯残差法)")
    ap.add_argument("--feather", type=int, default=0,
                    help="边缘羽化半径(像素),消除楼梯锯齿。推荐 2-4。默认 0")
    ap.add_argument("--no-rembg", action="store_true",
                    help="等价于 --engine none(仅棋盘格残差法)")
    args = ap.parse_args()

    inp = args.input
    if os.path.isdir(inp):
        files = [os.path.join(inp, f) for f in sorted(os.listdir(inp))
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    else:
        files = [inp]

    out_dir = args.output or (inp if os.path.isdir(inp) else os.path.dirname(inp) or ".")
    os.makedirs(out_dir, exist_ok=True)

    engine = "none" if args.no_rembg else args.engine
    for f in files:
        img = Image.open(f).convert("RGB")
        rgb = np.array(img)
        rgba, info = process_image(rgb, use_rembg=(engine != "none"),
                                   feather=args.feather, engine=engine)
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, f"{base}_nobg.png")
        Image.fromarray(rgba).save(out_path)
        flag = "" if "error" not in info else "  [!]"
        print(f"{flag}{os.path.basename(f)} → {os.path.basename(out_path)}  {info}")


if __name__ == "__main__":
    main()
