"""
评估真实数据融合结果
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel_h, sobel_v, laplace, gaussian
from skimage.transform import resize
import rasterio
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='评估真实数据融合结果（支持按run子目录）')
    parser.add_argument('--real_data_dir', default='./data/real_data', type=str,
                        help='真实数据目录（包含 MS_up_800 / PAN_cut_800）')
    parser.add_argument('--results_dir', default='./results/real', type=str,
                        help='融合结果目录（包含 fusion_8ch_*.tif）。如果启用了run子目录，可配合 --latest_run')
    parser.add_argument('--latest_run', action='store_true',
                        help='如果 results_dir 下存在 run_*/ 子目录，则自动选择最新的一个进行评估')

    # QNR (no-reference) params
    parser.add_argument('--scale', default=4, type=int,
                        help='PAN相对MS的分辨率比例（用于QNR降采样）。常见为4。')
    parser.add_argument('--qnr_alpha', default=1.0, type=float, help='QNR中 Dλ 的权重 α')
    parser.add_argument('--qnr_beta', default=1.0, type=float, help='QNR中 Ds 的权重 β')
    return parser.parse_args()


def _load_run_config(results_dir: str) -> dict:
    cfg_path = Path(results_dir) / 'run_config.json'
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _pick_latest_run_dir(base_results_dir: str) -> str:
    base = Path(base_results_dir)
    if not base.exists():
        return base_results_dir
    run_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith('run_')]
    if not run_dirs:
        return str(base)
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(run_dirs[0])


def _count_run_dirs(base_results_dir: str) -> int:
    base = Path(base_results_dir)
    if not base.exists():
        return 0
    return sum(1 for p in base.iterdir() if p.is_dir() and p.name.startswith('run_'))


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    return (x - x_min) / (x_max - x_min + 1e-8)


def _uiqi(a: np.ndarray, b: np.ndarray) -> float:
    """Universal Image Quality Index (global).

    Returns a value in [-1, 1]. Higher is better.
    """
    a = a.astype(np.float32, copy=False).ravel()
    b = b.astype(np.float32, copy=False).ravel()
    if a.size == 0 or b.size == 0:
        return float('nan')
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())

    denom = (mu_a * mu_a + mu_b * mu_b) * (var_a + var_b) + 1e-8
    return float((4.0 * mu_a * mu_b * cov) / denom)


def _downsample(img2d: np.ndarray, scale: int) -> np.ndarray:
    """Gaussian blur + resize downsampling."""
    if scale <= 1:
        return img2d.astype(np.float32, copy=False)
    h, w = img2d.shape
    out_h = max(1, int(round(h / scale)))
    out_w = max(1, int(round(w / scale)))
    # Mild blur to mimic sensor MTF before decimation (approximation)
    blurred = gaussian(img2d.astype(np.float32, copy=False), sigma=max(scale / 2.0, 1.0), preserve_range=True)
    return resize(
        blurred,
        (out_h, out_w),
        order=1,
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def calculate_qnr_metrics(ms_up: np.ndarray, fused: np.ndarray, pan: np.ndarray, scale: int = 4,
                          alpha: float = 1.0, beta: float = 1.0) -> dict:
    """Approximate QNR / Ds / Dλ using available real-data inputs.

    Notes:
    - Classic QNR assumes access to MS at its native (lower) resolution.
      Here we only have MS_up_800; we approximate MS by downsampling MS_up.
    - This still works well as a *relative* metric for ablations (e.g., spatial_weight 0 vs 0.02).
    """
    if pan is None:
        return {}

    # Prepare shapes
    if pan.ndim == 3:
        pan2d = pan[0]
    else:
        pan2d = pan

    if ms_up.ndim != 3 or fused.ndim != 3:
        return {}

    c = min(ms_up.shape[0], fused.shape[0])
    h = min(ms_up.shape[1], fused.shape[1], pan2d.shape[0])
    w = min(ms_up.shape[2], fused.shape[2], pan2d.shape[1])

    ms_up = ms_up[:c, :h, :w]
    fused = fused[:c, :h, :w]
    pan2d = pan2d[:h, :w]

    # Normalize per-band / per-image
    ms_n = _normalize_01(ms_up)
    fused_n = _normalize_01(fused)
    pan_n = _normalize_01(pan2d)

    # Downsample MS_up and fused to approximate MS native resolution
    ms_lr = np.stack([_downsample(ms_n[i], scale) for i in range(c)], axis=0)
    fused_lr = np.stack([_downsample(fused_n[i], scale) for i in range(c)], axis=0)
    pan_lr = _downsample(pan_n, scale)

    # D_lambda: spectral distortion (band-to-band structure difference)
    d_lambda_vals = []
    for i in range(c):
        for j in range(i + 1, c):
            q_ms = _uiqi(ms_lr[i], ms_lr[j])
            q_f = _uiqi(fused_lr[i], fused_lr[j])
            if not (np.isnan(q_ms) or np.isnan(q_f)):
                d_lambda_vals.append(abs(q_ms - q_f))
    d_lambda = float(np.mean(d_lambda_vals)) if d_lambda_vals else float('nan')

    # D_s: spatial distortion (band-to-PAN structure difference across scales)
    d_s_vals = []
    for i in range(c):
        q_ms = _uiqi(ms_lr[i], pan_lr)
        q_f = _uiqi(fused_n[i], pan_n)
        if not (np.isnan(q_ms) or np.isnan(q_f)):
            d_s_vals.append(abs(q_ms - q_f))
    d_s = float(np.mean(d_s_vals)) if d_s_vals else float('nan')

    if np.isnan(d_lambda) or np.isnan(d_s):
        return {}

    qnr = float((max(0.0, 1.0 - d_lambda) ** alpha) * (max(0.0, 1.0 - d_s) ** beta))
    return {
        'qnr': qnr,
        'd_lambda': d_lambda,
        'd_s': d_s,
        'qnr_scale': int(scale),
    }


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False).ravel()
    b = b.astype(np.float32, copy=False).ravel()
    if a.size == 0 or b.size == 0:
        return float('nan')
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8)
    return float(np.sum(a * b) / denom)


def calculate_pan_spatial_metrics(pan: np.ndarray, fused: np.ndarray) -> dict:
    """No-reference spatial metrics that relate fused image to PAN.

    These are useful because PSNR/SSIM/SAM here are computed vs MS_up (which is
    smoother). PAN-guided spatial losses can improve sharpness but not improve
    MS-reference metrics.
    """
    if pan is None:
        return {}

    if pan.ndim == 3:
        pan2d = pan[0]
    else:
        pan2d = pan

    if fused.ndim == 3:
        fused_intensity = fused.mean(axis=0)
    else:
        fused_intensity = fused

    # Crop to common size
    h = min(pan2d.shape[0], fused_intensity.shape[0])
    w = min(pan2d.shape[1], fused_intensity.shape[1])
    pan2d = pan2d[:h, :w]
    fused_intensity = fused_intensity[:h, :w]

    pan_n = _normalize_01(pan2d)
    fused_n = _normalize_01(fused_intensity)

    # Vectorized edge/high-pass features
    pan_gx = sobel_h(pan_n)
    pan_gy = sobel_v(pan_n)
    fused_gx = sobel_h(fused_n)
    fused_gy = sobel_v(fused_n)

    pan_gm = np.sqrt(pan_gx * pan_gx + pan_gy * pan_gy)
    fused_gm = np.sqrt(fused_gx * fused_gx + fused_gy * fused_gy)

    pan_hp = laplace(pan_n)
    fused_hp = laplace(fused_n)

    return {
        # Pearson correlation of gradient magnitude (edge map similarity)
        'pan_grad_corr': _pearson_corr(pan_gm, fused_gm),
        # Pearson correlation of Laplacian (high-frequency similarity)
        'pan_hp_corr': _pearson_corr(pan_hp, fused_hp),
        # Edge energy ratio (how sharp fused is relative to PAN)
        'pan_edge_energy_ratio': float((fused_gm.mean() + 1e-8) / (pan_gm.mean() + 1e-8)),
    }


def load_image_pair(ms_path, fused_path):
    """加载MS和融合图像对"""
    try:
        with rasterio.open(ms_path) as src:
            ms_data = src.read()

        with rasterio.open(fused_path) as src:
            fused_data = src.read()

        return ms_data, fused_data
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None, None


def calculate_metrics(ms, fused):
    """计算评估指标"""
    # 确保形状匹配
    if ms.shape != fused.shape:
        print(f"形状不匹配: MS{ms.shape} vs Fused{fused.shape}")
        # 取最小公共尺寸
        min_c = min(ms.shape[0], fused.shape[0])
        min_h = min(ms.shape[1], fused.shape[1])
        min_w = min(ms.shape[2], fused.shape[2])

        ms = ms[:min_c, :min_h, :min_w]
        fused = fused[:min_c, :min_h, :min_w]

    # 归一化
    ms_norm = (ms - ms.min()) / (ms.max() - ms.min() + 1e-8)
    fused_norm = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    # 计算PSNR
    psnr_values = []
    for c in range(min(3, ms_norm.shape[0])):  # 只计算前3个波段
        psnr_val = psnr(ms_norm[c], fused_norm[c], data_range=1.0)
        psnr_values.append(psnr_val)
    avg_psnr = np.mean(psnr_values)

    # 计算SSIM
    ssim_values = []
    for c in range(min(3, ms_norm.shape[0])):
        ssim_val = ssim(ms_norm[c], fused_norm[c], data_range=1.0, win_size=3)
        ssim_values.append(ssim_val)
    avg_ssim = np.mean(ssim_values)

    # 计算SAM
    ms_flat = ms_norm.reshape(ms_norm.shape[0], -1)
    fused_flat = fused_norm.reshape(fused_norm.shape[0], -1)

    dot_product = np.sum(ms_flat * fused_flat, axis=0)
    norm_ms = np.sqrt(np.sum(ms_flat ** 2, axis=0))
    norm_fused = np.sqrt(np.sum(fused_flat ** 2, axis=0))

    epsilon = 1e-8
    cos_theta = dot_product / (norm_ms * norm_fused + epsilon)
    cos_theta = np.clip(cos_theta, -1 + epsilon, 1 - epsilon)

    sam_rad = np.arccos(cos_theta)
    sam_deg = np.degrees(sam_rad)
    avg_sam = np.mean(sam_deg)

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'sam': avg_sam,
        'shape': ms.shape
    }


def visualize_comparison(ms, fused, pan, save_path, sample_name):
    """可视化对比"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 用于显示的归一化（避免16位数据直接imshow导致全白）
    ms_disp = (ms - ms.min()) / (ms.max() - ms.min() + 1e-8)
    fused_disp = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
    pan_disp = None
    if pan is not None:
        pan_disp = (pan - pan.min()) / (pan.max() - pan.min() + 1e-8)

    # 获取RGB波段（假设前3个波段）
    if ms_disp.shape[0] >= 3:
        ms_rgb = ms_disp[:3]
        fused_rgb = fused_disp[:3]
    else:
        ms_rgb = ms_disp[0:1]  # 单波段
        fused_rgb = fused_disp[0:1]

    # 调整维度用于显示
    if len(ms_rgb.shape) == 3 and ms_rgb.shape[0] == 3:
        ms_rgb_display = np.transpose(ms_rgb, (1, 2, 0))
        fused_rgb_display = np.transpose(fused_rgb, (1, 2, 0))
    else:
        ms_rgb_display = ms_rgb[0]
        fused_rgb_display = fused_rgb[0]

    # 1. PAN图像
    if pan_disp is not None:
        axes[0, 0].imshow(pan_disp.squeeze(), cmap='gray')
    else:
        axes[0, 0].text(0.5, 0.5, 'PAN N/A', ha='center', va='center')
    axes[0, 0].set_title('PAN Image')
    axes[0, 0].axis('off')

    # 2. MS图像（RGB）
    if len(ms_rgb_display.shape) == 3:
        axes[0, 1].imshow(np.clip(ms_rgb_display, 0, 1))
    else:
        axes[0, 1].imshow(ms_rgb_display, cmap='gray')
    axes[0, 1].set_title('MS Image (Reference)')
    axes[0, 1].axis('off')

    # 3. 融合结果（RGB）
    if len(fused_rgb_display.shape) == 3:
        axes[0, 2].imshow(np.clip(fused_rgb_display, 0, 1))
    else:
        axes[0, 2].imshow(fused_rgb_display, cmap='gray')
    axes[0, 2].set_title('Fused Image')
    axes[0, 2].axis('off')

    # 4. 差异图
    diff = np.abs(ms_rgb_display - fused_rgb_display)
    if len(diff.shape) == 3:
        diff = diff.mean(axis=2)

    im = axes[0, 3].imshow(diff, cmap='hot')
    axes[0, 3].set_title('Difference (MS - Fused)')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # 5-6. 各波段对比
    for c in range(min(4, ms.shape[0])):
        row = 1
        col = c
        if c < 4:
            axes[row, col].imshow(ms_disp[c], cmap='gray')
            axes[row, col].set_title(f'MS Band {c + 1}')
            axes[row, col].axis('off')

    plt.suptitle(f'Sample: {sample_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_real_data(real_data_dir: str = './data/real_data', results_dir: str = './results/real',
                       qnr_scale: int = 4, qnr_alpha: float = 1.0, qnr_beta: float = 1.0):
    """评估真实数据融合结果"""
    print("=" * 60)
    print("真实数据融合质量评估")
    print("=" * 60)

    # 路径设置
    real_data_dir = real_data_dir
    results_dir = results_dir

    # 检查目录
    if not os.path.exists(real_data_dir):
        print(f"错误: 真实数据目录不存在: {real_data_dir}")
        return

    if not os.path.exists(results_dir):
        print(f"错误: 结果目录不存在: {results_dir}")
        return

    # 获取MS和PAN文件
    ms_dir = Path(real_data_dir) / 'MS_up_800'
    pan_dir = Path(real_data_dir) / 'PAN_cut_800'

    if not ms_dir.exists():
        print(f"错误: MS目录不存在: {ms_dir}")
        return

    if not pan_dir.exists():
        print(f"错误: PAN目录不存在: {pan_dir}")
        return

    # 获取文件列表
    ms_files = sorted(list(ms_dir.glob('*.tif')))
    pan_files = sorted(list(pan_dir.glob('*.tif')))
    fused_files = sorted(list(Path(results_dir).glob('fusion_8ch_*.tif')))

    print(f"找到 {len(ms_files)} 个MS文件")
    print(f"找到 {len(pan_files)} 个PAN文件")
    print(f"找到 {len(fused_files)} 个融合结果")

    # 确保数量匹配
    min_pairs = min(len(ms_files), len(fused_files))
    if min_pairs == 0:
        print("没有找到可评估的数据对")
        return

    print(f"\n开始评估 {min_pairs} 个样本...")

    all_metrics = []
    all_visualizations = []

    for i in range(min_pairs):
        print(f"\n处理样本 {i + 1}/{min_pairs}:")
        print(f"  MS: {ms_files[i].name}")
        print(f"  PAN: {pan_files[i].name if i < len(pan_files) else 'N/A'}")
        print(f"  Fused: {fused_files[i].name}")

        # 加载数据
        ms_data, fused_data = load_image_pair(ms_files[i], fused_files[i])

        if ms_data is None or fused_data is None:
            print(f"  加载失败，跳过此样本")
            continue

        # 加载PAN（用于可视化）
        pan_data = None
        if i < len(pan_files):
            with rasterio.open(pan_files[i]) as src:
                pan_data = src.read()

        # 计算指标
        metrics = calculate_metrics(ms_data, fused_data)

        # 计算与PAN相关的无参考空间指标（更能反映“清晰度/细节注入”）
        pan_spatial = calculate_pan_spatial_metrics(pan_data, fused_data)
        metrics.update(pan_spatial)

        # QNR / Ds / Dλ（近似无参考指标）
        qnr_metrics = calculate_qnr_metrics(ms_data, fused_data, pan_data, scale=qnr_scale, alpha=qnr_alpha, beta=qnr_beta)
        metrics.update(qnr_metrics)
        metrics['sample_name'] = ms_files[i].stem
        metrics['ms_file'] = ms_files[i].name
        metrics['fused_file'] = fused_files[i].name

        print(f"  形状: {metrics['shape']}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  SAM:  {metrics['sam']:.2f}°")

        if 'pan_grad_corr' in metrics:
            print(f"  PAN-GradCorr: {metrics['pan_grad_corr']:.4f}")
            print(f"  PAN-HighPassCorr: {metrics['pan_hp_corr']:.4f}")
            print(f"  EdgeEnergyRatio(Fused/PAN): {metrics['pan_edge_energy_ratio']:.4f}")

        if 'qnr' in metrics:
            print(f"  QNR(scale={metrics['qnr_scale']}): {metrics['qnr']:.4f}")
            print(f"  D_lambda: {metrics['d_lambda']:.4f}")
            print(f"  D_s: {metrics['d_s']:.4f}")

        all_metrics.append(metrics)

        # 可视化
        save_path = str(Path(results_dir) / f'comparison_{i:04d}.png')
        visualize_comparison(ms_data, fused_data, pan_data, save_path, ms_files[i].stem)
        all_visualizations.append(save_path)
        print(f"  可视化已保存: {save_path}")

    if not all_metrics:
        print("没有成功评估任何样本")
        return

    # 计算平均指标
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    avg_sam = np.mean([m['sam'] for m in all_metrics])

    qnr_vals = [m.get('qnr') for m in all_metrics if m.get('qnr') is not None]
    d_lambda_vals = [m.get('d_lambda') for m in all_metrics if m.get('d_lambda') is not None]
    d_s_vals = [m.get('d_s') for m in all_metrics if m.get('d_s') is not None]
    avg_qnr = float(np.mean(qnr_vals)) if qnr_vals else None
    avg_d_lambda = float(np.mean(d_lambda_vals)) if d_lambda_vals else None
    avg_d_s = float(np.mean(d_s_vals)) if d_s_vals else None

    # PAN相关空间指标（若PAN存在）
    pan_grad_corr_vals = [m.get('pan_grad_corr') for m in all_metrics if m.get('pan_grad_corr') is not None]
    pan_hp_corr_vals = [m.get('pan_hp_corr') for m in all_metrics if m.get('pan_hp_corr') is not None]
    pan_edge_ratio_vals = [m.get('pan_edge_energy_ratio') for m in all_metrics if m.get('pan_edge_energy_ratio') is not None]

    avg_pan_grad_corr = float(np.mean(pan_grad_corr_vals)) if pan_grad_corr_vals else None
    avg_pan_hp_corr = float(np.mean(pan_hp_corr_vals)) if pan_hp_corr_vals else None
    avg_pan_edge_ratio = float(np.mean(pan_edge_ratio_vals)) if pan_edge_ratio_vals else None

    print(f"样本数量: {len(all_metrics)}")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均 SAM:  {avg_sam:.2f}°")

    if avg_pan_grad_corr is not None:
        print(f"平均 PAN-GradCorr: {avg_pan_grad_corr:.4f}")
        print(f"平均 PAN-HighPassCorr: {avg_pan_hp_corr:.4f}")
        print(f"平均 EdgeEnergyRatio(Fused/PAN): {avg_pan_edge_ratio:.4f}")

    if avg_qnr is not None:
        print(f"平均 QNR: {avg_qnr:.4f}")
        print(f"平均 D_lambda: {avg_d_lambda:.4f}")
        print(f"平均 D_s: {avg_d_s:.4f}")

    # 性能评估
    print("\n性能评估:")
    print("-" * 40)

    if avg_psnr > 30:
        print(f"  PSNR:  优秀 (>30 dB)")
    elif avg_psnr > 25:
        print(f"  PSNR:  良好 (25-30 dB)")
    else:
        print(f"  PSNR:  一般 (<25 dB)")

    if avg_ssim > 0.9:
        print(f"  SSIM:  优秀 (>0.9)")
    elif avg_ssim > 0.8:
        print(f"  SSIM:  良好 (0.8-0.9)")
    else:
        print(f"  SSIM:  一般 (<0.8)")

    if avg_sam < 10:
        print(f"  SAM:   优秀 (<10°)")
    elif avg_sam < 20:
        print(f"  SAM:   良好 (10-20°)")
    else:
        print(f"  SAM:   一般 (>20°)")

    print("-" * 40)

    # 生成报告
    generate_report(all_metrics, avg_psnr, avg_ssim, avg_sam, results_dir)

    return all_metrics


def generate_report(metrics, avg_psnr, avg_ssim, avg_sam, results_dir: str):
    """生成评估报告"""
    report_path = str(Path(results_dir) / 'evaluation_report.txt')

    run_cfg = _load_run_config(results_dir)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("真实数据融合质量评估报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估目录: {Path(results_dir).resolve()}\n")
        if run_cfg:
            args = run_cfg.get('args') or {}
            sw = args.get('spatial_weight')
            fch = args.get('feat_ch')
            lr = args.get('lr')
            ep = args.get('epochs')
            f.write("Run配置(来自run_config.json):\n")
            f.write(f"  spatial_weight: {sw}\n")
            f.write(f"  feat_ch: {fch}\n")
            f.write(f"  lr: {lr}\n")
            f.write(f"  epochs: {ep}\n")
        f.write(f"样本数量: {len(metrics)}\n\n")

        f.write("平均指标:\n")
        f.write("-" * 40 + "\n")
        f.write(f"PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"SAM:  {avg_sam:.2f}°\n\n")

        qnr_vals = [m.get('qnr') for m in metrics if m.get('qnr') is not None]
        d_lambda_vals = [m.get('d_lambda') for m in metrics if m.get('d_lambda') is not None]
        d_s_vals = [m.get('d_s') for m in metrics if m.get('d_s') is not None]
        if qnr_vals:
            f.write("QNR(无参考，近似):\n")
            f.write("-" * 40 + "\n")
            f.write(f"QNR: {float(np.mean(qnr_vals)):.4f}\n")
            f.write(f"D_lambda: {float(np.mean(d_lambda_vals)):.4f}\n")
            f.write(f"D_s: {float(np.mean(d_s_vals)):.4f}\n\n")

        pan_grad_corr_vals = [m.get('pan_grad_corr') for m in metrics if m.get('pan_grad_corr') is not None]
        pan_hp_corr_vals = [m.get('pan_hp_corr') for m in metrics if m.get('pan_hp_corr') is not None]
        pan_edge_ratio_vals = [m.get('pan_edge_energy_ratio') for m in metrics if m.get('pan_edge_energy_ratio') is not None]
        if pan_grad_corr_vals:
            f.write("PAN空间一致性(无参考):\n")
            f.write("-" * 40 + "\n")
            f.write(f"PAN-GradCorr: {float(np.mean(pan_grad_corr_vals)):.4f}\n")
            f.write(f"PAN-HighPassCorr: {float(np.mean(pan_hp_corr_vals)):.4f}\n")
            f.write(f"EdgeEnergyRatio(Fused/PAN): {float(np.mean(pan_edge_ratio_vals)):.4f}\n\n")

        f.write("性能评估:\n")
        f.write("-" * 40 + "\n")
        f.write(f"PSNR: {'优秀' if avg_psnr > 30 else '良好' if avg_psnr > 25 else '一般'}\n")
        f.write(f"SSIM: {'优秀' if avg_ssim > 0.9 else '良好' if avg_ssim > 0.8 else '一般'}\n")
        f.write(f"SAM:  {'优秀' if avg_sam < 10 else '良好' if avg_sam < 20 else '一般'}\n\n")

        f.write("样本详细指标:\n")
        f.write("=" * 60 + "\n")
        for i, m in enumerate(metrics):
            f.write(f"\n样本 {i + 1}: {m['sample_name']}\n")
            f.write(f"  MS文件: {m['ms_file']}\n")
            f.write(f"  融合文件: {m['fused_file']}\n")
            f.write(f"  形状: {m['shape']}\n")
            f.write(f"  PSNR: {m['psnr']:.2f} dB\n")
            f.write(f"  SSIM: {m['ssim']:.4f}\n")
            f.write(f"  SAM:  {m['sam']:.2f}°\n")
            if m.get('pan_grad_corr') is not None:
                f.write(f"  PAN-GradCorr: {m['pan_grad_corr']:.4f}\n")
                f.write(f"  PAN-HighPassCorr: {m['pan_hp_corr']:.4f}\n")
                f.write(f"  EdgeEnergyRatio(Fused/PAN): {m['pan_edge_energy_ratio']:.4f}\n")
            if m.get('qnr') is not None:
                f.write(f"  QNR(scale={m.get('qnr_scale', '')}): {m['qnr']:.4f}\n")
                f.write(f"  D_lambda: {m['d_lambda']:.4f}\n")
                f.write(f"  D_s: {m['d_s']:.4f}\n")

    print(f"\n评估报告已保存: {report_path}")

    # 生成HTML报告
    html_path = str(Path(results_dir) / 'evaluation_report.html')
    generate_html_report(metrics, avg_psnr, avg_ssim, avg_sam, html_path)
    print(f"HTML报告已保存: {html_path}")


def generate_html_report(metrics, avg_psnr, avg_ssim, avg_sam, output_path):
    """生成HTML报告"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>真实数据融合质量评估报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #007acc;
            }}
            .metrics-summary {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                flex-wrap: wrap;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                min-width: 200px;
                margin: 10px;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-name {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .samples-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .sample-card {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                background: #f9f9f9;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #007acc;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 真实数据融合质量评估报告</h1>
                <p>评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="metrics-summary">
                <div class="metric-card">
                    <div class="metric-name">PSNR</div>
                    <div class="metric-value">{avg_psnr:.2f} dB</div>
                    <div class="metric-rating">{'优秀' if avg_psnr > 30 else '良好' if avg_psnr > 25 else '一般'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">SSIM</div>
                    <div class="metric-value">{avg_ssim:.4f}</div>
                    <div class="metric-rating">{'优秀' if avg_ssim > 0.9 else '良好' if avg_ssim > 0.8 else '一般'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">SAM</div>
                    <div class="metric-value">{avg_sam:.2f}°</div>
                    <div class="metric-rating">{'优秀' if avg_sam < 10 else '良好' if avg_sam < 20 else '一般'}</div>
                </div>
            </div>

            <h2>PAN空间一致性(无参考)</h2>
            <p style="color:#444;">PSNR/SSIM/SAM 是对比 MS_up_800（更平滑）。如果你加入 PAN 引导的空间损失让结果更锐利，这些指标未必会上升；下面指标更能反映与 PAN 的细节一致性。</p>
            <table>
                <tr>
                    <th>PAN-GradCorr</th>
                    <th>PAN-HighPassCorr</th>
                    <th>EdgeEnergyRatio(Fused/PAN)</th>
                </tr>
                <tr>
                    <td>{float(np.mean([m.get('pan_grad_corr') for m in metrics if m.get('pan_grad_corr') is not None])) if any(m.get('pan_grad_corr') is not None for m in metrics) else 'N/A'}</td>
                    <td>{float(np.mean([m.get('pan_hp_corr') for m in metrics if m.get('pan_hp_corr') is not None])) if any(m.get('pan_hp_corr') is not None for m in metrics) else 'N/A'}</td>
                    <td>{float(np.mean([m.get('pan_edge_energy_ratio') for m in metrics if m.get('pan_edge_energy_ratio') is not None])) if any(m.get('pan_edge_energy_ratio') is not None for m in metrics) else 'N/A'}</td>
                </tr>
            </table>

            <h2>QNR / Ds / Dλ (无参考，近似)</h2>
            <p style="color:#444;">这里按常用 QNR 公式近似计算：先将 MS_up 和融合结果降采样到更低分辨率（由 --scale 指定，常用4），再计算谱失真 Dλ 与空域失真 Ds，最后得到 QNR。用于对比不同训练设置的相对变化更可靠。</p>
            <table>
                <tr>
                    <th>QNR</th>
                    <th>Dλ</th>
                    <th>Ds</th>
                </tr>
                <tr>
                    <td>{float(np.mean([m.get('qnr') for m in metrics if m.get('qnr') is not None])) if any(m.get('qnr') is not None for m in metrics) else 'N/A'}</td>
                    <td>{float(np.mean([m.get('d_lambda') for m in metrics if m.get('d_lambda') is not None])) if any(m.get('d_lambda') is not None for m in metrics) else 'N/A'}</td>
                    <td>{float(np.mean([m.get('d_s') for m in metrics if m.get('d_s') is not None])) if any(m.get('d_s') is not None for m in metrics) else 'N/A'}</td>
                </tr>
            </table>

            <h2>样本详情</h2>
            <table>
                <tr>
                    <th>样本</th>
                    <th>MS文件</th>
                    <th>融合文件</th>
                    <th>PSNR (dB)</th>
                    <th>SSIM</th>
                    <th>SAM (°)</th>
                    <th>PAN-GradCorr</th>
                    <th>PAN-HighPassCorr</th>
                    <th>QNR</th>
                    <th>Dλ</th>
                    <th>Ds</th>
                </tr>
    """

    for i, m in enumerate(metrics):
        grad_corr_str = '' if m.get('pan_grad_corr') is None else f"{m['pan_grad_corr']:.4f}"
        hp_corr_str = '' if m.get('pan_hp_corr') is None else f"{m['pan_hp_corr']:.4f}"
        qnr_str = '' if m.get('qnr') is None else f"{m['qnr']:.4f}"
        dlam_str = '' if m.get('d_lambda') is None else f"{m['d_lambda']:.4f}"
        ds_str = '' if m.get('d_s') is None else f"{m['d_s']:.4f}"
        html_content += f"""
                <tr>
                    <td>样本 {i + 1}</td>
                    <td>{m['ms_file']}</td>
                    <td>{m['fused_file']}</td>
                    <td>{m['psnr']:.2f}</td>
                    <td>{m['ssim']:.4f}</td>
                    <td>{m['sam']:.2f}</td>
                    <td>{grad_corr_str}</td>
                    <td>{hp_corr_str}</td>
                    <td>{qnr_str}</td>
                    <td>{dlam_str}</td>
                    <td>{ds_str}</td>
                </tr>
        """

    html_content += f"""
            </table>

            <h2>可视化对比</h2>
            <div class="samples-grid">
    """

    for i in range(len(metrics)):
        img_path = f'./comparison_{i:04d}.png'
        html_content += f"""
                <div class="sample-card">
                    <h3>样本 {i + 1}</h3>
                    <img src="{img_path}" style="width: 100%; height: auto;">
                    <p>PSNR: {metrics[i]['psnr']:.2f} dB | SSIM: {metrics[i]['ssim']:.4f}</p>
                </div>
        """

    html_content += f"""
            </div>

            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
                <p>空谱融合系统 | 评估完成</p>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    args = parse_args()

    results_dir = args.results_dir

    # Friendly hint: multiple runs exist but latest_run not chosen
    try:
        base = Path(results_dir)
        if base.exists() and base.is_dir() and base.name == 'real' and not args.latest_run:
            run_cnt = _count_run_dirs(results_dir)
            if run_cnt >= 2:
                print(f"提示：检测到 {run_cnt} 个 run_* 子目录，可用 --latest_run 自动评估最新一次输出，或用 --results_dir 指定某个 run_*/ 目录。")
    except Exception:
        pass

    if args.latest_run:
        results_dir = _pick_latest_run_dir(results_dir)

    # 确保目录存在（评估结果也写入该目录）
    os.makedirs(results_dir, exist_ok=True)

    # 运行评估
    evaluate_real_data(
        real_data_dir=args.real_data_dir,
        results_dir=results_dir,
        qnr_scale=args.scale,
        qnr_alpha=args.qnr_alpha,
        qnr_beta=args.qnr_beta,
    )