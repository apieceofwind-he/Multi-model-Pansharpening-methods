"""
基于残差学习的轻量级卷积神经网络模型训练 - 自监督版本
修复版：解决损失值异常问题
"""

import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tifffile as tiff
import numpy as np
from tqdm import tqdm
import warnings
import argparse
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio
from pathlib import Path

warnings.filterwarnings('ignore')


# ===================== 1. 命令行参数解析 =====================
def parse_args():
    parser = argparse.ArgumentParser(description='自监督轻量Pansharpening融合模型')

    # 基础路径参数
    parser.add_argument('--data_root', default='./data', type=str,
                        help='数据根目录')

    # 训练参数
    parser.add_argument('--batch_size', default=4, type=int, help='批次大小')
    parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
    parser.add_argument('--lr', default=1e-4, type=float, help='学习率')
    parser.add_argument('--feat_ch', default=32, type=int, help='特征通道数')

    # 损失权重（调整为更合理的值）
    parser.add_argument('--recon_weight', default=1.0, type=float, help='重建损失权重')
    parser.add_argument('--spatial_weight', default=0.2, type=float, help='空间损失权重')  # 降低
    parser.add_argument('--spectral_weight', default=0.1, type=float, help='光谱损失权重')  # 大幅降低

    # # 损失权重（调整为更合理的值）
    # parser.add_argument('--recon_weight', default=1.0, type=float, help='重建损失权重')
    # parser.add_argument('--spatial_weight', default=0.05, type=float, help='空间损失权重')  # 降低
    # parser.add_argument('--spectral_weight', default=0.1, type=float, help='光谱损失权重')  # 大幅降低

    # 设备参数
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str, help='训练设备')

    args = parser.parse_args()
    return args


# ===================== 2. 修复的损失函数 =====================
class StablePanGuidedSpatialLoss(nn.Module):
    """稳定的空间损失函数"""

    def __init__(self):
        super().__init__()

        # 定义稳定的卷积核
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('lap', lap)

    def gradient_map(self, x):
        """计算梯度图（添加稳定性处理）"""
        # 确保输入在合理范围内
        x = torch.clamp(x, 0.0, 1.0)

        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.abs(gx) + torch.abs(gy)

    def laplacian_map(self, x):
        """计算拉普拉斯图"""
        x = torch.clamp(x, 0.0, 1.0)
        return F.conv2d(x, self.lap, padding=1)

    def forward(self, fused_ms, pan):
        """计算空间损失（添加数值稳定性）"""
        # 强度分量计算
        fused_i = torch.clamp(fused_ms.mean(dim=1, keepdim=True), 0.0, 1.0)
        pan = torch.clamp(pan, 0.0, 1.0)

        # 梯度一致性（添加稳定性处理）
        fused_grad = self.gradient_map(fused_i)
        pan_grad = self.gradient_map(pan)
        grad_loss = F.l1_loss(fused_grad, pan_grad)

        # 拉普拉斯一致性
        fused_lap = self.laplacian_map(fused_i)
        pan_lap = self.laplacian_map(pan)
        lap_loss = F.l1_loss(torch.abs(fused_lap), torch.abs(pan_lap))

        return grad_loss, lap_loss


class StableSelfSupervisedLoss(nn.Module):
    """稳定的自监督损失函数"""

    def __init__(self, recon_weight=1.0, spatial_weight=0.05, spectral_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.spatial_weight = spatial_weight
        self.spectral_weight = spectral_weight

        self.spatial_loss_fn = StablePanGuidedSpatialLoss()

    def degrade_image(self, hr_image, scale_factor=4):
        """稳定的图像退化模拟"""
        if scale_factor <= 1:
            return hr_image

        batch_size, channels, height, width = hr_image.shape
        new_height = max(height // scale_factor, 1)
        new_width = max(width // scale_factor, 1)

        # 双三次插值下采样+上采样
        degraded = F.interpolate(hr_image, size=(new_height, new_width),
                                 mode='bicubic', align_corners=False)
        degraded_hr = F.interpolate(degraded, size=(height, width),
                                    mode='bicubic', align_corners=False)

        return torch.clamp(degraded_hr, 0.0, 1.0)

    def stable_spectral_loss(self, fused, ms_lr):
        """稳定的光谱损失计算"""
        # 上采样MS到融合结果尺寸
        ms_up = F.interpolate(ms_lr, size=fused.shape[2:],
                              mode='bicubic', align_corners=False)

        # 简单的L1损失（更稳定）
        spectral_loss = F.l1_loss(fused, ms_up)

        return spectral_loss

    def forward(self, fused, pan, ms_lr):
        """稳定的损失计算"""
        losses = {}

        # 确保输入在合理范围内
        fused = torch.clamp(fused, 0.0, 1.0)
        pan = torch.clamp(pan, 0.0, 1.0)
        ms_lr = torch.clamp(ms_lr, 0.0, 1.0)

        # 1. 重建损失
        fused_degraded = self.degrade_image(fused)
        recon_loss = F.l1_loss(fused_degraded, ms_lr)
        losses['recon'] = recon_loss

        # 2. 空间损失
        grad_loss, lap_loss = self.spatial_loss_fn(fused, pan)
        spatial_loss = grad_loss + 0.2 * lap_loss
        losses['spatial'] = spatial_loss

        # 3. 光谱损失（使用稳定版本）
        spectral_loss = self.stable_spectral_loss(fused, ms_lr)
        losses['spectral'] = spectral_loss

        # 总损失（添加数值检查）
        total_loss = (self.recon_weight * recon_loss +
                      self.spatial_weight * spatial_loss +
                      self.spectral_weight * spectral_loss)

        # 检查损失值是否合理
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(
                f"警告: 损失值异常 - recon: {recon_loss:.4f}, spatial: {spatial_loss:.4f}, spectral: {spectral_loss:.4f}")
            # 使用默认损失
            total_loss = recon_loss + 0.1 * spatial_loss + 0.01 * spectral_loss

        losses['total'] = total_loss

        return total_loss, losses


# ===================== 3. 简化的模型定义 =====================
class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        return self.act(x + residual)


class SimplePansharpen(nn.Module):
    """简化但稳定的模型"""

    def __init__(self, ms_ch=8, pan_ch=1, feat_ch=32):
        super().__init__()

        # PAN分支
        self.pan_conv = nn.Sequential(
            nn.Conv2d(pan_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # MS分支
        self.ms_conv = nn.Sequential(
            nn.Conv2d(ms_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, ms_ch, 1)
        )

    def forward(self, ms, pan):
        # MS上采样
        ms_up = F.interpolate(ms, scale_factor=1, mode='bicubic', align_corners=False)

        # 特征提取
        pan_feat = self.pan_conv(pan)
        ms_feat = self.ms_conv(ms_up)

        # 融合
        fused_feat = torch.cat([ms_feat, pan_feat], dim=1)
        output = self.fusion(fused_feat)

        return torch.clamp(output, 0.0, 1.0)


# ===================== 4. 稳定的数据集 ======================
class StablePansharpenDataset(Dataset):
    """稳定的数据集加载"""

    def __init__(self, data_root, phase='train'):
        self.phase = phase
        self.data_root = data_root

        if phase == 'train':
            data_dir = os.path.join(data_root, 'train_data', 'train')
        elif phase == 'test':
            data_dir = os.path.join(data_root, 'test_data', 'test')
        else:  # real_test
            data_dir = os.path.join(data_root, 'real_data')

        self.data_dir = data_dir
        self.file_pairs = self._collect_file_pairs()
        print(f"【{phase}阶段】找到 {len(self.file_pairs)} 个样本")

    def _collect_file_pairs(self):
        """收集文件对"""
        file_pairs = []

        if self.phase in ['train', 'test']:
            # 训练/测试阶段
            pan_files = sorted(glob.glob(os.path.join(self.data_dir, '*_pan.tif')))
            mul_files = sorted(glob.glob(os.path.join(self.data_dir, '*_mul.tif')))

            # 按文件名配对
            pan_dict = {}
            for f in pan_files:
                base_name = os.path.basename(f).replace('_pan.tif', '')
                pan_dict[base_name] = f

            mul_dict = {}
            for f in mul_files:
                base_name = os.path.basename(f).replace('_mul.tif', '')
                mul_dict[base_name] = f

            common_keys = set(pan_dict.keys()) & set(mul_dict.keys())
            for key in sorted(common_keys):
                file_pairs.append((pan_dict[key], mul_dict[key]))

        else:  # real_test
            # 真实测试阶段
            ms_up_dir = os.path.join(self.data_dir, 'MS_up_800')
            pan_cut_dir = os.path.join(self.data_dir, 'PAN_cut_800')

            if os.path.exists(ms_up_dir) and os.path.exists(pan_cut_dir):
                ms_files = sorted(glob.glob(os.path.join(ms_up_dir, '*.tif')))
                pan_files = sorted(glob.glob(os.path.join(pan_cut_dir, '*.tif')))

                min_len = min(len(ms_files), len(pan_files))
                for i in range(min_len):
                    file_pairs.append((pan_files[i], ms_files[i]))

        return file_pairs

    def safe_normalize(self, data):
        """安全的归一化函数"""
        if len(data.shape) == 2:
            min_val = data.min()
            max_val = data.max()
            if max_val - min_val < 1e-6:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val + 1e-8)
        else:
            normalized = []
            for i in range(data.shape[0]):
                band = data[i]
                min_val = band.min()
                max_val = band.max()
                if max_val - min_val < 1e-6:
                    normalized.append(np.zeros_like(band))
                else:
                    normalized.append((band - min_val) / (max_val - min_val + 1e-8))
            return np.stack(normalized, axis=0)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pan_path, ms_path = self.file_pairs[idx]

        try:
            # 读取数据
            with rasterio.open(pan_path) as src:
                pan = src.read().astype(np.float32)
            with rasterio.open(ms_path) as src:
                ms = src.read().astype(np.float32)

            # 安全归一化
            pan = self.safe_normalize(pan)
            ms = self.safe_normalize(ms)

            # 调整维度
            if len(pan.shape) == 2:
                pan = pan[np.newaxis, :]  # (1, H, W)
            if len(ms.shape) == 2:
                ms = ms[np.newaxis, :]  # (C, H, W)

            # 转换为张量
            pan_tensor = torch.FloatTensor(pan)
            ms_tensor = torch.FloatTensor(ms)

            return {
                'pan': pan_tensor,
                'ms': ms_tensor,
                'pan_path': pan_path,
                'ms_path': ms_path
            }

        except Exception as e:
            print(f"读取样本 {idx} 失败: {e}")
            # 返回安全的默认数据
            return {
                'pan': torch.zeros((1, 256, 256)),
                'ms': torch.zeros((8, 64, 64)),
                'pan_path': 'error',
                'ms_path': 'error'
            }


# ===================== 5. 稳定的训练函数 =====================
def stable_train(model, train_loader, val_loader, args):
    """稳定的训练函数"""

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用稳定的损失函数
    criterion = StableSelfSupervisedLoss(
        recon_weight=args.recon_weight,
        spatial_weight=args.spatial_weight,
        spectral_weight=args.spectral_weight
    )

    # 优化器（添加梯度裁剪）
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练历史
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_recon': [], 'val_recon': [],
        'train_spatial': [], 'val_spatial': [],
        'train_spectral': [], 'val_spectral': []
    }

    best_val_loss = float('inf')
    best_model_path = output_dir / 'best_model.pth'

    print("开始稳定训练...")

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_metrics = {'total': 0.0, 'recon': 0.0, 'spatial': 0.0, 'spectral': 0.0}
        train_samples = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch in pbar:
            pan = batch['pan'].to(args.device)
            ms = batch['ms'].to(args.device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            fused = model(ms, pan)

            # 计算损失
            total_loss, losses = criterion(fused, pan, ms)

            # 检查损失是否合理
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"跳过批次: 损失值异常")
                continue

            # 反向传播（添加梯度裁剪）
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录指标
            batch_size = pan.size(0)
            train_metrics['total'] += total_loss.item() * batch_size
            train_metrics['recon'] += losses['recon'].item() * batch_size
            train_metrics['spatial'] += losses['spatial'].item() * batch_size
            train_metrics['spectral'] += losses['spectral'].item() * batch_size
            train_samples += batch_size

            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Recon': f'{losses["recon"].item():.4f}',
                'Spatial': f'{losses["spatial"].item():.4f}'
            })

        # 验证阶段
        model.eval()
        val_metrics = {'total': 0.0, 'recon': 0.0, 'spatial': 0.0, 'spectral': 0.0}
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                pan = batch['pan'].to(args.device)
                ms = batch['ms'].to(args.device)

                fused = model(ms, pan)
                total_loss, losses = criterion(fused, pan, ms)

                batch_size = pan.size(0)
                val_metrics['total'] += total_loss.item() * batch_size
                val_metrics['recon'] += losses['recon'].item() * batch_size
                val_metrics['spatial'] += losses['spatial'].item() * batch_size
                val_metrics['spectral'] += losses['spectral'].item() * batch_size
                val_samples += batch_size

        # 计算平均损失
        train_loss = train_metrics['total'] / max(train_samples, 1)
        val_loss = val_metrics['total'] / max(val_samples, 1)

        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_metrics['recon'] / max(train_samples, 1))
        history['val_recon'].append(val_metrics['recon'] / max(val_samples, 1))
        history['train_spatial'].append(train_metrics['spatial'] / max(train_samples, 1))
        history['val_spatial'].append(val_metrics['spatial'] / max(val_samples, 1))
        history['train_spectral'].append(train_metrics['spectral'] / max(train_samples, 1))
        history['val_spectral'].append(val_metrics['spectral'] / max(val_samples, 1))

        print(f'Epoch {epoch + 1}/{args.epochs}: '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss and not (
                torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss))):
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 保存最佳模型: {best_model_path}")

    # 保存训练历史
    save_training_history(history, output_dir / 'training_history.csv')
    plot_training_curves(history, output_dir / 'training_curves.png')

    return model, history


def save_training_history(history, save_path):
    """保存训练历史"""
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss',
                         'train_recon', 'val_recon',
                         'train_spatial', 'val_spatial',
                         'train_spectral', 'val_spectral'])

        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_recon'][i],
                history['val_recon'][i],
                history['train_spatial'][i],
                history['val_spatial'][i],
                history['train_spectral'][i],
                history['val_spectral'][i]
            ])


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 总损失
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 重建损失
    axes[0, 1].plot(history['epoch'], history['train_recon'], label='Train')
    axes[0, 1].plot(history['epoch'], history['val_recon'], label='Validation')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 空间损失
    axes[1, 0].plot(history['epoch'], history['train_spatial'], label='Train')
    axes[1, 0].plot(history['epoch'], history['val_spatial'], label='Validation')
    axes[1, 0].set_title('Spatial Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 光谱损失
    axes[1, 1].plot(history['epoch'], history['train_spectral'], label='Train')
    axes[1, 1].plot(history['epoch'], history['val_spectral'], label='Validation')
    axes[1, 1].set_title('Spectral Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


# ===================== 6. 测试函数 =====================
def stable_test(model, test_loader, args, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='测试融合')):
            try:
                pan = batch['pan'].to(args.device)
                ms = batch['ms'].to(args.device)

                # 生成融合结果
                fused = model(ms, pan)

                # 重要：检查模型输出形状
                print(f"模型输出形状: {fused.shape}")  # 应该是[1, 8, H, W]

                # 转换为numpy - 正确处理维度
                fused_np = fused.squeeze(0).cpu().numpy()  # 移除batch维度 -> [8, H, W]
                print(f"保存前形状: {fused_np.shape}")  # 应该是(8, 800, 800)

                # 确保是8个波段
                if fused_np.shape[0] != 8:
                    print(f"⚠️ 警告: 融合结果只有{fused_np.shape[0]}个波段，应该是8个!")
                    # 可能需要检查模型输出

                # 数值范围检查
                fused_np = np.clip(fused_np, 0.0, 1.0)

                # 转换为16位整数
                fused_uint16 = (fused_np * 65535).astype(np.uint16)

                # 保存8通道TIFF
                output_path = os.path.join(save_dir, f'fusion_8ch_{i:04d}.tif')
                tiff.imwrite(output_path, fused_uint16,
                             photometric='minisblack',
                             planarconfig='separate')  # 确保多波段正确保存

            except Exception as e:
                print(f"处理第{i}个样本时出错: {e}")
                continue
    print(f"\n✅ 融合完成! 共处理 {len(test_loader.dataset)} 个样本")
    print(f"📁 结果保存在: {save_dir}")


def visual_quality_check(ms, pan, fused, save_path):
    """视觉质量检查"""
    import matplotlib.pyplot as plt

    # 选择用于显示的波段（假彩色：7,3,2 -> NIR, R, G）
    rgb_bands = [6, 2, 1]  # 0-indexed

    # 创建RGB图像
    def create_rgb(image, bands):
        if image.shape[0] >= max(bands) + 1:
            rgb = image[bands].transpose(1, 2, 0)
            # 对比度拉伸
            for i in range(3):
                p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
                if p98 > p2:
                    rgb[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)
            return rgb
        return None

    # 生成各图像的RGB视图
    ms_rgb = create_rgb(ms, rgb_bands) if ms.shape[0] >= 7 else None
    fused_rgb = create_rgb(fused, rgb_bands) if fused.shape[0] >= 7 else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 显示MS RGB
    if ms_rgb is not None:
        axes[0, 0].imshow(ms_rgb)
        axes[0, 0].set_title('参考MS (RGB)')
        axes[0, 0].axis('off')

    # 显示PAN
    axes[0, 1].imshow(pan[0], cmap='gray')
    axes[0, 1].set_title('PAN图像')
    axes[0, 1].axis('off')

    # 显示融合结果RGB
    if fused_rgb is not None:
        axes[0, 2].imshow(fused_rgb)
        axes[0, 2].set_title('融合结果 (RGB)')
        axes[0, 2].axis('off')

    # 显示差异
    if ms_rgb is not None and fused_rgb is not None:
        diff = np.abs(ms_rgb - fused_rgb).mean(axis=2)
        im = axes[1, 0].imshow(diff, cmap='hot')
        axes[1, 0].set_title('差异图 (MS vs Fused)')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])

    # 显示边缘对比
    from scipy import ndimage
    if fused_rgb is not None:
        edges = ndimage.sobel(fused_rgb.mean(axis=2))
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('融合结果边缘')
        axes[1, 1].axis('off')

    # 显示PAN边缘
    pan_edges = ndimage.sobel(pan[0])
    axes[1, 2].imshow(pan_edges, cmap='gray')
    axes[1, 2].set_title('PAN边缘')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
# ===================== 7. 主函数 =====================
def main():
    args = parse_args()

    # 设置输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = f'./results/stable_pansharpen_{timestamp}'

    print("=" * 60)
    print("稳定版自监督Pansharpening训练")
    print(f"设备: {args.device}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 显示配置
    print("\n训练配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 创建模型
    model = SimplePansharpen(ms_ch=8, pan_ch=1, feat_ch=args.feat_ch).to(args.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 检查数据目录
    if not os.path.exists(args.data_root):
        print(f"错误: 数据目录不存在: {args.data_root}")
        return

    # 加载数据
    print("\n加载数据...")
    try:
        train_dataset = StablePansharpenDataset(args.data_root, 'train')
        val_dataset = StablePansharpenDataset(args.data_root, 'test')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        print(f"训练集: {len(train_dataset)} 个样本")
        print(f"验证集: {len(val_dataset)} 个样本")

    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    # 开始训练
    print("\n开始稳定训练...")
    model, history = stable_train(model, train_loader, val_loader, args)

    print("\n训练完成!")

    # 测试真实数据
    if os.path.exists(os.path.join(args.data_root, 'real_data')):
        print("\n测试真实数据...")

        # 加载最佳模型
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=args.device))
            print(f"加载最佳模型: {best_model_path}")

        # 测试
        test_dataset = StablePansharpenDataset(args.data_root, 'real_test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        test_save_dir = os.path.join(args.output_dir, 'fusion_results')
        stable_test(model, test_loader, args, test_save_dir)

    print(f"\n🎉 所有流程完成!")
    print(f"📁 完整结果保存在: {args.output_dir}")

    return model, history


if __name__ == '__main__':
    main()