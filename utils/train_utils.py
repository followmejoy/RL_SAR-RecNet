# -*- coding: utf-8 -*-
"""
通用训练工具模块（包含完整评估指标）
"""

import os
import time
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import scipy.io
from skimage.metrics import structural_similarity as ssim




# ------------------ Seed ------------------ #
def set_random_seed(seed: int = 42, deterministic: bool = True):
    """设置随机种子以确保可重复性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------ IO: load mask & thetas ------------------ #
def load_mask(sample_rate: int, device: str) -> torch.Tensor:
    """
    加载 mask_{sample_rate}.mat 文件

    Args:
        sample_rate: 采样率
        device: 设备

    Returns:
        掩码张量 [1,1,H,W] (0/1)
    """
    path = f"mask_{sample_rate}.mat"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}. Put it in current working directory.")
    mat = scipy.io.loadmat(path)
    if "mask" not in mat:
        raise KeyError(f"{path} must contain variable 'mask'. Keys: {list(mat.keys())}")

    m = torch.tensor(mat["mask"])
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 3:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected mask dims: {m.shape}")

    return m.to(device).float()


def load_thetas(device: str) -> List[torch.Tensor]:
    """
    加载 thetas.mat 文件

    Args:
        device: 设备

    Returns:
        [Theta1, Theta2, Theta3] 复数张量列表
    """
    path = "thetas.mat"
    if not os.path.exists(path):
        raise FileNotFoundError("Cannot find thetas.mat in current working directory.")
    mat = scipy.io.loadmat(path)

    for k in ["Theta1", "Theta2", "Theta3"]:
        if k not in mat:
            raise KeyError(f"thetas.mat must contain {k}. Keys: {list(mat.keys())}")

    theta1 = torch.tensor(mat["Theta1"], dtype=torch.complex64, device=device)
    theta2 = torch.tensor(mat["Theta2"], dtype=torch.complex64, device=device)
    theta3 = torch.tensor(mat["Theta3"], dtype=torch.complex64, device=device)
    return [theta1, theta2, theta3]


#------------------ 完整评估指标（根据论文公式） ------------------ #
@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算PSNR（公式23）
    PSNR = (1/T) Σ -10 * log10(||X - X̂||_F^2 / N_X)
    其中N_X是图像总像素数
    """
    batch_size = pred.shape[0]
    total_psnr = 0.0

    for i in range(batch_size):
        # 提取单个图像
        pred_i = pred[i]
        target_i = target[i]

        # 计算Frobenius范数的平方（||X - X̂||_F^2）
        diff = pred_i - target_i
        mse = torch.sum(diff ** 2)  # Frobenius范数的平方

        # 图像总像素数 N_X
        N_X = torch.numel(target_i)

        # 避免log10(0)
        if mse <= 1e-10:
            psnr_i = float('inf')
        else:
            psnr_i = -10 * torch.log10(mse / N_X)

        total_psnr += psnr_i.item()

    # 平均PSNR
    return total_psnr / batch_size
# @torch.no_grad()
# def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
#     batch_size = pred.shape[0]
#     total_psnr = 0.0
#
#     for i in range(batch_size):
#         pred_i = pred[i]
#         target_i = target[i]
#
#         diff = pred_i - target_i
#         mse = torch.sum(diff ** 2)
#         N_X = torch.numel(target_i)
#
#         print(f"DEBUG: batch {i}: mse={mse.item():.6e}, N_X={N_X}")  # 添加调试输出
#
#         if mse <= 1e-10:
#             psnr_i = float('inf')
#         else:
#             psnr_i = -10 * torch.log10(mse / N_X)
#
#         total_psnr += psnr_i.item()
#
#     return total_psnr / batch_size


@torch.no_grad()
def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算NMSE（常用公式）
    NMSE = ||X - X̂||_F^2 / ||X||_F^2
    """
    batch_size = pred.shape[0]
    total_nmse = 0.0

    for i in range(batch_size):
        # 提取单个图像
        pred_i = pred[i]
        target_i = target[i]

        # 计算分子：||X - X̂||_F^2
        diff = pred_i - target_i
        numerator = torch.sum(diff ** 2)

        # 计算分母：||X||_F^2
        denominator = torch.sum(target_i ** 2)

        # 避免除零
        if denominator <= 1e-10:
            nmse_i = float('inf')
        else:
            nmse_i = numerator / denominator

        total_nmse += nmse_i.item()

    # 平均NMSE
    return total_nmse / batch_size


@torch.no_grad()
def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算SSIM（公式24）
    使用skimage的ssim函数，C1=(0.01)^2, C2=(0.03)^2
    """
    batch_size = pred.shape[0]
    total_ssim = 0.0

    for i in range(batch_size):
        # 提取单个图像并转换为numpy
        pred_np = pred[i].squeeze().cpu().numpy()
        target_np = target[i].squeeze().cpu().numpy()

        # 确保数据在[0, 1]范围内
        pred_np = np.clip(pred_np, 0, 1)
        target_np = np.clip(target_np, 0, 1)

        # 数据范围
        data_range = 1.0

        # 使用skimage的ssim，它实现了公式24
        # 注意：论文公式是SSIM(X, X̂)，但skimage参数顺序是(gt, pred)
        ssim_i = ssim(target_np, pred_np,
                      data_range=data_range,
                      win_size=3,  # 根据论文，这是局部窗口大小
                      channel_axis=None)  # 灰度图像

        total_ssim += ssim_i

    # 平均SSIM
    return total_ssim / batch_size


@torch.no_grad()
def compute_entropy(pred: torch.Tensor) -> float:
    """
    计算熵（公式25）
    ENT = (1/T) Σ (ln E_t - (1/E_t) Σ Σ P_t)
    其中 E_t = ||X̂(t)||_F^2 是第t帧的总能量
    P_t = |X̂(t,m,n)|² ln|X̂(t,m,n)|²
    """
    batch_size = pred.shape[0]
    total_entropy = 0.0

    for i in range(batch_size):
        # 提取单个图像
        X_hat = pred[i]

        # 确保非负（SAR图像通常是幅度图像）
        X_hat = torch.abs(X_hat)

        # 计算总能量 E_t = ||X̂||_F^2
        E_t = torch.sum(X_hat ** 2)

        # 避免log(0)，添加小值
        X_hat_sq = X_hat ** 2 + 1e-10

        # 计算 P = X̂² ln(X̂²)
        # 注意：当X̂²=0时，使用极限值0 * ln(0) = 0
        mask = X_hat_sq > 1e-10
        P = torch.zeros_like(X_hat_sq)
        P[mask] = X_hat_sq[mask] * torch.log(X_hat_sq[mask])

        # 计算熵项
        if E_t > 1e-10:
            # ln E_t
            ln_E_t = torch.log(E_t)

            # (1/E_t) Σ Σ P
            sum_P = torch.sum(P)
            term2 = sum_P / E_t

            # 总熵 = ln E_t - (1/E_t) Σ Σ P
            entropy_i = ln_E_t - term2
        else:
            entropy_i = 0.0

        # 如果entropy_i是float类型，直接累加
        if isinstance(entropy_i, float):  # 检查是否是float类型
            total_entropy += entropy_i
        else:
            total_entropy += entropy_i.item()  # 否则使用item()

    # 平均熵
    return total_entropy / batch_size


@torch.no_grad()
def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]

    Returns:
        包含所有指标的字典
    """
    return {
        "psnr": compute_psnr(pred, target),
        "nmse": compute_nmse(pred, target),
        "ssim": compute_ssim(pred, target),
        "entropy": compute_entropy(pred)
    }




# ------------------ Data Loading ------------------ #
def create_data_loaders(
        train_path: str,
        val_path: str,
        batch_size: int = 8,
        img_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False  # 添加这个参数
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        batch_size: 批次大小
        img_size: 图像尺寸
        num_workers: 数据加载工作线程数

    Returns:
        (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(train_path, transform=transform)
    val_set = datasets.ImageFolder(val_path, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    return train_loader, val_loader


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, save_dir: str, log_filename: str = "train_log.csv"):
        """
        初始化训练日志记录器

        Args:
            save_dir: 保存目录
            log_filename: 日志文件名
        """
        self.save_dir = save_dir
        self.log_path = os.path.join(save_dir, log_filename)

        # 创建目录
        os.makedirs(save_dir, exist_ok=True)

    def write_header(self, header: str):
        """写入CSV文件头"""
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")

    def log_epoch(self, data: Dict[str, Any]):
        """记录一个epoch的数据"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            line = ",".join([str(data.get(col, "")) for col in data])
            f.write(line + "\n")


# ------------------ Model Checkpoint Manager ------------------ #
class CheckpointManager:
    """模型检查点管理器"""

    def __init__(self, save_dir: str, model_name: str):
        """
        初始化检查点管理器

        Args:
            save_dir: 保存目录
            model_name: 模型名称
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.ckpt_dir = os.path.join(save_dir, "checkpoints")
        self.fig_dir = os.path.join(save_dir, "figures")

        # 创建目录
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def save_checkpoint(
            self,
            epoch: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            val_loss: float,
            metrics: Dict[str, float] = None,
            args: Any = None,
            filename: str = None
    ) -> str:
        """
        保存模型检查点

        Args:
            epoch: 当前epoch
            model: 模型
            optimizer: 优化器
            val_loss: 验证损失
            metrics: 评估指标字典
            args: 训练参数
            filename: 文件名（如为None则自动生成）

        Returns:
            保存的路径
        """
        if filename is None:
            filename = f"epoch_{epoch:03d}.pth"

        save_path = os.path.join(self.ckpt_dir, filename)

        # 构建保存字典
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
        }

        # 添加评估指标
        if metrics:
            checkpoint.update(metrics)

        # 添加训练参数
        if args:
            checkpoint["args"] = vars(args) if hasattr(args, '__dict__') else args

        torch.save(checkpoint, save_path)

        return save_path

    def save_best_checkpoint(
            self,
            epoch: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            val_loss: float,
            metrics: Dict[str, float] = None,
            args: Any = None
    ) -> str:
        """保存最佳模型检查点"""
        return self.save_checkpoint(
            epoch, model, optimizer, val_loss, metrics, args, f"best_{self.model_name}.pth"
        )

    def get_figure_path(self, epoch: int) -> str:
        """获取可视化图像保存路径"""
        return os.path.join(self.fig_dir, f"recon_epoch_{epoch:03d}.png")


# ------------------ 通用训练循环组件 ------------------ #
def train_epoch_generic(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        forward_func: callable,  # 自定义前向传播函数
        **forward_kwargs  # 前向传播的额外参数
) -> Tuple[float, Dict[str, float]]:
    """
    通用训练epoch函数

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        forward_func: 前向传播函数，签名为 forward_func(models, batch, **forward_kwargs)
        forward_kwargs: 前向传播的额外参数

    Returns:
        (平均训练损失, 额外训练指标字典)
    """
    model.train()
    train_loss_sum = 0.0
    n_train = 0
    extra_metrics = {}

    for X, _ in train_loader:
        X = X.float().to(device)

        optimizer.zero_grad()

        # 使用自定义前向传播函数
        pred, batch_extra_metrics = forward_func(model, X, **forward_kwargs)

        # 计算损失
        loss = criterion(pred, X)
        loss.backward()
        optimizer.step()

        # 累积统计
        bs = X.shape[0]
        train_loss_sum += loss.item() * bs
        n_train += bs

        # 累积额外指标
        for key, value in batch_extra_metrics.items():
            if key not in extra_metrics:
                extra_metrics[key] = 0.0
            extra_metrics[key] += value * bs

    # 计算平均值
    avg_loss = train_loss_sum / max(n_train, 1)
    for key in extra_metrics:
        extra_metrics[key] = extra_metrics[key] / max(n_train, 1)

    return avg_loss, extra_metrics


def validate_generic(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: str,
        forward_func: callable,  # 自定义前向传播函数
        **forward_kwargs  # 前向传播的额外参数
) -> Tuple[float, Dict[str, float]]:
    """
    通用验证函数

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        forward_func: 前向传播函数，签名为 forward_func(models, batch, **forward_kwargs)
        forward_kwargs: 前向传播的额外参数

    Returns:
        (平均验证损失, 包含所有评估指标的字典)
    """
    model.eval()
    val_loss_sum = 0.0
    n_val = 0

    # 初始化指标累积器
    metrics_accumulator = {}

    with torch.no_grad():
        for X, _ in val_loader:
            X = X.float().to(device)

            # 使用自定义前向传播函数
            pred, batch_extra_metrics = forward_func(model, X, **forward_kwargs)

            # 计算损失
            loss = criterion(pred, X)

            # 计算所有评估指标
            batch_metrics = compute_all_metrics(pred, X)

            # 累积统计
            bs = X.shape[0]
            val_loss_sum += loss.item() * bs
            n_val += bs

            # 累积指标
            for key, value in batch_metrics.items():
                if key not in metrics_accumulator:
                    metrics_accumulator[key] = 0.0
                metrics_accumulator[key] += value * bs

    # 计算平均值
    avg_loss = val_loss_sum / max(n_val, 1)
    for key in metrics_accumulator:
        metrics_accumulator[key] = metrics_accumulator[key] / max(n_val, 1)

    return avg_loss, metrics_accumulator


# ------------------ 打印进度函数 ------------------ #
def print_epoch_progress(
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
        time_elapsed: float
) -> None:
    """
    打印epoch进度信息

    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        train_loss: 训练损失
        val_loss: 验证损失
        metrics: 评估指标字典
        time_elapsed: 耗时（秒）
    """
    # 格式化NMSE显示
    nmse_val = metrics.get("nmse", 0.0)
    if nmse_val < 1e-6:
        nmse_str = f"{nmse_val:.2e}"
    else:
        nmse_str = f"{nmse_val:.6f}"

    print(f"[Epoch {epoch:03d}/{total_epochs}] "
          f"train_loss={train_loss:.6e} | val_loss={val_loss:.6e} | "
          f"PSNR={metrics.get('psnr', 0.0):.2f} dB | "
          f"SSIM={metrics.get('ssim', 0.0):.3f} | "
          f"NMSE={nmse_str} | "
          f"Entropy={metrics.get('entropy', 0.0):.3f} | "
          f"{time_elapsed:.1f}s")