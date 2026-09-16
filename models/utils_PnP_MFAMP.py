import torch
import torch.nn as nn
import torch.nn.functional as F


class SARPhysicalFeatureExtractor(nn.Module):
    """
    精简版 SAR 物理特征提取器。

    只保留 3 个最基础、互补性较强的特征：
    1) amplitude      : 幅度，保留目标强度信息
    2) phase_grad     : 相位梯度强度，保留相位突变/边缘线索
    3) hf_structure   : 高频结构响应，保留轮廓和细节结构

    删除：
    - power          : 与 amplitude 高度冗余
    - coherence      : 物理先验过强，容易替代 RL 的调节作用
    - speckle_index  : 容易让网络过度学习散斑统计，削弱 RL 改善空间
    """

    def __init__(self, cond_channels=16, downsample=2, eps=1e-8):
        super().__init__()
        self.cond_channels = cond_channels
        self.eps = eps
        self.num_features = 3
        self.down = nn.AvgPool2d(downsample) if downsample > 1 else nn.Identity()

        self.feature_names = ["amplitude", "phase_grad", "hf_structure"]
        self.feature_weights = nn.Parameter(torch.ones(self.num_features))

        # 特征投影层
        self.cond_proj = nn.Sequential(
            nn.Conv2d(self.num_features, cond_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Laplacian 核（高频结构提取）
        lap_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap_kernel)

    def _normalize_feature(self, x):
        """特征归一化到 [0,1]"""
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + self.eps)

    def forward(self, x_complex, override_weights=None):
        # 确保输入是复数形式
        if not torch.is_complex(x_complex):
            x_complex = torch.complex(x_complex, torch.zeros_like(x_complex))

        # 1. 幅度特征
        amp = torch.abs(x_complex)

        # 2. 相位梯度特征
        phase = torch.angle(x_complex + self.eps)
        grad_x = phase[:, :, :, 1:] - phase[:, :, :, :-1]
        grad_y = phase[:, :, 1:, :] - phase[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        phase_grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

        # 3. 高频结构特征
        hf_structure = F.conv2d(amp, self.lap_kernel.to(dtype=amp.dtype), padding=1).abs()

        # 特征拼接与归一化
        raw_feats = torch.cat(
            [
                self._normalize_feature(amp),
                self._normalize_feature(phase_grad),
                self._normalize_feature(hf_structure),
            ],
            dim=1,
        )

        # 特征权重（支持外部覆盖 / RL action 覆盖）
        if override_weights is None:
            alpha = torch.softmax(self.feature_weights, dim=0)
        else:
            alpha = torch.as_tensor(override_weights, dtype=raw_feats.dtype, device=raw_feats.device)
            if alpha.numel() != self.num_features:
                raise ValueError(f"override_weights must have {self.num_features} values, got {alpha.numel()}")
            alpha = torch.softmax(alpha, dim=0)

        weighted_feats = self.down(raw_feats * alpha.view(1, -1, 1, 1))
        cond = self.cond_proj(weighted_feats)
        return raw_feats, weighted_feats, cond, alpha


class WeakFiLMDnCNN(nn.Module):
    """
    弱 FiLM 去噪器。

    原来是：out = out * gamma + beta
    现在是：out = out * (1 + 0.1*tanh(gamma)) + 0.1*tanh(beta)

    这样保留物理条件调制，但不会在不可见域强行改变特征分布。
    """

    def __init__(self, in_channels=1, cond_channels=16, num_layers=7, num_features=32, film_strength=0.1):
        super().__init__()
        self.film_strength = film_strength
        self.conv1 = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # 中间卷积层
        layers = []
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        self.mid = nn.Sequential(*layers)

        # 输出层与FiLM调制层
        self.conv_out = nn.Conv2d(num_features, in_channels, 3, padding=1, bias=False)
        self.gamma_conv = nn.Conv2d(cond_channels, num_features, 1)
        self.beta_conv = nn.Conv2d(cond_channels, num_features, 1)

    def forward(self, x, cond):
        # 对齐条件特征尺寸
        if x.shape[-2:] != cond.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # FiLM 弱调制
        out = self.conv1(x)
        gamma = self.film_strength * torch.tanh(self.gamma_conv(cond))
        beta = self.film_strength * torch.tanh(self.beta_conv(cond))
        out = out * (1.0 + gamma) + beta
        out = self.relu(out)

        # 中间层与残差输出
        out = self.mid(out)
        residual = self.conv_out(out)
        return x - residual


class PhysicalPriorHead(nn.Module):
    """将物理特征映射为弱成像先验修正项。"""

    def __init__(self, in_channels, hidden_channels=16, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)