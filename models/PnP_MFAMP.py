import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.SARop import CSA_echo, CSA_imag
from models.utils_PnP_MFAMP import (
    SARPhysicalFeatureExtractor,
    WeakFiLMDnCNN,
    PhysicalPriorHead
)


class PnP_MFAMP_Feat(nn.Module):
    """
    PnP-MFAMP-Feat。

    设计目标：
    1. 保留可学习参数，后续可接强化学习做参数控制。
    2. 让数据一致性 DC 成为主导项，避免不可见域被物理先验拉崩。
    3. 物理分支和去噪分支只做弱残差修正。
    4. prior 在低采样率更强，高采样率自动减弱。
    """

    def __init__(
        self,
        iter_num,
        in_channels=1,
        out_channels=1,
        cond_channels=16,
        feature_downsample=2,
        max_lambda_dn=0.20,
        max_lambda_pr=0.05,
        max_onsager=0.20,
        use_onsager=False,
        use_prior=True,
        use_beta_scale=True,
        film_strength=0.10,
    ):
        super().__init__()
        self.r = iter_num
        self.max_lambda_dn = max_lambda_dn
        self.max_lambda_pr = max_lambda_pr
        self.max_onsager = max_onsager
        self.use_onsager = use_onsager
        self.use_prior = use_prior
        self.use_beta_scale = use_beta_scale

        # 导入拆分后的组件
        self.feature_extractor = SARPhysicalFeatureExtractor(
            cond_channels=cond_channels,
            downsample=feature_downsample,
        )
        self.denoiser = WeakFiLMDnCNN(
            in_channels=in_channels,
            cond_channels=cond_channels,
            num_layers=7,
            num_features=32,
            film_strength=film_strength,
        )
        self.prior_head = PhysicalPriorHead(
            in_channels=self.feature_extractor.num_features,
            hidden_channels=16,
            out_channels=out_channels,
        )

        # 可学习参数（后续可由RL覆盖）
        self.lambda_dn_logits = nn.Parameter(torch.ones(iter_num) * -2.0)
        self.lambda_pr_logits = nn.Parameter(torch.ones(iter_num) * -3.0)
        self.onsager_logits = nn.Parameter(torch.ones(iter_num) * -3.0)
        self.beta_logits = nn.Parameter(torch.zeros(iter_num))

    def _check_thetas(self, thetas):
        if not (isinstance(thetas, list) and len(thetas) == 3):
            raise ValueError(f"Expected thetas list of 3, got {len(thetas) if hasattr(thetas, '__len__') else 'unknown'}")

    def forward_G(self, x, thetas, mask):
        y_pred = CSA_echo(x, thetas)
        if mask is not None:
            y_pred = y_pred * mask
        return y_pred

    def backward_M(self, r, thetas, mask):
        if mask is not None:
            r = r * mask
        return CSA_imag(r, thetas)

    def estimate_onsager(self, x):
        # 保留接口，但默认 use_onsager=False
        with torch.no_grad():
            mag = x.abs() if torch.is_complex(x) else x
            b = (mag > 1e-6).float().mean()
        return b

    def safe_complex_nan_to_num(self, z, nan=0.0, posinf=None, neginf=None):
        if torch.is_complex(z):
            real = torch.nan_to_num(z.real, nan=nan, posinf=posinf, neginf=neginf)
            imag = torch.nan_to_num(z.imag, nan=nan, posinf=posinf, neginf=neginf)
            return torch.complex(real, imag)
        return torch.nan_to_num(z, nan=nan, posinf=posinf, neginf=neginf)

    def _normalize_image(self, x):
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def _sampling_rate(self, mask, y):
        if mask is None:
            return torch.tensor(1.0, dtype=y.real.dtype if torch.is_complex(y) else y.dtype, device=y.device)
        m = mask.float() if not torch.is_complex(mask) else mask.abs().float()
        return m.mean().clamp(0.0, 1.0)

    def get_feature_weights(self):
        return torch.softmax(self.feature_extractor.feature_weights.detach(), dim=0)

    def get_control_params(self):
        with torch.no_grad():
            lambda_dn = self.max_lambda_dn * torch.sigmoid(self.lambda_dn_logits)
            lambda_pr = self.max_lambda_pr * torch.sigmoid(self.lambda_pr_logits)
            onsager = self.max_onsager * torch.sigmoid(self.onsager_logits)
            beta_scale = 0.9 + 0.2 * torch.sigmoid(self.beta_logits)
        return {
            "lambda_dn": lambda_dn.detach().cpu(),
            "lambda_pr": lambda_pr.detach().cpu(),
            "onsager": onsager.detach().cpu(),
            "beta_scale": beta_scale.detach().cpu(),
        }

    def _get_action_value(self, action, key, i, default_value, dtype, device):
        """
        action 可选，用于后续接 RL。
        支持形式：
        action = {
            "lambda_dn": Tensor/list/scalar,
            "lambda_pr": Tensor/list/scalar,
            "onsager": Tensor/list/scalar,
            "beta_scale": Tensor/list/scalar,
            "feature_weights": Tensor/list length 3
        }
        """
        if action is None or key not in action or action[key] is None:
            return default_value

        v = torch.as_tensor(action[key], dtype=dtype, device=device)
        if v.numel() == 1:
            return v.reshape(())
        return v.flatten()[i]

    def forward(
        self,
        y,
        thetas,
        mask=None,
        feature_weights_override=None,
        action=None,
        return_info=False,
    ):
        self._check_thetas(thetas)

        x0_complex = CSA_imag(y, thetas)
        x0_complex = self.safe_complex_nan_to_num(x0_complex, nan=0.0, posinf=1.0, neginf=0.0)
        x = x0_complex.abs() if torch.is_complex(x0_complex) else x0_complex
        x = self.safe_complex_nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = self._normalize_image(x)

        dtype = x.dtype
        device = x.device
        sampling_rate = self._sampling_rate(mask, y).to(dtype=dtype, device=device)
        prior_gate = (1.0 - sampling_rate).clamp(0.0, 1.0)

        r_prev = torch.zeros_like(y)
        info = {
            "feature_weights": [],
            "lambda_dn": [],
            "lambda_pr": [],
            "onsager_strength": [],
            "beta_scale": [],
            "sampling_rate": sampling_rate.detach().cpu(),
            "raw_feats": [],
        }

        for i in range(self.r):
            y_pred = self.forward_G(x, thetas, mask)

            # Onsager 项（默认关闭）
            base_onsager = self.max_onsager * torch.sigmoid(self.onsager_logits[i])
            onsager_strength = self._get_action_value(action, "onsager", i, base_onsager, dtype, device)
            onsager_strength = onsager_strength.clamp(0.0, self.max_onsager)

            if self.use_onsager and i > 0:
                b_k = self.estimate_onsager(x).to(dtype=dtype, device=device)
                r = y - y_pred + onsager_strength * b_k * r_prev
            else:
                r = y - y_pred

            r = self.safe_complex_nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0)
            backproj = self.backward_M(r, thetas, mask)
            backproj = self.safe_complex_nan_to_num(backproj, nan=0.0, posinf=1.0, neginf=0.0)

            # 构造复数值代理
            if torch.is_complex(backproj):
                x_complex_proxy = torch.complex(x, torch.zeros_like(x)) + backproj
            else:
                x_complex_proxy = torch.complex(x + backproj, torch.zeros_like(x))
            x_complex_proxy = self.safe_complex_nan_to_num(x_complex_proxy, nan=0.0, posinf=1.0, neginf=0.0)

            # DC anchor：后续所有分支都只围绕它做弱残差
            x_base = x_complex_proxy.abs()
            x_base = self.safe_complex_nan_to_num(x_base, nan=0.0, posinf=1.0, neginf=0.0)
            x_base = self._normalize_image(x_base)

            # 特征提取（支持外部权重覆盖）
            fw_override = feature_weights_override
            if action is not None and "feature_weights" in action and action["feature_weights"] is not None:
                fw_override = action["feature_weights"]

            raw_feats, weighted_feats, cond, alpha_feat = self.feature_extractor(
                x_complex_proxy,
                override_weights=fw_override,
            )

            # 去噪分支
            x_denoised = self.denoiser(x_base, cond)
            x_denoised = self.safe_complex_nan_to_num(x_denoised, nan=0.0, posinf=1.0, neginf=0.0)
            x_denoised = self._normalize_image(x_denoised)

            # 物理先验分支
            if self.use_prior:
                x_prior = self.prior_head(weighted_feats)
                if x_prior.shape[-2:] != x_base.shape[-2:]:
                    x_prior = F.interpolate(x_prior, size=x_base.shape[-2:], mode="bilinear", align_corners=False)
                x_prior = self._normalize_image(x_prior)
                x_prior = self.safe_complex_nan_to_num(x_prior, nan=0.0, posinf=1.0, neginf=0.0)
            else:
                x_prior = x_base

            # 获取控制参数
            base_lambda_dn = self.max_lambda_dn * torch.sigmoid(self.lambda_dn_logits[i])
            base_lambda_pr = self.max_lambda_pr * torch.sigmoid(self.lambda_pr_logits[i])
            base_beta_scale = 0.9 + 0.2 * torch.sigmoid(self.beta_logits[i])

            lambda_dn = self._get_action_value(action, "lambda_dn", i, base_lambda_dn, dtype, device)
            lambda_pr = self._get_action_value(action, "lambda_pr", i, base_lambda_pr, dtype, device)
            beta_scale = self._get_action_value(action, "beta_scale", i, base_beta_scale, dtype, device)

            lambda_dn = lambda_dn.clamp(0.0, self.max_lambda_dn)
            lambda_pr = lambda_pr.clamp(0.0, self.max_lambda_pr) * prior_gate
            beta_scale = beta_scale.clamp(0.9, 1.1)

            # DC主导 + 残差修正
            x = x_base + lambda_dn * (x_denoised - x_base) + lambda_pr * (x_prior - x_base)

            if self.use_beta_scale:
                x = beta_scale * x

            x = self.safe_complex_nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x = self._normalize_image(x)
            r_prev = r

            # 记录日志信息
            if return_info:
                info["feature_weights"].append(alpha_feat.detach().cpu())
                info["lambda_dn"].append(lambda_dn.detach().cpu())
                info["lambda_pr"].append(lambda_pr.detach().cpu())
                info["onsager_strength"].append(onsager_strength.detach().cpu())
                info["beta_scale"].append(beta_scale.detach().cpu())
                info["raw_feats"].append(raw_feats.detach().cpu())

        if return_info:
            return x, info
        return x

    # ===================== 新增：强化学习所需的单步迭代方法 =====================
    def initialize(self, y, thetas, mask):
        """
        初始化重建图像（对应原forward方法的前几步）
        返回: 初始重建图像x
        """
        self._check_thetas(thetas)
        x0_complex = CSA_imag(y, thetas)
        x0_complex = self.safe_complex_nan_to_num(x0_complex, nan=0.0, posinf=1.0, neginf=0.0)
        x = x0_complex.abs() if torch.is_complex(x0_complex) else x0_complex
        x = self.safe_complex_nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = self._normalize_image(x)
        return x

    def forward_step(self, x, y, thetas, mask, t, lambda_dn, lambda_pr, r_prev=None):
        """
        执行第t次单步迭代（对应原forward方法的一次循环体）

        参数:
            x: 当前重建图像 [B, C, H, W]
            y: 观测数据
            thetas: 成像参数
            mask: 采样mask
            t: 当前迭代次数（0-based）
            lambda_dn: 当前步的去噪权重（float或tensor）
            lambda_pr: 当前步的先验权重（float或tensor）
            r_prev: 上一步的残差（用于Onsager项）

        返回:
            x_next: 下一步的重建图像
            r: 当前步的残差
        """
        self._check_thetas(thetas)
        dtype = x.dtype
        device = x.device

        # 计算采样率和先验门控
        sampling_rate = self._sampling_rate(mask, y).to(dtype=dtype, device=device)
        prior_gate = (1.0 - sampling_rate).clamp(0.0, 1.0)

        # 1. 前向投影
        y_pred = self.forward_G(x, thetas, mask)

        # 2. Onsager项（默认关闭，使用内部参数）
        base_onsager = self.max_onsager * torch.sigmoid(self.onsager_logits[t])
        onsager_strength = base_onsager.clamp(0.0, self.max_onsager)

        if self.use_onsager and t > 0 and r_prev is not None:
            b_k = self.estimate_onsager(x).to(dtype=dtype, device=device)
            r = y - y_pred + onsager_strength * b_k * r_prev
        else:
            r = y - y_pred

        r = self.safe_complex_nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0)

        # 3. 反向投影
        backproj = self.backward_M(r, thetas, mask)
        backproj = self.safe_complex_nan_to_num(backproj, nan=0.0, posinf=1.0, neginf=0.0)

        # 4. 构造复数值代理
        if torch.is_complex(backproj):
            x_complex_proxy = torch.complex(x, torch.zeros_like(x)) + backproj
        else:
            x_complex_proxy = torch.complex(x + backproj, torch.zeros_like(x))
        x_complex_proxy = self.safe_complex_nan_to_num(x_complex_proxy, nan=0.0, posinf=1.0, neginf=0.0)

        # 5. DC anchor（主导项）
        x_base = x_complex_proxy.abs()
        x_base = self.safe_complex_nan_to_num(x_base, nan=0.0, posinf=1.0, neginf=0.0)
        x_base = self._normalize_image(x_base)

        # 6. 特征提取
        raw_feats, weighted_feats, cond, alpha_feat = self.feature_extractor(
            x_complex_proxy,
            override_weights=None,
        )

        # 7. 去噪分支
        x_denoised = self.denoiser(x_base, cond)
        x_denoised = self.safe_complex_nan_to_num(x_denoised, nan=0.0, posinf=1.0, neginf=0.0)
        x_denoised = self._normalize_image(x_denoised)

        # 8. 物理先验分支
        if self.use_prior:
            x_prior = self.prior_head(weighted_feats)
            if x_prior.shape[-2:] != x_base.shape[-2:]:
                x_prior = F.interpolate(x_prior, size=x_base.shape[-2:], mode="bilinear", align_corners=False)
            x_prior = self._normalize_image(x_prior)
            x_prior = self.safe_complex_nan_to_num(x_prior, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            x_prior = x_base

        # 9. 获取beta_scale（使用内部参数）
        base_beta_scale = 0.9 + 0.2 * torch.sigmoid(self.beta_logits[t])
        beta_scale = base_beta_scale.clamp(0.9, 1.1)

        # 10. 应用传入的lambda参数（修复：处理float输入）
        lambda_dn = torch.tensor(lambda_dn, dtype=dtype, device=device).clamp(0.0, self.max_lambda_dn)
        lambda_pr = torch.tensor(lambda_pr, dtype=dtype, device=device).clamp(0.0, self.max_lambda_pr) * prior_gate

        # 11. DC主导 + 残差修正
        x_next = x_base + lambda_dn * (x_denoised - x_base) + lambda_pr * (x_prior - x_base)
        if self.use_beta_scale:
            x_next = beta_scale * x_next
        x_next = self.safe_complex_nan_to_num(x_next, nan=0.0, posinf=1.0, neginf=0.0)
        x_next = self._normalize_image(x_next)

        return x_next, r