"""
TC-DANN: Task-Conditioned, Domain-Adversarial Network for Bridge2AI-Voice.

Design rationale (each choice maps to a finding in the Section 2 audit):

1. Four-branch fusion (spec, mel, EMA, static) instead of six.
   Audit Slide 18: PPG contributes negligibly, MFCC restoration did not close
   the MARVEL gap. Fewer branches -> less fusion conflict.

2. Explicit task-ID embedding as a fusion token.
   Audit Section 2.2: the task histogram alone is a strong predictor of
   cognitive impairment and Parkinson's. If task is a GIVEN input rather than
   something implicitly learnable from which-recordings-exist, the model
   cannot use it as a label proxy. The disease head is task-conditional.

3. Gradient reversal adversaries on site, age bucket, sex.
   Audit Section 2.4: country gap (Parkinson's, Canada vs USA), age gap
   (psych_history, 71+), sex gap (PTSD, female vs male).
   DANN (Ganin & Lempitsky 2015) actively strips these from the shared
   representation.

4. Disease-specific attention queries (kept from update presentation).
   Retains the inductive bias from the Mar 26 architecture and provides
   interpretable per-disease token weights for Section 4 figures.

5. Pre-LN transformer, masked stats pooling, key-padding mask.
   Retained from update for stability with missing modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GradientReversal, MaskedStatsPooling


# ------------------------------------------------------------------
# Per-modality encoders
# ------------------------------------------------------------------

class SpectrogramEncoder(nn.Module):
    """EfficientNet-B0 on 1-channel spectrogram, projected to d_model."""

    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                global_pool="avg",
            )
            feat_dim = self.backbone.num_features  # 1280
        except ImportError:
            self.backbone = _SmallCNN(in_ch=1, out_dim=1280)
            feat_dim = 1280
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class MelEncoder(nn.Module):
    """ResNet18 adapted to 1-channel mel-spectrogram."""

    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        orig = backbone.conv1.weight.data
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(orig.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class EMAEncoder(nn.Module):
    """1D-CNN on SPARC articulatory EMA features [B, C_in=12, T]."""

    def __init__(self, in_dim: int = 12, hidden: int = 256, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.pool = MaskedStatsPooling()
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = self.conv(x)
        h = h.transpose(1, 2)
        pooled = self.pool(h, mask)
        return self.proj(pooled)


class StaticEncoder(nn.Module):
    """MLP on static + prosodic aggregate features. Width is configurable."""

    def __init__(self, in_dim: int = 131, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SmallCNN(nn.Module):
    def __init__(self, in_ch: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Full TC-DANN
# ------------------------------------------------------------------

class TCDANN(nn.Module):
    """
    Task-Conditioned, Domain-Adversarial multimodal voice model.

    Args:
        n_diseases:     number of binary disease heads
        n_tasks:        number of distinct recording tasks in Bridge2AI
        n_sites:        number of sites / countries
        n_age_buckets:  number of age buckets (default 4)
        n_sex:          number of sex categories (default 2)
        n_static:       width of the static-features vector (auto-detected in run.py)
        d_model:        shared embedding dimension
        nhead:          transformer attention heads
        nlayers:        transformer encoder layers
    """

    N_TOKENS = 6  # CLS + spec + mel + ema + static + task

    def __init__(
        self,
        n_diseases: int,
        n_tasks: int,
        n_sites: int,
        n_age_buckets: int = 4,
        n_sex: int = 2,
        n_static: int = 131,
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_diseases = n_diseases
        self.d_model = d_model

        self.spec_enc = SpectrogramEncoder(out_dim=d_model)
        self.mel_enc = MelEncoder(out_dim=d_model)
        self.ema_enc = EMAEncoder(out_dim=d_model)
        self.static_enc = StaticEncoder(in_dim=n_static, out_dim=d_model)

        self.task_emb = nn.Embedding(n_tasks, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.token_type = nn.Parameter(torch.randn(1, self.N_TOKENS, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.disease_queries = nn.Parameter(torch.randn(n_diseases, d_model) * 0.02)
        self.disease_scale = d_model ** -0.5

        self.disease_heads = nn.ModuleList(
            [nn.Sequential(nn.Dropout(0.5), nn.Linear(d_model, 1))
             for _ in range(n_diseases)]
        )

        self.grl = GradientReversal(lambda_=0.0)
        self.adv_site = self._adv_head(d_model, n_sites)
        self.adv_age = self._adv_head(d_model, n_age_buckets)
        self.adv_sex = self._adv_head(d_model, n_sex)

        self.task_head = self._adv_head(d_model, n_tasks)

    @staticmethod
    def _adv_head(in_dim: int, n_out: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_out),
        )

    def set_grl_lambda(self, lambda_: float) -> None:
        self.grl.lambda_ = lambda_

    def forward(self, batch: dict) -> dict:
        B = batch["task_id"].size(0)
        device = batch["task_id"].device

        spec_t = self.spec_enc(batch["spec"])
        mel_t = self.mel_enc(batch["mel"])
        ema_t = self.ema_enc(batch["ema"], batch.get("ema_mask"))
        stat_t = self.static_enc(batch["static"])
        task_t = self.task_emb(batch["task_id"])

        cls = self.cls_token.expand(B, -1, -1)
        mod_tokens = torch.stack([spec_t, mel_t, ema_t, stat_t, task_t], dim=1)
        tokens = torch.cat([cls, mod_tokens], dim=1)
        tokens = tokens + self.token_type

        present = batch.get("present")
        if present is None:
            present = torch.ones(B, 4, dtype=torch.bool, device=device)
        kp = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            ~present,
            torch.zeros(B, 1, dtype=torch.bool, device=device),
        ], dim=1)

        fused = self.fusion(tokens, src_key_padding_mask=kp)
        cls_out = fused[:, 0]

        Q = self.disease_queries.unsqueeze(0).expand(B, -1, -1)
        K = fused
        V = fused
        scores = torch.matmul(Q, K.transpose(1, 2)) * self.disease_scale
        scores = scores.masked_fill(kp.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        disease_ctx = torch.matmul(attn, V)

        disease_logits = torch.stack(
            [head(disease_ctx[:, i]).squeeze(-1) for i, head in enumerate(self.disease_heads)],
            dim=1,
        )

        cls_reversed = self.grl(cls_out)
        site_logits = self.adv_site(cls_reversed)
        age_logits = self.adv_age(cls_reversed)
        sex_logits = self.adv_sex(cls_reversed)
        task_logits = self.task_head(cls_out)

        return {
            "disease_logits": disease_logits,
            "site_logits": site_logits,
            "age_logits": age_logits,
            "sex_logits": sex_logits,
            "task_logits": task_logits,
            "embedding": cls_out,
            "disease_attn": attn,
        }
