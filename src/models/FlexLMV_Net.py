import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from asymmetry_model.mirai_localized_dif_head import extract_mirai_backbone
from models.model_utils import (
    SpatialTransformerBlock,
    ContinuousPosEncoding,
    CumulativeProbabilityLayer,
    CrossAttentionBlock,
)

class SpatialAttentionModule(nn.Module):
    """Computes spatial attention maps for feature fusion."""
    def __init__(self, channels: int):
        super().__init__()
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Spatial attention (deeper block)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, feat):
        """
        Args:
            feat: (B, C, H, W) feature map
        Returns:
            attn_map: (B, 1, H, W) spatial attention map
            refined_feat: (B, C, H, W) feature with attention applied
        """
        # Channel attention
        ca = self.channel_attn(feat)
        feat_ca = feat * ca

        # Spatial attention
        sa = self.spatial_attn(feat_ca)

        refined_feat = feat_ca * sa
        return sa, refined_feat



class FeatureAwareTemporalAggregator(nn.Module):
    def __init__(
        self,
        channels: int,
        max_priors: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.channels = channels
        self.max_priors = max_priors

        # Temporal importance scoring network with stronger regularization
        self.score_net = nn.Sequential(
            nn.Linear(channels*2+1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.out_dropout = nn.Dropout2d(dropout)
        # Shared spatial attention module for prior and diff features
        self.spatial_attn = SpatialAttentionModule(channels)
        
        self.spatial_attention_maps = None  # Store for visualization
        self.temporal_weights = None  # Store temporal importance weights

    def _gap_tensor(self, gap, batch_size, device):
        if isinstance(gap, torch.Tensor):
            gap = gap.to(device=device, dtype=torch.float32)
            if gap.numel() == 1:
                gap = gap.reshape(1).expand(batch_size)
            return gap.reshape(batch_size)

        return torch.full((batch_size,), float(gap), device=device)

    def _aggregate_one(self, cur_feat, feats, gaps, spatial_attn_module):
        """
        Fuse multiple prior features using spatial + temporal attention.
        
        Args:
            cur_feat: (B, C, H, W) current features
            feats: list of (B, C, H, W) prior features
            gaps: list of time gaps
            spatial_attn_module: SpatialAttentionModule instance
            
        Returns:
            fused: (B, C, H, W) fused features
            weights: (B, N) temporal importance weights
            spatial_maps: (B, N, 1, H, W) spatial attention maps for visualization
        """
        if len(feats) == 0:
            zeros = cur_feat.new_zeros(cur_feat.shape)
            weights = cur_feat.new_zeros(cur_feat.size(0), 0)
            return zeros, weights, None

        feats = feats[: self.max_priors]
        gaps = gaps[: len(feats)]

        B, C, H, W = cur_feat.shape
        pooled_cur = F.adaptive_avg_pool2d(cur_feat, 1).flatten(1)

        scores = []
        spatial_attn_maps = []
        refined_feats = []

        for feat, gap in zip(feats, gaps):
            pooled_feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            gap_feat = self._gap_tensor(gap, B, cur_feat.device).unsqueeze(1)

            score_input = torch.cat([pooled_cur, pooled_feat, gap_feat], dim=1)
            scores.append(self.score_net(score_input))

            spatial_map, refined_feat = spatial_attn_module(feat)
            spatial_attn_maps.append(spatial_map)
            refined_feats.append(refined_feat)

        score_tensor = torch.cat(scores, dim=1)
        temporal_weights = torch.softmax(score_tensor, dim=1)

        stacked_feats = torch.stack(refined_feats, dim=1)
        stacked_spatial = torch.stack(spatial_attn_maps, dim=1)

        combined_weights = temporal_weights[:, :, None, None, None] * stacked_spatial
        combined_weights = combined_weights.squeeze(2)
        combined_weights /= combined_weights.sum(dim=1, keepdim=True) + 1e-8

        fused = torch.sum(combined_weights[:, :, None] * stacked_feats, dim=1)
        fused = self.out_dropout(fused)

        return fused, temporal_weights, torch.stack(spatial_attn_maps, dim=1)

    def forward(self, cur_feat, prior_feats, diff_feats, absolute_gaps, sequential_gaps):
        f_prior, w_prior, spatial_prior = self._aggregate_one(
            cur_feat, prior_feats, absolute_gaps, self.spatial_attn
        )
        f_diff, w_diff, spatial_diff = self._aggregate_one(
            cur_feat, diff_feats, absolute_gaps, self.spatial_attn
        )
        
        # Store attention maps and temporal weights for visualization
        self.spatial_attention_maps = {
            'prior': spatial_prior,
            'diff': spatial_diff,
        }
        self.temporal_weights = {
            'prior': w_prior,  # (B, N) - temporal importance for priors
            'diff': w_diff,    # (B, N) - temporal importance for differences
        }
        
        return f_prior, f_diff, w_prior, w_diff


class LongitudinalFeatureProcessorMultiplePriors(nn.Module):
    """Process current and multiple prior images to produce longitudinal features."""

    def __init__(self, mammo_reg_net, finetune=False, aggregation_mode="gated",
                 num_prior_img=2, aggregator_hidden_dim=64):
        super().__init__()
        self.num_prior_img, self.aggregation_mode = num_prior_img, aggregation_mode
        self.encoder = extract_mirai_backbone('/path/to/mirai_pretrained_backbone/mgh_mammo_MIRAI_Base_May20_2019.p')
        self.mammo_reg_net = mammo_reg_net
        self.feat_transformer = SpatialTransformerBlock(mode="bilinear")
        self.positional_encoding = ContinuousPosEncoding(dim=512)

        self.encoder.requires_grad_(not finetune)
        self.mammo_reg_net.requires_grad_(False)
        self.mammo_reg_net.eval()

        if finetune:
            print("Finetuning Mirai encoder")
            self.encoder.train()
        else:
            self.encoder.eval()

        self.aggregator = FeatureAwareTemporalAggregator(
            channels=512, max_priors=num_prior_img, dropout=0.4, hidden_dim=aggregator_hidden_dim
        ) if aggregation_mode == "gated" else None

    @staticmethod
    def _normalize_image(img, device):
        if img.ndim == 2: img = img[None, None]
        elif img.ndim == 3: img = img[None]
        if img.ndim != 4: raise ValueError(f"Expected image with 2-4 dimensions, got {img.shape}")
        return img.to(device)

    @staticmethod
    def _sequential_gaps(gaps):
        return [gap if i == 0 else gap - gaps[i - 1] for i, gap in enumerate(gaps)]

    @staticmethod
    def _to_gap_tensor(gap, device):
        return gap.to(device=device, dtype=torch.float32).reshape(()) if isinstance(gap, torch.Tensor) \
            else torch.tensor(float(gap), device=device, dtype=torch.float32)

    def _get_patient_priors(self, priors, b):
        if priors is None: return []
        if isinstance(priors, torch.Tensor):
            if priors.ndim == 3: return [priors[b]]
            if priors.ndim == 4: return [priors[b, i] for i in range(priors.size(1))]
            if priors.ndim == 5: return [priors[b, i] for i in range(priors.size(1))]
            raise ValueError(f"Unexpected prior tensor shape: {priors.shape}")
        return list(priors[b]) if b < len(priors) and priors[b] is not None else []

    def _get_patient_gaps(self, gaps, b, num_priors, device):
        if num_priors == 0: return []
        if gaps is None: return [torch.tensor(1.0, device=device)] * num_priors

        if isinstance(gaps, torch.Tensor):
            if gaps.ndim == 0: return [gaps.to(device=device, dtype=torch.float32)] * num_priors
            if gaps.ndim == 1:
                if gaps.numel() > b: return [gaps[b].to(device=device, dtype=torch.float32)] * num_priors
                return [gaps[i].to(device=device, dtype=torch.float32) for i in range(min(num_priors, gaps.numel()))]
            if gaps.ndim == 2:
                return [gaps[b, i].to(device=device, dtype=torch.float32) for i in range(min(num_priors, gaps.size(1)))]
            raise ValueError(f"Unexpected time gap tensor shape: {gaps.shape}")

        if b >= len(gaps) or gaps[b] is None: return [torch.tensor(1.0, device=device)] * num_priors
        patient_gaps = gaps[b]

        if isinstance(patient_gaps, torch.Tensor):
            if patient_gaps.ndim == 0: return [patient_gaps.to(device=device, dtype=torch.float32)] * num_priors
            return [patient_gaps[i].to(device=device, dtype=torch.float32) for i in range(min(num_priors, patient_gaps.numel()))]

        return [self._to_gap_tensor(gap, device) for gap in patient_gaps[:num_priors]]

    def _encode_image(self, img):
        img = self._normalize_image(img, img.device)
        return self.encoder(img.expand(-1, 3, -1, -1))

    def _align_prior(self, img_cur, f_cur, img_prior):
        img_prior = self._normalize_image(img_prior, img_cur.device)
        f_prior = self._encode_image(img_prior)
        deformation_field = self.mammo_reg_net(img_cur, img_prior)[1].detach()

        deformation_field = F.interpolate(deformation_field, size=f_cur.shape[-2:], mode="bilinear", align_corners=True)

        deformation_field[:, 0] *= f_cur.shape[-1] / img_cur.shape[-1]
        deformation_field[:, 1] *= f_cur.shape[-2] / img_cur.shape[-2]

        return self.feat_transformer(f_prior, deformation_field)

    def _encode_difference(self, f_previous, f_current, sequential_gap):
        diff = torch.abs(f_previous - f_current)
        if sequential_gap.ndim == 0: sequential_gap = sequential_gap.unsqueeze(0)

        B, C, H, W = diff.shape
        diff = diff.flatten(2).permute(2, 0, 1)
        diff = self.positional_encoding(diff, sequential_gap)
        return diff.permute(1, 2, 0).reshape(B, C, H, W)

    def _aggregate(self, current, prior_features, diff_features, absolute_gaps, sequential_gaps):
        if not prior_features: return torch.zeros_like(current), torch.zeros_like(current)
        if self.aggregation_mode != "gated":
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation_mode}")

        prior, diff, _, _ = self.aggregator(current, prior_features, diff_features, absolute_gaps, sequential_gaps)
        return prior, diff

    def _process_view(self, img_current, img_priors, time_gaps):
        B, device = img_current.size(0), img_current.device
        f_current_all = self._encode_image(img_current)
        f_prior_all, f_diff_all = torch.zeros_like(f_current_all), torch.zeros_like(f_current_all)

        for b in range(B):
            img_current_b, f_current_b = img_current[b:b + 1], f_current_all[b:b + 1]
            priors = self._get_patient_priors(img_priors, b)[:self.num_prior_img]
            if not priors: continue

            gaps = self._get_patient_gaps(time_gaps, b, len(priors), device)
            if len(gaps) < len(priors): gaps += [gaps[-1]] * (len(priors) - len(gaps))
            gaps, sequential_gaps = gaps[:len(priors)], self._sequential_gaps(gaps)

            aligned_priors = [self._align_prior(img_current_b, f_current_b, p) for p in priors]
            sequence = [f_current_b, *aligned_priors]
            diff_features = [
                self._encode_difference(sequence[i - 1], sequence[i], sequential_gaps[i - 1])
                for i in range(1, len(sequence))
            ]

            f_prior, f_diff = self._aggregate(f_current_b, aligned_priors, diff_features, gaps, sequential_gaps)
            f_prior_all[b:b + 1], f_diff_all[b:b + 1] = f_prior, f_diff

        return torch.cat([f_current_all, f_prior_all, f_diff_all], dim=1)

    def forward(self, img_cur_cc, img_pris_cc, img_cur_mlo, img_pris_mlo, time_gap):
        return {
            "f_cc_long": self._process_view(img_cur_cc, img_pris_cc, time_gap),
            "f_mlo_long": self._process_view(img_cur_mlo, img_pris_mlo, time_gap),
        }

  
class LongitudinalMultiViewRiskModel_multiple_prior(nn.Module):
    """
    Adjusted to handle multiple prior images.
    """
    def __init__(
        self, mammo_reg_net: nn.Module, finetune=False, max_followup=5,
        num_attn_blocks=1, dropout=0.4, drop_path=0.2,
        aggregation_mode='none', num_prior_img=2, aggregator_hidden_dim=64,
    ):
        super().__init__()
        feat_dim_atten = 1536
        self.longitudinal_feat_processor = LongitudinalFeatureProcessorMultiplePriors(
            mammo_reg_net=mammo_reg_net,finetune=finetune,  aggregation_mode = aggregation_mode, num_prior_img=num_prior_img, aggregator_hidden_dim=aggregator_hidden_dim
        )

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(in_channels=feat_dim_atten, reduced_channels=feat_dim_atten,
                                heads=4, dropout=dropout, drop_path=drop_path, ffn_expansion_factor=2)
            for _ in range(num_attn_blocks)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.view_fc = nn.Linear(feat_dim_atten * 2, feat_dim_atten)
        self.cumulative_risk = CumulativeProbabilityLayer(num_features=feat_dim_atten, max_followup=max_followup)

    def forward(self, img_cur_cc, img_pris_cc,
                      img_cur_mlo, img_pris_mlo, time_gap):

        features = self.longitudinal_feat_processor(img_cur_cc, img_pris_cc, img_cur_mlo, img_pris_mlo, time_gap)
        f_cc, f_mlo = features['f_cc_long'], features['f_mlo_long']

        for blk in self.cross_attn_blocks:
            f_cc, f_mlo = blk(f_cc, f_mlo)

        # Multi-view fusion
        pooled_cc = self.global_avg_pool(f_cc).flatten(1)
        pooled_mlo = self.global_avg_pool(f_mlo).flatten(1)
        pooled_multi = self.view_fc(torch.cat([pooled_cc, pooled_mlo], dim=1))

        # Risk prediction
        risk_multi = self.cumulative_risk(pooled_multi)
        risk_cc = self.cumulative_risk(pooled_cc)
        risk_mlo = self.cumulative_risk(pooled_mlo)

        return {
            'risk_multi': risk_multi, 
            'risk_cc': risk_cc, 
            'risk_mlo': risk_mlo,
        }