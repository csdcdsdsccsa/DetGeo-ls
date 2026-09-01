"""Lightweight frozen-DetGeo candidate ranker for the first ranking experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateRanker(nn.Module):
    """Score K NMS proposals from query/candidate features and detector metadata."""

    def __init__(self, feature_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_features, candidate_features, detector_scores, boxes_xyxy, image_size):
        """Return one logit per candidate.

        Args:
            query_features: [B, C, Hq, Wq]
            candidate_features: [B, K, C, Hr, Wr]
            detector_scores: [B, K] sigmoid-normalized detection confidences
            boxes_xyxy: [B, K, 4] in input-image coordinates
        """
        batch_size, candidate_count, channels = candidate_features.shape[:3]
        query_global = F.adaptive_avg_pool2d(query_features, 1).flatten(1)
        candidate_global = candidate_features.mean(dim=(-1, -2))

        centres = (boxes_xyxy[..., :2] + boxes_xyxy[..., 2:]) * 0.5 / float(image_size)
        sizes = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp(min=0) / float(image_size)
        metadata = torch.cat((detector_scores.unsqueeze(-1), centres, sizes), dim=-1)
        query_global = query_global.unsqueeze(1).expand(-1, candidate_count, -1)
        features = torch.cat((query_global, candidate_global, metadata), dim=-1)
        return self.mlp(features.reshape(batch_size * candidate_count, -1)).reshape(batch_size, candidate_count)


class ResidualCandidateRanker(CandidateRanker):
    """Learn a correction while preserving the detector ordering at initialization."""

    def __init__(self, feature_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, query_features, candidate_features, detector_logits, boxes_xyxy, image_size):
        delta_scores = super().forward(
            query_features, candidate_features, detector_logits, boxes_xyxy, image_size)
        return detector_logits + delta_scores


class NormalizedResidualCandidateRanker(CandidateRanker):
    """Residual scorer using per-image Top-K z-normalized detector logits."""

    def __init__(self, feature_dim=512, hidden_dim=256, dropout=0.1,
                 residual_alpha=1.0, score_eps=1e-6):
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.residual_alpha = float(residual_alpha)
        self.score_eps = float(score_eps)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def normalize_scores(self, detector_logits):
        mean = detector_logits.mean(dim=1, keepdim=True)
        std = detector_logits.std(dim=1, keepdim=True, unbiased=False)
        return (detector_logits - mean) / (std + self.score_eps)

    def forward(self, query_features, candidate_features, detector_logits, boxes_xyxy, image_size):
        normalized_scores = self.normalize_scores(detector_logits)
        delta_scores = super().forward(
            query_features, candidate_features, normalized_scores, boxes_xyxy, image_size)
        return normalized_scores + self.residual_alpha * delta_scores


class ResidualCrossAttentionRanker(nn.Module):
    """Lightweight spatial Query-to-Candidate matching with a residual score head."""

    def __init__(self, feature_dim=512, match_dim=128, num_heads=4,
                 hidden_dim=64, dropout=0.1, residual_alpha=1.0):
        super().__init__()
        if match_dim % num_heads != 0:
            raise ValueError("match_dim must be divisible by num_heads")
        self.match_dim = match_dim
        self.num_heads = num_heads
        self.head_dim = match_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.residual_alpha = float(residual_alpha)
        self.attention_dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(feature_dim, match_dim, bias=False)
        self.key_projection = nn.Linear(feature_dim, match_dim, bias=False)
        self.value_projection = nn.Linear(feature_dim, match_dim, bias=False)
        self.match_norm = nn.LayerNorm(match_dim)
        self.score_head = nn.Sequential(
            nn.Linear(match_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def _split_heads(self, tokens):
        batch_size, token_count, _ = tokens.shape
        return tokens.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query_features, candidate_features, detector_logits, boxes_xyxy, image_size):
        del boxes_xyxy, image_size  # Kept in the shared ranker interface; matching is appearance-only.
        batch_size, candidate_count, channels = candidate_features.shape[:3]
        query_tokens = query_features.flatten(2).transpose(1, 2)
        candidate_tokens = candidate_features.flatten(3).permute(0, 1, 3, 2)

        query_tokens = self.query_projection(query_tokens)
        query_tokens = query_tokens[:, None].expand(-1, candidate_count, -1, -1)
        query_tokens = query_tokens.reshape(batch_size * candidate_count, -1, self.match_dim)
        candidate_tokens = candidate_tokens.reshape(batch_size * candidate_count, -1, channels)
        key_tokens = self.key_projection(candidate_tokens)
        value_tokens = self.value_projection(candidate_tokens)

        query_heads = self._split_heads(query_tokens)
        key_heads = self._split_heads(key_tokens)
        value_heads = self._split_heads(value_tokens)
        attention = torch.softmax(
            torch.matmul(query_heads, key_heads.transpose(-2, -1)) * self.scale, dim=-1)
        attention = self.attention_dropout(attention)
        matched = torch.matmul(attention, value_heads)
        matched = matched.transpose(1, 2).reshape(
            batch_size * candidate_count, -1, self.match_dim)
        matched = self.match_norm(matched).mean(dim=1)
        delta_scores = self.score_head(matched).reshape(batch_size, candidate_count)
        return detector_logits + self.residual_alpha * delta_scores
