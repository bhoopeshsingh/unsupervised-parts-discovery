# src/models/lab_extractor.py
"""
Lab Record Feature Extractor

Converts a DataFrame of raw lab values into normalized feature tensors
ready for clustering. Three methods:

  clinical_deviation  [DEFAULT]
    Encodes each value as a signed deviation from its clinical normal range.
      0    = exactly in the middle of the normal range
      ±1   = at the boundary of the normal range
      >±1  = abnormal (capped at ±4 to reduce extreme outlier influence)
    This is the recommended method — clusters will naturally group by
    clinical abnormality pattern (anaemia, CKD, dyslipidaemia, etc.)

  zscore
    Simple population z-score using RobustScaler. Fast baseline.

  biomedbert
    Stringifies each record ("WBC: 11.2 (HIGH), HGB: 9.4 (LOW)…") and
    passes through BiomedBERT → 768-dim CLS embedding. Requires transformers.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import RobustScaler


class LabExtractor:
    """
    Fit on training records, transform any new records into [N, D] tensors.
    Compatible with PatchClusterer (same interface: torch.Tensor → cluster).
    """

    def __init__(self, config_path: str = "configs/config_lab.yaml"):
        cfg = yaml.safe_load(open(config_path))
        self.method = cfg["embedding"]["method"]
        self.clinical_ranges: Dict[str, List[float]] = cfg["embedding"]["clinical_ranges"]
        self.biomedbert_model_name: str = cfg["embedding"].get(
            "biomedbert_model",
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        )
        self._scaler = RobustScaler()
        self._fallback_stats: Dict[str, Tuple[float, float]] = {}
        self._bert_tokenizer = None
        self._bert_model = None
        self._feature_labels: Dict[str, str] = {}
        # Load feature labels from config
        for panel in cfg["lab_data"]["panels"].values():
            self._feature_labels.update(panel["features"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, features_df: pd.DataFrame, feature_cols: List[str]
    ) -> torch.Tensor:
        """Fit on data and return [N, D] tensor."""
        return self._encode(features_df, feature_cols, fit=True)

    def transform(
        self, features_df: pd.DataFrame, feature_cols: List[str]
    ) -> torch.Tensor:
        """Transform new records using fitted parameters."""
        return self._encode(features_df, feature_cols, fit=False)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "method": self.method,
                    "scaler": self._scaler,
                    "fallback_stats": self._fallback_stats,
                    "clinical_ranges": self.clinical_ranges,
                    "feature_labels": self._feature_labels,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, config_path: str = "configs/config_lab.yaml") -> "LabExtractor":
        obj = cls(config_path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj.method = state["method"]
        obj._scaler = state["scaler"]
        obj._fallback_stats = state["fallback_stats"]
        obj.clinical_ranges = state["clinical_ranges"]
        obj._feature_labels = state["feature_labels"]
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(
        self, features_df: pd.DataFrame, feature_cols: List[str], fit: bool
    ) -> torch.Tensor:
        X = features_df[feature_cols].values.astype(np.float32)

        if self.method == "clinical_deviation":
            X = self._clinical_deviation(X, feature_cols, fit=fit)
        elif self.method == "zscore":
            X = self._scaler.fit_transform(X) if fit else self._scaler.transform(X)
            X = X.astype(np.float32)
        elif self.method == "biomedbert":
            X = self._biomedbert_embed(features_df, feature_cols)
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")

        return torch.tensor(X, dtype=torch.float32)

    def _clinical_deviation(
        self, X: np.ndarray, feature_cols: List[str], fit: bool
    ) -> np.ndarray:
        """
        Per-feature: deviation = (value − midpoint) / half_range
        Features without a clinical range fall back to z-score.
        Result is capped at ±4 to limit extreme outlier influence.
        """
        result = X.copy()
        for i, col in enumerate(feature_cols):
            if col in self.clinical_ranges:
                lo, hi = self.clinical_ranges[col]
                midpoint = (lo + hi) / 2.0
                half_range = max((hi - lo) / 2.0, 1e-6)
                result[:, i] = (X[:, i] - midpoint) / half_range
            else:
                # z-score fallback for features with no clinical range defined
                if fit:
                    mu = float(np.nanmean(X[:, i]))
                    sigma = float(np.nanstd(X[:, i]))
                    self._fallback_stats[col] = (mu, sigma)
                else:
                    mu, sigma = self._fallback_stats.get(col, (0.0, 1.0))
                sigma = max(sigma, 1e-6)
                result[:, i] = (X[:, i] - mu) / sigma

        return np.clip(result, -4.0, 4.0).astype(np.float32)

    def _biomedbert_embed(
        self, features_df: pd.DataFrame, feature_cols: List[str]
    ) -> np.ndarray:
        """
        Stringify each record, embed via BiomedBERT CLS token.
        Returns [N, 768] float32 array.
        """
        from transformers import AutoModel, AutoTokenizer

        if self._bert_tokenizer is None:
            print(f"  Loading BiomedBERT ({self.biomedbert_model_name})…")
            self._bert_tokenizer = AutoTokenizer.from_pretrained(self.biomedbert_model_name)
            self._bert_model = AutoModel.from_pretrained(self.biomedbert_model_name)
            self._bert_model.eval()

        texts = []
        for _, row in features_df.iterrows():
            parts = []
            for col in feature_cols:
                val = row.get(col, float("nan"))
                if pd.notna(val):
                    label = self._feature_labels.get(col, col)
                    status = self._range_flag(col, float(val))
                    parts.append(f"{label}: {val:.1f}{status}")
            texts.append(". ".join(parts))

        embeddings = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start: start + batch_size]
            enc = self._bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self._bert_model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_emb)

        return np.vstack(embeddings).astype(np.float32)

    def _range_flag(self, col: str, val: float) -> str:
        if col not in self.clinical_ranges:
            return ""
        lo, hi = self.clinical_ranges[col]
        if val > hi:
            return " (HIGH)"
        if val < lo:
            return " (LOW)"
        return ""
