# src/pipeline/lab_concept_advisor.py
"""
LLM-based Clinical Cluster Label Advisor

For each cluster, builds a structured deviation profile and queries a
medical LLM to suggest a clinical label, confidence level, and reasoning.
The human SME reviews and accepts/modifies the suggestion.

Backends supported (configure in config_lab.yaml):
  ollama     — local BioMistral/Meditron via Ollama (free, reproducible, recommended)
  anthropic  — Claude claude-sonnet-4-6 via Anthropic API (best reasoning, needs API key)

Install Ollama: https://ollama.ai
Then pull a model:
  ollama pull mistral          # fast, general
  ollama pull meditron         # medical-focused (fine-tuned on PubMed + clinical guidelines)
  ollama pull llama3           # strong reasoning
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Deviation profile builder
# ---------------------------------------------------------------------------

def build_deviation_profile(
    cluster_id: int,
    features: torch.Tensor,
    cluster_labels: np.ndarray,
    feature_cols: List[str],
    feature_labels: Dict[str, str],
    clinical_ranges: Dict[str, List[float]],
    min_deviation: float = 0.75,
) -> Dict:
    """
    Compute the mean clinical deviation for each feature in the cluster
    and classify as SIGNIFICANTLY_HIGH / MILDLY_HIGH / NORMAL /
    MILDLY_LOW / SIGNIFICANTLY_LOW.

    Returns a structured dict ready to format into an LLM prompt.
    """
    mask = cluster_labels == cluster_id
    cluster_feats = features[mask].numpy()   # already in deviation space
    pop_feats = features.numpy()

    cluster_means = cluster_feats.mean(axis=0)
    pop_means = pop_feats.mean(axis=0)

    categories = {
        "significantly_high": [],   # deviation > 2.0
        "mildly_high":        [],   # 0.75 < deviation <= 2.0
        "normal":             [],   # |deviation| <= 0.75
        "mildly_low":         [],   # -2.0 <= deviation < -0.75
        "significantly_low":  [],   # deviation < -2.0
    }

    for i, col in enumerate(feature_cols):
        dev = float(cluster_means[i])
        pop_dev = float(pop_means[i])
        label = feature_labels.get(col, col).split(" (")[0]  # strip units for readability

        entry = {
            "code":       col,
            "label":      label,
            "deviation":  round(dev, 2),
            "vs_pop":     round(dev - pop_dev, 2),
        }

        if abs(dev) < min_deviation:
            categories["normal"].append(entry)
        elif dev >= 2.0:
            categories["significantly_high"].append(entry)
        elif dev >= min_deviation:
            categories["mildly_high"].append(entry)
        elif dev <= -2.0:
            categories["significantly_low"].append(entry)
        else:
            categories["mildly_low"].append(entry)

    return {
        "cluster_id":   cluster_id,
        "n_records":    int(mask.sum()),
        "categories":   categories,
    }


def format_profile_for_prompt(profile: Dict, feature_labels: Dict[str, str]) -> str:
    """Format a deviation profile as a structured clinical summary for the LLM."""
    lines = [
        f"Cluster of {profile['n_records']} patients with the following lab pattern:",
        "",
    ]

    cat_map = {
        "significantly_high": "SIGNIFICANTLY ELEVATED",
        "mildly_high":        "MILDLY ELEVATED",
        "significantly_low":  "SIGNIFICANTLY LOW",
        "mildly_low":         "MILDLY LOW",
    }

    for key, heading in cat_map.items():
        items = profile["categories"][key]
        if items:
            lines.append(f"  {heading}:")
            for e in sorted(items, key=lambda x: abs(x["deviation"]), reverse=True):
                lines.append(f"    • {e['label']}  (deviation {e['deviation']:+.1f} from normal midpoint)")

    normal_items = profile["categories"]["normal"]
    if normal_items:
        normal_names = ", ".join(e["label"] for e in normal_items[:6])
        if len(normal_items) > 6:
            normal_names += f", +{len(normal_items)-6} more"
        lines.append(f"\n  WITHIN NORMAL RANGE: {normal_names}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _build_prompt(profile_text: str) -> str:
    return f"""You are an expert clinical pathologist and haematologist.
A machine-learning pipeline has discovered the following lab value pattern
in an unsupervised cluster of patients from the NHANES population survey.

{profile_text}

Based solely on this lab pattern, provide:
1. PRIMARY_LABEL: A concise snake_case clinical label (e.g. iron_deficiency_anaemia, metabolic_syndrome, CKD_pattern, normal_population)
2. CONFIDENCE: high / medium / low
3. KEY_FINDINGS: 1-2 sentences explaining which abnormalities drove your conclusion
4. DIFFERENTIALS: 2-3 alternative clinical explanations to consider
5. SME_NOTE: One sentence of advice for the human expert reviewing this cluster

Respond in valid JSON with exactly these five keys."""


def query_ollama(
    profile_text: str,
    model: str = "llama3",
    host: str = "http://localhost:11434",
    timeout: int = 60,
) -> Optional[Dict]:
    """Query a locally running Ollama model."""
    try:
        import requests
        prompt = _build_prompt(profile_text)
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def query_anthropic(
    profile_text: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 600,
) -> Optional[Dict]:
    """Query Claude via the Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        prompt = _build_prompt(profile_text)
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Extract JSON block if wrapped in markdown
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

class LabConceptAdvisor:
    """
    Wraps the chosen LLM backend and provides one method: advise(cluster_id).
    Returns a dict with suggested label, confidence, reasoning.
    """

    def __init__(
        self,
        features: torch.Tensor,
        cluster_labels: np.ndarray,
        feature_cols: List[str],
        feature_labels: Dict[str, str],
        clinical_ranges: Dict[str, List[float]],
        backend: str = "ollama",          # "ollama" | "anthropic"
        ollama_model: str = "llama3",
        ollama_host: str = "http://localhost:11434",
        anthropic_model: str = "claude-sonnet-4-6",
    ):
        self.features = features
        self.cluster_labels = cluster_labels
        self.feature_cols = feature_cols
        self.feature_labels = feature_labels
        self.clinical_ranges = clinical_ranges
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.anthropic_model = anthropic_model

    def advise(self, cluster_id: int) -> Dict:
        """
        Build deviation profile for cluster_id and query the LLM.

        Returns dict with keys:
            PRIMARY_LABEL, CONFIDENCE, KEY_FINDINGS, DIFFERENTIALS,
            SME_NOTE, profile_text, error (if any)
        """
        profile = build_deviation_profile(
            cluster_id,
            self.features,
            self.cluster_labels,
            self.feature_cols,
            self.feature_labels,
            self.clinical_ranges,
        )
        profile_text = format_profile_for_prompt(profile, self.feature_labels)

        if self.backend == "anthropic":
            result = query_anthropic(profile_text, model=self.anthropic_model)
        else:
            result = query_ollama(profile_text, model=self.ollama_model, host=self.ollama_host)

        result["profile_text"] = profile_text
        result["n_records"] = profile["n_records"]
        return result

    @classmethod
    def from_cache(
        cls,
        cache_dir: str,
        config_path: str = "configs/config_lab.yaml",
        backend: str = "ollama",
        ollama_model: str = "llama3",
    ) -> "LabConceptAdvisor":
        """Convenience constructor that loads everything from the pipeline cache."""
        import yaml
        from pathlib import Path
        from src.data.lab_loader import get_feature_labels, get_clinical_ranges

        cfg = yaml.safe_load(open(config_path))
        cache = Path(cache_dir)

        data = torch.load(cache / "features.pt", weights_only=False)
        cluster_labels = torch.load(cache / "clusters.pt", weights_only=True).numpy()

        return cls(
            features=data["features"],
            cluster_labels=cluster_labels,
            feature_cols=data["feature_cols"],
            feature_labels=get_feature_labels(config_path),
            clinical_ranges=get_clinical_ranges(config_path),
            backend=backend,
            ollama_model=ollama_model,
        )
