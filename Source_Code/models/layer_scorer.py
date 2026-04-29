"""
models/layer_scorer.py — Per-Layer Humanity Score Computation
Computes separate scores for each RTT layer (Section V-A):
  1. Embodied Knowledge Verification
  2. Cognitive Asymmetry Challenges
  3. Temporal Behavioural Analysis
Combines them into a weighted Humanity Score.
"""

import numpy as np
import config


class LayerScorer:
    """
    Computes layer-wise Humanity Scores using threshold-based normalisation.
    Each layer outputs a score ∈ [0, 1] where higher = more human-like.
    """

    def __init__(self):
        self.weights = config.LAYER_WEIGHTS

    def embodied_score(self, df):
        """
        Layer 1: Embodied Knowledge Verification (Section V-B).
        Based on embodied_score feature directly from RTT challenge responses.
        Humans score higher due to grounded physical metaphor comprehension.
        """
        return np.clip(df["embodied_score"].values, 0, 1)

    def cognitive_score(self, df):
        """
        Layer 2: Cognitive Asymmetry Challenges (Section V-C).
        Combines:
          - cognitive_bias_score (anchoring + confirmation + availability bias)
          - moral_reasoning_speed (humans slower under time pressure)
          - response_entropy (humans produce more diverse/idiosyncratic text)
        """
        # Normalise bias score [0-4] → [0-1]; higher bias = more human
        bias_norm = np.clip(df["cognitive_bias_score"].values / 4.0, 0, 1)

        # Moral reasoning speed: slow (>5000ms) = human, fast (<1000ms) = AI
        moral_ms = df["moral_reasoning_speed"].values
        moral_norm = np.clip((moral_ms - 300) / (30000 - 300), 0, 1)

        # Entropy: humans have moderate-high entropy; calibrate around 3.5
        entropy = df["response_entropy"].values
        entropy_norm = np.clip((entropy - 1.5) / (5.5 - 1.5), 0, 1)

        return (0.45 * bias_norm + 0.35 * moral_norm + 0.20 * entropy_norm)

    def behavioural_score(self, df):
        """
        Layer 3: Temporal Behavioural Analysis (Section V-A point 3).
        Combines keystroke dynamics, cursor corrections, response latency.
        Humans exhibit higher variability and longer latencies.
        """
        # Response latency: humans 1000-15000ms; AI <500ms
        latency = df["response_latency_ms"].values
        latency_norm = np.clip((latency - 30) / (20000 - 30), 0, 1)

        # Backspace count: humans make more edits
        backspace = df["backspace_count"].values
        backspace_norm = np.clip(backspace / 20.0, 0, 1)

        # Edit intensity
        edit = df["edit_intensity"].values

        # Cursor corrections: humans have more irregular paths
        cursor = df["cursor_corrections"].values
        cursor_norm = np.clip(cursor / 20.0, 0, 1)

        # Pause duration: humans have longer pauses
        pause = df["avg_pause_duration_ms"].values
        pause_norm = np.clip((pause - 10) / (3000 - 10), 0, 1)

        return (0.30 * latency_norm +
                0.20 * backspace_norm +
                0.15 * edit +
                0.20 * cursor_norm +
                0.15 * pause_norm)

    def compute_all(self, df):
        """
        Compute all three layer scores and weighted Humanity Score.
        Returns DataFrame with per-layer scores and final score.
        """
        import pandas as pd
        result = df.copy()
        result["layer_embodied"]    = self.embodied_score(df)
        result["layer_cognitive"]   = self.cognitive_score(df)
        result["layer_behavioural"] = self.behavioural_score(df)
        result["humanity_score_pred"] = (
            self.weights["embodied"]    * result["layer_embodied"] +
            self.weights["cognitive"]   * result["layer_cognitive"] +
            self.weights["behavioural"] * result["layer_behavioural"]
        )
        return result

    def mean_layer_scores(self, df, group_col="agent_type"):
        """Return mean per-layer scores grouped by agent type."""
        scored = self.compute_all(df)
        cols = ["layer_embodied", "layer_cognitive", "layer_behavioural",
                "humanity_score_pred", group_col]
        return scored[cols].groupby(group_col).mean().round(3)
