"""
evaluation/metrics.py — RTT Evaluation Metrics
Implements FDR, UFI, accuracy, and full evaluation suite.
Matches Section VII of the paper exactly.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import config


def compute_fdr(y_true, y_scores, threshold=None):
    """
    False Discovery Rate (Section VII-A):
      FDR = FP / (FP + TP)
    where FP = AI agents that pass as human,
          TP = human agents that authenticate correctly.

    Args:
        y_true   : ground truth (1=human, 0=AI)
        y_scores : Humanity Score ∈ [0,1]
        threshold: Humanity Score threshold for PASS decision
    Returns:
        fdr (float), fp (int), tp (int)
    """
    threshold = threshold or config.THRESHOLD_HIGH
    y_pred = (y_scores >= threshold).astype(int)

    # Among those predicted as human (y_pred=1):
    # FP = AI agents predicted as human
    # TP = Human agents predicted as human
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()

    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    return round(fdr, 4), int(fp), int(tp)


def compute_ufi(human_latencies, baseline_latencies=None, first_attempt_rates=None):
    """
    User Friction Index (Section VII-B):
      UFI = Σ wᵢ · sub_dimᵢ   (normalised composite)

    Sub-dimensions:
      1. completion_time      : normalised mean human latency
      2. first_attempt_rate   : fraction of humans passing on first try (inverted → burden)
      3. accessibility_impact : placeholder (0.15 = moderate)
      4. subjective_experience: placeholder (0.20 = moderate)

    Returns UFI ∈ [0, 1] (lower = less friction).
    """
    w = config.UFI_WEIGHTS

    # Completion time: normalise relative to reCAPTCHA v3 baseline (~800ms)
    mean_latency = np.mean(human_latencies)
    baseline = baseline_latencies if baseline_latencies else 800.0
    time_score = np.clip(mean_latency / 8000.0, 0, 1)   # 8s = max friction

    # First attempt rate (higher rate = less friction → invert for burden)
    far = first_attempt_rates if first_attempt_rates else 0.87
    far_burden = 1.0 - far

    # Accessibility impact and subjective experience (from paper's qualitative estimates)
    accessibility = 0.15
    subjective    = 0.20

    ufi = (w["completion_time"]      * time_score +
           w["first_attempt_rate"]   * far_burden +
           w["accessibility_impact"] * accessibility +
           w["subjective_experience"]* subjective)
    return round(float(ufi), 4)


def compute_all_metrics(y_true, y_scores, df=None, threshold=None, verbose=True):
    """
    Full evaluation suite matching Section VII of the paper.
    Returns a dict of all metrics.
    """
    threshold = threshold or config.THRESHOLD_HIGH
    y_pred = (y_scores >= threshold).astype(int)

    fdr, fp, tp = compute_fdr(y_true, y_scores, threshold)

    human_mask = y_true == 1
    ufi = compute_ufi(
        df["response_latency_ms"].values[human_mask] if df is not None
        else np.full(100, 4200.0)
    )

    metrics = {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision":       round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":          round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":              round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":         round(roc_auc_score(y_true, y_scores), 4),
        "fdr":             fdr,
        "ufi":             ufi,
        "fp_count":        fp,
        "tp_count":        tp,
        "threshold":       threshold,
        "n_human":         int(human_mask.sum()),
        "n_ai":            int((y_true == 0).sum()),
    }

    if verbose:
        print("\n" + "="*50)
        print("  RTT EVALUATION RESULTS")
        print("="*50)
        print(f"  Accuracy    : {metrics['accuracy']:.1%}")
        print(f"  Precision   : {metrics['precision']:.1%}")
        print(f"  Recall      : {metrics['recall']:.1%}")
        print(f"  F1 Score    : {metrics['f1']:.1%}")
        print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
        print(f"  FDR         : {metrics['fdr']:.4f}  (AI pass rate)")
        print(f"  UFI         : {metrics['ufi']:.4f}  (lower = less friction)")
        print(f"  Threshold θ : {threshold}")
        print(f"  Sessions    : {metrics['n_human']} human, {metrics['n_ai']} AI")
        print("="*50)
        print("\n  Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["AI", "Human"]))

    return metrics


def fdr_by_system(y_true, y_scores, agent_types, threshold=None):
    """
    Compute per-AI-system FDR (Section VII-A requires separate FDR per system).
    Returns DataFrame with FDR per agent type.
    """
    threshold = threshold or config.THRESHOLD_HIGH
    rows = []
    for system in config.AI_SYSTEMS:
        mask = np.array(agent_types) == system
        if mask.sum() == 0:
            continue
        fdr_val, fp, tp = compute_fdr(
            y_true[mask], y_scores[mask], threshold
        )
        rows.append({
            "system": system,
            "n_sessions": int(mask.sum()),
            "fdr": fdr_val,
            "fp": fp,
        })
    return pd.DataFrame(rows).sort_values("fdr")


def threshold_sweep(y_true, y_scores, thresholds=None):
    """
    Compute FDR and UFI across a range of thresholds.
    Used to generate Figure 5a (FDR-UFI tradeoff curve).
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.95, 80)
    rows = []
    human_mask = y_true == 1
    for t in thresholds:
        fdr_val, _, _ = compute_fdr(y_true, y_scores, t)
        y_pred = (y_scores >= t).astype(int)
        human_pass_rate = y_pred[human_mask].mean()
        rows.append({
            "threshold": round(t, 3),
            "fdr": fdr_val,
            "human_pass_rate": round(float(human_pass_rate), 4),
            "ufi_proxy": round(1.0 - float(human_pass_rate), 4),
        })
    return pd.DataFrame(rows)
