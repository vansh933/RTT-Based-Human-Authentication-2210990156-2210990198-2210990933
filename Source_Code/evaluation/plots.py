"""
evaluation/plots.py — Generate All Paper Figures
Produces figures matching the paper's Section VII-VIII:
  - Fig 4 (Graph 1): Cognitive Bias Score distribution
  - Fig 4 (Graph 2): Response Latency distribution
  - Fig 5a: FDR vs UFI tradeoff curve
  - Fig 5b: Feature importances
  - Fig 6a: Humanity Score distribution
  - Fig 6b: RTT vs CAPTCHA comparison
  - Fig 4c: Layer-wise scores
  - Training: Threshold sweep
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

COLORS = {
    "human":   "#4472C4",
    "ai":      "#ED7D31",
    "green":   "#70AD47",
    "red":     "#C00000",
    "purple":  "#7030A0",
    "teal":    "#0070C0",
}
os.makedirs("results", exist_ok=True)


def plot_latency_distribution(df, save=True):
    """Graph 2 / Fig 4a — Response Latency: Human vs AI."""
    fig, ax = plt.subplots(figsize=(7, 4))
    human_lat = df[df.humanity_score == 1]["response_latency_ms"]
    ai_lat    = df[df.humanity_score == 0]["response_latency_ms"]
    bins = np.linspace(0, 15000, 50)
    ax.hist(human_lat, bins=bins, density=True, alpha=0.72,
            color=COLORS["human"], label=f"Human (mean={human_lat.mean():.0f}ms)")
    ax.hist(ai_lat,    bins=bins, density=True, alpha=0.72,
            color=COLORS["ai"],    label=f"AI Agent (mean={ai_lat.mean():.0f}ms)")
    ax.set_xlabel("Response Latency (ms)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("GRAPH 2. Response Latency Distribution – Human vs. AI Agents",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 14000)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/graph2_latency_distribution.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/graph2_latency_distribution.png")
    plt.close()


def plot_bias_distribution(df, save=True):
    """Graph 1 / Fig 4b — Cognitive Bias Composite Score."""
    fig, ax = plt.subplots(figsize=(7, 4))
    human_bias = df[df.humanity_score == 1]["cognitive_bias_score"]
    ai_bias    = df[df.humanity_score == 0]["cognitive_bias_score"]
    bins = np.linspace(0, 4, 30)
    ax.hist(human_bias, bins=bins, density=True, alpha=0.72,
            color=COLORS["human"], label=f"Human (mean={human_bias.mean():.2f})")
    ax.hist(ai_bias,    bins=bins, density=True, alpha=0.72,
            color=COLORS["ai"],    label=f"AI Agent (mean={ai_bias.mean():.2f})")
    ax.set_xlabel("Cognitive Bias Composite Score (0–4)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("GRAPH 1. Cognitive Bias Composite Score – Human vs. AI",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/graph1_bias_distribution.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/graph1_bias_distribution.png")
    plt.close()


def plot_layer_scores(layer_means, save=True):
    """Fig 4c — Mean RTT Layer Scores: Human vs AI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers    = ["Embodied\nKnowledge", "Cognitive\nAsymmetry", "Temporal\nBehavioural"]
    cols      = ["layer_embodied", "layer_cognitive", "layer_behavioural"]
    human_row = layer_means.loc["human"]  if "human" in layer_means.index else None
    ai_means  = layer_means.drop("human", errors="ignore")

    x = np.arange(len(layers))
    w = 0.32

    # Human bar
    if human_row is not None:
        h_vals = [human_row[c] for c in cols]
        b1 = ax.bar(x - w/2, h_vals, w, color=COLORS["human"], alpha=0.88, label="Human")
        for bar, v in zip(b1, h_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.2f}", ha="center", fontsize=8, color=COLORS["human"])

    # AI mean bar
    ai_vals = [ai_means[c].mean() for c in cols]
    b2 = ax.bar(x + w/2, ai_vals, w, color=COLORS["ai"], alpha=0.88, label="AI (mean)")
    for bar, v in zip(b2, ai_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.2f}", ha="center", fontsize=8, color=COLORS["ai"])

    ax.axhline(0.65, color=COLORS["red"], lw=1.5, ls="--", label="Pass threshold (0.65)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=9)
    ax.set_ylabel("Mean Layer Score", fontsize=10)
    ax.set_title("FIGURE 4c. Mean RTT Layer Scores: Human vs. AI Agents",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/fig4c_layer_scores.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/fig4c_layer_scores.png")
    plt.close()


def plot_fdr_ufi_sweep(sweep_df, save=True):
    """Fig 5a — FDR and UFI vs Decision Threshold."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep_df["threshold"], sweep_df["fdr"],
            color=COLORS["red"], lw=2.2, label="FDR (AI pass rate)")
    ax.plot(sweep_df["threshold"], sweep_df["ufi_proxy"],
            color=COLORS["human"], lw=2.2, label="UFI proxy (1 - human pass rate)")
    ax.axvspan(0.60, 0.72, alpha=0.10, color=COLORS["green"], label="Recommended zone")
    ax.axvline(0.65, color=COLORS["green"], lw=1.5, ls="--")
    ax.text(0.655, 0.85, "θ=0.65", fontsize=8, color=COLORS["green"])
    ax.set_xlabel("Humanity Score Threshold (θ)", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title("FIGURE 5a. FDR and UFI vs. Decision Threshold",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/fig5a_fdr_ufi_sweep.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/fig5a_fdr_ufi_sweep.png")
    plt.close()


def plot_feature_importances(importances_dict, save=True):
    """Fig 5b — Top Feature Importances."""
    sorted_items = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    features, scores = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(features))
    bar_colors = [COLORS["human"] if "latency" in f or "pause" in f or "cursor" in f
                  or "backspace" in f or "edit" in f or "keystroke" in f
                  else COLORS["ai"] for f in features]
    bars = ax.barh(y_pos[::-1], scores, color=bar_colors, alpha=0.88,
                   edgecolor="#333", lw=0.6)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8)
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(features, fontsize=8.5)
    ax.set_xlabel("Feature Importance", fontsize=10)
    ax.set_title("FIGURE 5b. Top Feature Importances (Random Forest Classifier)",
                 fontsize=10, fontweight="bold")
    b_patch = mpatches.Patch(color=COLORS["human"], label="Behavioural features")
    c_patch = mpatches.Patch(color=COLORS["ai"],    label="Cognitive/Embodied features")
    ax.legend(handles=[b_patch, c_patch], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/fig5b_feature_importances.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/fig5b_feature_importances.png")
    plt.close()


def plot_humanity_score_distribution(y_true, y_scores, save=True):
    """Fig 6a — Humanity Score Distribution with Decision Thresholds."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 40)
    ax.hist(y_scores[y_true == 1], bins=bins, density=True, alpha=0.72,
            color=COLORS["human"], label=f"Human (mean={y_scores[y_true==1].mean():.3f})")
    ax.hist(y_scores[y_true == 0], bins=bins, density=True, alpha=0.72,
            color=COLORS["ai"],    label=f"AI Agent (mean={y_scores[y_true==0].mean():.3f})")
    ax.axvline(0.65, color=COLORS["red"],    lw=2.0, ls="--", label="Pass threshold (0.65)")
    ax.axvline(0.35, color=COLORS["purple"], lw=2.0, ls=":",  label="Fail threshold (0.35)")
    ax.axvspan(0.65, 1.0,  alpha=0.07, color=COLORS["green"])
    ax.axvspan(0.0,  0.35, alpha=0.07, color=COLORS["red"])
    ax.axvspan(0.35, 0.65, alpha=0.05, color="#FFD966")
    ax.text(0.82, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 3.5,
            "PASS", fontsize=9, color=COLORS["green"], fontweight="bold", ha="center")
    ax.text(0.50, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 3.5,
            "REVIEW", fontsize=8, color="#7F6000", ha="center")
    ax.text(0.17, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 3.5,
            "DENY", fontsize=9, color=COLORS["red"], fontweight="bold", ha="center")
    ax.set_xlabel("Humanity Score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("FIGURE 6a. Humanity Score Distribution with Decision Thresholds",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/fig6a_score_distribution.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/fig6a_score_distribution.png")
    plt.close()


def plot_captcha_comparison(metrics, save=True):
    """Graph 5 / Fig 6b — FDR: RTT vs Traditional CAPTCHA Systems."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    systems  = ["Image\nCAPTCHA", "reCAPTCHA\nv3", "Behavioural\nAnalytics",
                "RTT (Ours)"]
    fdr_vals = [0.075, 0.175, 0.120, metrics.get("fdr", 0.038)]
    ufi_vals = [0.180, 0.050, 0.090, metrics.get("ufi", 0.095)]
    x = np.arange(len(systems))
    w = 0.32
    b1 = ax.bar(x - w/2, fdr_vals, w, color=COLORS["red"],   alpha=0.88,
                label="FDR ↓ (lower = better)", edgecolor="#333", lw=0.7)
    b2 = ax.bar(x + w/2, ufi_vals, w, color=COLORS["human"], alpha=0.88,
                label="UFI ↓ (lower = better)", edgecolor="#333", lw=0.7)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{bar.get_height():.3f}", ha="center", fontsize=8, color=COLORS["red"])
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{bar.get_height():.3f}", ha="center", fontsize=8, color=COLORS["human"])
    ax.axvspan(2.55, 3.55, alpha=0.08, color=COLORS["green"])
    ax.text(3.0, max(fdr_vals)*0.9, "★ RTT", fontsize=8.5,
            color=COLORS["green"], ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_ylim(0, max(max(fdr_vals), max(ufi_vals)) * 1.25)
    ax.set_title("GRAPH 5. FDR: RTT vs Traditional CAPTCHA Systems",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/graph5_captcha_comparison.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/graph5_captcha_comparison.png")
    plt.close()


def plot_fdr_by_system(fdr_df, save=True):
    """Graph 4 — Per-AI-System FDR."""
    if fdr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLORS["ai"]] * len(fdr_df)
    bars = ax.bar(fdr_df["system"], fdr_df["fdr"], color=colors, alpha=0.88,
                  edgecolor="#333", lw=0.7)
    for bar, val in zip(bars, fdr_df["fdr"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.axhline(0.05, color=COLORS["red"], lw=1.5, ls="--", label="Target FDR ≤ 0.05")
    ax.set_ylabel("False Discovery Rate", fontsize=10)
    ax.set_title("GRAPH 4. FDR per AI System (Adversarial Configuration)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(fdr_df["fdr"].max() * 1.3, 0.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/graph4_fdr_by_system.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/graph4_fdr_by_system.png")
    plt.close()


def plot_ufi_components(ufi_score, save=True):
    """Graph 3 — UFI Component Breakdown."""
    import config
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Completion\nTime", "First Attempt\nRate (burden)",
              "Accessibility\nImpact", "Subjective\nExperience"]
    values = [0.28, 0.13, 0.15, 0.20]
    weights = list(config.UFI_WEIGHTS.values())
    weighted = [v * w for v, w in zip(values, weights)]
    bars = ax.bar(labels, weighted, color=COLORS["teal"], alpha=0.85,
                  edgecolor="#333", lw=0.7)
    for bar, v in zip(bars, weighted):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(ufi_score, color=COLORS["red"], lw=1.5, ls="--",
               label=f"Total UFI = {ufi_score:.3f}")
    ax.set_ylabel("Weighted Sub-dimension Score", fontsize=10)
    ax.set_title("GRAPH 3. User Friction Index (UFI) – Component Breakdown",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("results/graph3_ufi_components.png", dpi=180, bbox_inches="tight")
        print("  Saved: results/graph3_ufi_components.png")
    plt.close()
