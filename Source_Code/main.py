"""
main.py — RTT Project Entry Point
Runs full pipeline: generate dataset → train model → evaluate → plot results.

Usage:
    python main.py                    # full pipeline
    python main.py --mode evaluate    # skip training, load saved model
    python main.py --no_plot          # skip matplotlib output
    python main.py --threshold 0.70   # custom Humanity Score threshold
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def parse_args():
    p = argparse.ArgumentParser(description="RTT: Reverse Turing Test Framework")
    p.add_argument("--mode",      default="full",
                   choices=["full", "train", "evaluate"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Humanity Score threshold (default: 0.65 from config)")
    p.add_argument("--no_plot",   action="store_true")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def run(args):
    np.random.seed(args.seed)
    threshold = args.threshold or config.THRESHOLD_HIGH

    # ── 1. Generate / Load dataset ─────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    if not os.path.exists(config.DATA_PATH):
        print("  Dataset not found — generating...")
        from data.generate_dataset import (
            gen_human, gen_ai,
            N_HUMAN, N_GPT4, N_CLAUDE, N_GEMINI, N_LLAMA, N_MISTRAL
        )
        df = pd.concat([
            gen_human(N_HUMAN),
            gen_ai(N_GPT4,    "gpt4",    5.8, 0.35, 1.8, 9,  2.5, 8,  0.8),
            gen_ai(N_CLAUDE,  "claude",  5.6, 0.30, 1.5, 10, 2.0, 9,  0.6),
            gen_ai(N_GEMINI,  "gemini",  5.9, 0.38, 2.0, 8,  2.8, 7,  0.9),
            gen_ai(N_LLAMA,   "llama",   6.2, 0.45, 2.2, 7,  2.2, 8,  1.1),
            gen_ai(N_MISTRAL, "mistral", 6.0, 0.40, 2.1, 7,  2.3, 7,  1.0),
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv(config.DATA_PATH, index=False)
    else:
        df = pd.read_csv(config.DATA_PATH)

    print(f"  {len(df)} sessions loaded  |  "
          f"Human: {(df.humanity_score==1).sum()}  |  "
          f"AI: {(df.humanity_score==0).sum()}")

    X = df[config.ALL_FEATURES].values
    y = df["humanity_score"].values
    agent_types = df["agent_type"].values

    X_train, X_test, y_train, y_test, at_train, at_test = train_test_split(
        X, y, agent_types,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    df_test = df.iloc[
        df.index.get_indexer_for(
            pd.DataFrame(X_test, columns=config.ALL_FEATURES).index
        ) if False else []
    ]
    # Simpler: rebuild df_test from indices
    test_idx = np.where(np.isin(np.arange(len(df)),
                                np.random.choice(len(df), len(y_test), replace=False)))[0]

    # ── 2. Layer Scorer ────────────────────────────────────────────────────
    print("\n[2/5] Computing RTT layer scores...")
    from models.layer_scorer import LayerScorer
    layer_scorer = LayerScorer()
    df_scored = layer_scorer.compute_all(df)
    layer_means = layer_scorer.mean_layer_scores(df)
    print("\n  Mean Layer Scores by Agent Type:")
    print(layer_means.to_string())

    # ── 3. Train / Load Classifier ─────────────────────────────────────────
    print("\n[3/5] Training Humanity Scoring Model...")
    from models.classifier import HumanityScorer
    if args.mode in ("full", "train") or not os.path.exists(config.MODEL_SAVE_PATH):
        scorer = HumanityScorer()
        scorer.fit(X_train, y_train)
        scorer.save()
    else:
        print("  Loading saved model...")
        scorer = HumanityScorer.load()

    y_scores_test = scorer.predict_score(X_test)
    importances   = scorer.feature_importances()

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    print("\n[4/5] Evaluating...")
    from evaluation.metrics import (
        compute_all_metrics, fdr_by_system, threshold_sweep
    )

    # Reconstruct df slice for test set (approximate)
    df_test_approx = pd.DataFrame(X_test, columns=config.ALL_FEATURES)
    df_test_approx["agent_type"]    = at_test
    df_test_approx["humanity_score"] = y_test

    metrics = compute_all_metrics(
        y_test, y_scores_test,
        df=df_test_approx,
        threshold=threshold,
        verbose=True,
    )

    fdr_df   = fdr_by_system(y_test, y_scores_test, at_test, threshold)
    sweep_df = threshold_sweep(y_test, y_scores_test)

    print("\n  FDR by AI System:")
    print(fdr_df.to_string(index=False))

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    fdr_df.to_csv("results/fdr_by_system.csv", index=False)
    sweep_df.to_csv("results/threshold_sweep.csv", index=False)
    df_scored[["agent_type","layer_embodied","layer_cognitive",
               "layer_behavioural","humanity_score_pred"]].to_csv(
        "results/layer_scores.csv", index=False
    )
    print("\n  Saved: results/metrics.json")
    print("  Saved: results/fdr_by_system.csv")
    print("  Saved: results/threshold_sweep.csv")
    print("  Saved: results/layer_scores.csv")

    # ── 5. Plot ────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n[5/5] Generating plots...")
        from evaluation.plots import (
            plot_latency_distribution, plot_bias_distribution,
            plot_layer_scores, plot_fdr_ufi_sweep, plot_feature_importances,
            plot_humanity_score_distribution, plot_captcha_comparison,
            plot_fdr_by_system, plot_ufi_components
        )
        plot_latency_distribution(df)
        plot_bias_distribution(df)
        plot_layer_scores(layer_means)
        plot_fdr_ufi_sweep(sweep_df)
        plot_feature_importances(importances)
        plot_humanity_score_distribution(y_test, y_scores_test)
        plot_captcha_comparison(metrics)
        plot_fdr_by_system(fdr_df)
        plot_ufi_components(metrics["ufi"])
    else:
        print("\n[5/5] Plotting skipped.")

    print("\n" + "╔" + "═"*40 + "╗")
    print("║     RTT Framework — Complete! ✓     ║")
    print("╚" + "═"*40 + "╝")
    print(f"  Accuracy    : {metrics['accuracy']:.1%}")
    print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    print(f"  FDR         : {metrics['fdr']:.4f}")
    print(f"  UFI         : {metrics['ufi']:.4f}")
    print(f"  Results     → results/")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    run(args)
