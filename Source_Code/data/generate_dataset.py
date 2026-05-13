"""
generate_dataset.py — RTT Simulated Session Dataset Generator
Generates rtt_sessions.csv matching the paper's experimental design (Section VII).

Session features:
  - response_latency_ms     : time from challenge display to first keystroke
  - avg_pause_duration_ms   : mean pause between keystrokes
  - backspace_count         : number of backspace/delete actions
  - edit_intensity          : fraction of characters that were deleted
  - cursor_corrections      : mouse/cursor path irregularity count
  - first_keystroke_delay_ms: delay before any interaction begins
  - cognitive_bias_score    : composite score [0-4] (anchoring + confirmation + availability)
  - embodied_score          : metaphor comprehension + physical scenario score [0-1]
  - moral_reasoning_speed   : response time under time-pressure moral dilemma [ms]
  - response_entropy        : lexical diversity / perplexity of text response
  - humanity_score          : ground-truth label (human=1, ai=0)
  - agent_type              : 'human', 'gpt4', 'claude', 'gemini', 'llama', 'mistral'
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_HUMAN   = 750
N_GPT4    = 120
N_CLAUDE  = 100
N_GEMINI  = 80
N_LLAMA   = 100
N_MISTRAL = 80

def gen_human(n):
    return pd.DataFrame({
        "response_latency_ms":      np.clip(np.random.lognormal(8.3, 0.5, n), 500, 25000),
        "avg_pause_duration_ms":    np.clip(np.random.lognormal(6.1, 0.6, n), 80, 3000),
        "backspace_count":          np.random.negative_binomial(5, 0.4, n),
        "edit_intensity":           np.clip(np.random.beta(2.5, 6, n), 0, 1),
        "cursor_corrections":       np.random.poisson(8.2, n),
        "first_keystroke_delay_ms": np.clip(np.random.lognormal(7.1, 0.7, n), 100, 8000),
        "cognitive_bias_score":     np.clip(np.random.beta(6, 3, n) * 4, 0, 4),
        "embodied_score":           np.clip(np.random.beta(7, 2.5, n), 0, 1),
        "moral_reasoning_speed":    np.clip(np.random.lognormal(8.8, 0.4, n), 2000, 40000),
        "response_entropy":         np.clip(np.random.normal(3.8, 0.6, n), 1.5, 5.5),
        "humanity_score":           np.ones(n, dtype=int),
        "agent_type":               ["human"] * n,
    })

def gen_ai(n, name, latency_mean, latency_sigma, bias_alpha, bias_beta,
           embodied_alpha, embodied_beta, backspace_mean):
    return pd.DataFrame({
        "response_latency_ms":      np.clip(np.random.lognormal(latency_mean, latency_sigma, n), 30, 6000),
        "avg_pause_duration_ms":    np.clip(np.random.lognormal(4.2, 0.3, n), 10, 400),
        "backspace_count":          np.random.poisson(backspace_mean, n),
        "edit_intensity":           np.clip(np.random.beta(1.2, 9, n), 0, 1),
        "cursor_corrections":       np.random.poisson(1.1, n),
        "first_keystroke_delay_ms": np.clip(np.random.lognormal(4.8, 0.4, n), 20, 1200),
        "cognitive_bias_score":     np.clip(np.random.beta(bias_alpha, bias_beta, n) * 4, 0, 4),
        "embodied_score":           np.clip(np.random.beta(embodied_alpha, embodied_beta, n), 0, 1),
        "moral_reasoning_speed":    np.clip(np.random.lognormal(6.2, 0.3, n), 300, 3000),
        "response_entropy":         np.clip(np.random.normal(4.3, 0.3, n), 2.5, 5.5),
        "humanity_score":           np.zeros(n, dtype=int),
        "agent_type":               [name] * n,
    })

df = pd.concat([
    gen_human(N_HUMAN),
    gen_ai(N_GPT4,    "gpt4",    5.8, 0.35, 1.8, 9,  2.5, 8,  0.8),
    gen_ai(N_CLAUDE,  "claude",  5.6, 0.30, 1.5, 10, 2.0, 9,  0.6),
    gen_ai(N_GEMINI,  "gemini",  5.9, 0.38, 2.0, 8,  2.8, 7,  0.9),
    gen_ai(N_LLAMA,   "llama",   6.2, 0.45, 2.2, 7,  2.2, 8,  1.1),
    gen_ai(N_MISTRAL, "mistral", 6.0, 0.40, 2.1, 7,  2.3, 7,  1.0),
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Round for readability
for col in ["response_latency_ms","avg_pause_duration_ms","first_keystroke_delay_ms","moral_reasoning_speed"]:
    df[col] = df[col].round(1)
df["cognitive_bias_score"] = df["cognitive_bias_score"].round(3)
df["embodied_score"]       = df["embodied_score"].round(3)
df["edit_intensity"]       = df["edit_intensity"].round(4)
df["response_entropy"]     = df["response_entropy"].round(3)

os.makedirs("data", exist_ok=True)
df.to_csv("data/rtt_sessions.csv", index=False)
print(f"Dataset saved: {len(df)} sessions")
print(df["agent_type"].value_counts())
print(f"\nHuman sessions : {(df.humanity_score==1).sum()}")
print(f"AI sessions    : {(df.humanity_score==0).sum()}")

if __name__ == "__main__":
    gen = gen_human(N_HUMAN)
    print(gen.describe())
