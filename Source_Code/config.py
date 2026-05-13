"""
config.py — RTT System Configuration
All values taken from the paper (Sections V, VI, VII).
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_PATH        = "data/rtt_sessions.csv"
RESULTS_DIR      = "results/"
MODEL_SAVE_PATH  = "results/rtt_model.pkl"
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

# ── Features (Section VI-A) ───────────────────────────────────────────────────
BEHAVIOURAL_FEATURES = [
    "response_latency_ms",       # Temporal Behavioural Analysis layer
    "avg_pause_duration_ms",
    "backspace_count",
    "edit_intensity",
    "cursor_corrections",
    "first_keystroke_delay_ms",
]

COGNITIVE_FEATURES = [
    "cognitive_bias_score",      # Cognitive Asymmetry Challenges layer
    "moral_reasoning_speed",
    "response_entropy",
]

EMBODIED_FEATURES = [
    "embodied_score",            # Embodied Knowledge Verification layer
]

ALL_FEATURES = BEHAVIOURAL_FEATURES + COGNITIVE_FEATURES + EMBODIED_FEATURES

# ── Humanity Score Thresholds (Section VI-B) ──────────────────────────────────
THRESHOLD_HIGH    = 0.65    # f(m) >= 0.65 → PASS (default)
THRESHOLD_REVIEW  = 0.35    # 0.35 <= f(m) < 0.65 → REVIEW
# f(m) < 0.35 → DENY

# ── Risk-Based Threshold Mapping (Section VI-B) ───────────────────────────────
RISK_THRESHOLDS = {
    "low":    0.50,    # browsing content
    "medium": 0.65,    # account login
    "high":   0.82,    # financial transactions / account modifications
}

# ── FDR Targets (Section VII-A) ───────────────────────────────────────────────
FDR_TARGET_HIGH_SECURITY  = 0.01   # financial applications
FDR_TARGET_LOW_STAKES     = 0.05   # general web

# ── Model Parameters ──────────────────────────────────────────────────────────
RF_N_ESTIMATORS  = 200
RF_MAX_DEPTH     = None
RF_MIN_SAMPLES   = 2

# ── Layer Weights for Ensemble Score ─────────────────────────────────────────
LAYER_WEIGHTS = {
    "embodied":   0.30,   # Embodied Knowledge Verification
    "cognitive":  0.40,   # Cognitive Asymmetry Challenges (highest — Section VIII-A)
    "behavioural":0.30,   # Temporal Behavioural Analysis
}

# ── UFI Sub-dimension Weights (Section VII-B) ─────────────────────────────────
UFI_WEIGHTS = {
    "completion_time":     0.30,
    "first_attempt_rate":  0.25,
    "accessibility_impact":0.30,
    "subjective_experience":0.15,
}

# ── AI Systems Tested (Section VII-A) ────────────────────────────────────────
AI_SYSTEMS = ["gpt4", "claude", "gemini", "llama", "mistral"]
