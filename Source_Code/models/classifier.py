"""
models/classifier.py — RTT Humanity Scoring Model
Random Forest classifier trained on behavioural + cognitive + embodied features.
Outputs a continuous Humanity Score ∈ [0, 1] (Section VI-A).
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import config


class HumanityScorer:
    """
    Trained classifier that outputs Humanity Score ∈ [0, 1].
    Score represents estimated probability that a session was produced by a human.

    Architecture (Section VI-A):
        - Features: behavioural (6) + cognitive (3) + embodied (1) = 10 features
        - Model: Random Forest (n_estimators=200)
        - Output: probability score from predict_proba
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ))
        ])
        self.feature_names = config.ALL_FEATURES
        self.is_fitted = False

    def fit(self, X, y):
        """Train the scorer on labelled session data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_score(self, X):
        """
        Return Humanity Score ∈ [0, 1] for each session.
        Score = P(human | features).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def classify(self, X, threshold=None):
        """
        Apply risk-based threshold to produce PASS / REVIEW / DENY decision.
        Returns dict with score, decision, and risk level.
        """
        if threshold is None:
            threshold = config.THRESHOLD_HIGH
        scores = self.predict_score(X)
        decisions = []
        for s in scores:
            if s >= threshold:
                decisions.append("PASS")
            elif s >= config.THRESHOLD_REVIEW:
                decisions.append("REVIEW")
            else:
                decisions.append("DENY")
        return scores, decisions

    def feature_importances(self):
        """Return feature importances from the Random Forest."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        rf = self.model.named_steps["clf"]
        return dict(zip(self.feature_names, rf.feature_importances_))

    def save(self, path=None):
        """Persist model to disk."""
        path = path or config.MODEL_SAVE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved → {path}")

    @staticmethod
    def load(path=None):
        """Load persisted model."""
        path = path or config.MODEL_SAVE_PATH
        with open(path, "rb") as f:
            return pickle.load(f)
