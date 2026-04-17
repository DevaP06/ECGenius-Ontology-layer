"""
ECGenius — evaluation/metrics.py
==================================
All evaluation metrics for the paper.

Metrics computed:
  - AUC-ROC per label + macro average
  - F1, Precision, Recall per label + macro
  - Top-K DDx accuracy (top-1, top-3)
  - Triage accuracy (Tier-1 never missed)
  - Confidence label calibration
  - Cohen's Kappa (cardiologist agreement)

Usage:
    from evaluation.metrics import ECGeniusEvaluator
    evaluator = ECGeniusEvaluator()
    results = evaluator.evaluate(y_true, y_pred_proba, ddx_outputs)
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score,
        recall_score, confusion_matrix, cohen_kappa_score,
        average_precision_score, brier_score_loss,
    )
    from sklearn.calibration import calibration_curve
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    logger.warning("scikit-learn not installed. Run: pip install scikit-learn")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LabelMetrics:
    label_id:          str
    auc_roc:           float
    average_precision: float
    f1:                float
    precision:         float
    recall:            float
    specificity:       float
    threshold:         float    # optimal threshold from Youden's J
    support:           int      # number of positive samples

    def to_dict(self) -> dict:
        return {
            "label_id":          self.label_id,
            "auc_roc":           round(self.auc_roc, 4),
            "average_precision": round(self.average_precision, 4),
            "f1":                round(self.f1, 4),
            "precision":         round(self.precision, 4),
            "recall":            round(self.recall, 4),
            "specificity":       round(self.specificity, 4),
            "optimal_threshold": round(self.threshold, 4),
            "support":           self.support,
        }


@dataclass
class SystemMetrics:
    """Overall system-level metrics."""
    auc_roc_macro:        float
    auc_roc_weighted:     float
    f1_macro:             float
    f1_weighted:          float
    precision_macro:      float
    recall_macro:         float
    top1_accuracy:        float    # correct Dx in rank 1
    top3_accuracy:        float    # correct Dx in top 3
    tier1_recall:         float    # Tier-1 labels never missed
    tier1_false_suppress: int      # how many Tier-1 got suppressed incorrectly
    mean_brier_score:     float    # probability calibration
    per_label:            list[LabelMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "auc_roc_macro":        round(self.auc_roc_macro, 4),
            "auc_roc_weighted":     round(self.auc_roc_weighted, 4),
            "f1_macro":             round(self.f1_macro, 4),
            "f1_weighted":          round(self.f1_weighted, 4),
            "precision_macro":      round(self.precision_macro, 4),
            "recall_macro":         round(self.recall_macro, 4),
            "top1_accuracy":        round(self.top1_accuracy, 4),
            "top3_accuracy":        round(self.top3_accuracy, 4),
            "tier1_recall":         round(self.tier1_recall, 4),
            "tier1_false_suppress": self.tier1_false_suppress,
            "mean_brier_score":     round(self.mean_brier_score, 4),
            "per_label":            [l.to_dict() for l in self.per_label],
        }


@dataclass
class AblationResult:
    """One row in the ablation study table."""
    variant:          str     # "AI only" / "AI + Ontology" / etc.
    auc_roc_macro:    float
    f1_macro:         float
    top1_accuracy:    float
    top3_accuracy:    float
    tier1_recall:     float
    description:      str

    def to_dict(self) -> dict:
        return {
            "variant":       self.variant,
            "auc_roc_macro": round(self.auc_roc_macro, 4),
            "f1_macro":      round(self.f1_macro, 4),
            "top1_accuracy": round(self.top1_accuracy, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
            "tier1_recall":  round(self.tier1_recall, 4),
            "description":   self.description,
        }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class ECGeniusEvaluator:
    """
    Compute all metrics for the ECGenius paper.

    Parameters
    ----------
    label_ids : list[str]
        Ordered list matching columns in y_true / y_pred
    tier1_labels : set[str]
        Labels that are Tier-1 (STEMI, VF, VT, NSTEMI)
        These get special safety metrics.
    """

    def __init__(
        self,
        label_ids:    list[str] = None,
        tier1_labels: set[str]  = None,
    ):
        if not _SKLEARN:
            raise RuntimeError("scikit-learn required. pip install scikit-learn")

        self.label_ids    = label_ids or []
        self.tier1_labels = tier1_labels or {"STEMI", "VF", "VT", "NSTEMI"}

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true:      np.ndarray,    # (N, L) binary ground truth
        y_pred_proba: np.ndarray,   # (N, L) model probabilities
        thresholds:  np.ndarray = None,  # (L,) per-label thresholds
        ddx_outputs: list = None,   # list of pipeline FusionOutput for top-K
    ) -> SystemMetrics:
        """
        Full evaluation.

        Parameters
        ----------
        y_true : np.ndarray shape (N, L)
            Binary ground truth. N = patients, L = labels.
        y_pred_proba : np.ndarray shape (N, L)
            Raw model probabilities per label.
        thresholds : np.ndarray shape (L,) optional
            Per-label decision thresholds. If None, uses Youden's J on val set.
        ddx_outputs : list optional
            Pipeline outputs for top-K DDx accuracy computation.
        """
        N, L = y_true.shape
        assert len(self.label_ids) == L, \
            f"label_ids length {len(self.label_ids)} != y_true columns {L}"

        # Find optimal thresholds if not provided
        if thresholds is None:
            thresholds = self.find_optimal_thresholds(y_true, y_pred_proba)

        y_pred_binary = (y_pred_proba >= thresholds).astype(int)

        # Per-label metrics
        per_label = []
        for i, lid in enumerate(self.label_ids):
            per_label.append(self._label_metrics(
                lid, y_true[:, i], y_pred_proba[:, i],
                y_pred_binary[:, i], thresholds[i],
            ))

        # Macro / weighted aggregates
        auc_macro    = roc_auc_score(y_true, y_pred_proba, average="macro")
        auc_weighted = roc_auc_score(y_true, y_pred_proba, average="weighted")
        f1_macro     = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
        f1_weighted  = f1_score(y_true, y_pred_binary, average="weighted", zero_division=0)
        prec_macro   = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
        rec_macro    = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)

        # Brier score (calibration)
        brier = float(np.mean([
            brier_score_loss(y_true[:, i], y_pred_proba[:, i])
            for i in range(L)
        ]))

        # Top-K DDx accuracy
        top1, top3 = self._topk_accuracy(y_true, y_pred_proba, thresholds)

        # Tier-1 safety
        tier1_recall, tier1_false_suppress = self._tier1_safety(
            y_true, y_pred_binary
        )

        return SystemMetrics(
            auc_roc_macro=auc_macro,
            auc_roc_weighted=auc_weighted,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            precision_macro=prec_macro,
            recall_macro=rec_macro,
            top1_accuracy=top1,
            top3_accuracy=top3,
            tier1_recall=tier1_recall,
            tier1_false_suppress=tier1_false_suppress,
            mean_brier_score=brier,
            per_label=per_label,
        )

    # ------------------------------------------------------------------
    # Threshold tuning (grid search)
    # ------------------------------------------------------------------

    def find_optimal_thresholds(
        self,
        y_true:      np.ndarray,
        y_pred_proba: np.ndarray,
        method:      str = "youden",
    ) -> np.ndarray:
        """
        Find optimal per-label thresholds.

        Methods:
          youden   — maximise TPR - FPR (Youden's J statistic)
          f1       — maximise F1 score
          balanced — balance precision and recall

        Returns np.ndarray shape (L,)
        """
        L = y_true.shape[1]
        thresholds = np.zeros(L)

        for i in range(L):
            yt = y_true[:, i]
            yp = y_pred_proba[:, i]

            if yt.sum() == 0:
                thresholds[i] = 0.5
                continue

            if method == "youden":
                thresholds[i] = self._youden_threshold(yt, yp)
            elif method == "f1":
                thresholds[i] = self._f1_threshold(yt, yp)
            else:
                thresholds[i] = 0.5

        logger.info("Optimal thresholds found: %s",
                    {self.label_ids[i]: round(thresholds[i], 3) for i in range(L)})
        return thresholds

    @staticmethod
    def _youden_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Youden's J = Sensitivity + Specificity - 1 = TPR - FPR."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    @staticmethod
    def _f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find threshold that maximises F1."""
        candidates = np.linspace(0.05, 0.95, 50)
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t

    # ------------------------------------------------------------------
    # Fusion weight grid search
    # ------------------------------------------------------------------

    def grid_search_weights(
        self,
        y_true:            np.ndarray,
        ai_scores:         np.ndarray,  # (N, L) — 0.5 × Pai(D)
        symptom_scores:    np.ndarray,  # (N, L) — S_symptom
        risk_scores:       np.ndarray,  # (N, L) — S_risk
        rule_scores:       np.ndarray,  # (N, L) — S_rule
        metric:            str = "auc_roc_macro",
    ) -> dict:
        """
        Grid search over fusion weight combinations.
        Finds best (w_ai, w_symptom, w_risk, w_rule) within caps.

        Returns dict with best weights and full grid results.
        """
        results = []

        # Grid: w_ai ∈ [0.3, 0.7], others share remaining weight
        w_ai_range  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        w_sym_range = [0.15, 0.20, 0.25, 0.30]
        w_risk_range = [0.05, 0.08, 0.10, 0.12]

        best_score  = -1.0
        best_config = {}

        total = len(w_ai_range) * len(w_sym_range) * len(w_risk_range)
        done  = 0

        for w_ai in w_ai_range:
            for w_sym in w_sym_range:
                for w_risk in w_risk_range:
                    w_rule = max(0.0, 1.0 - w_ai - w_sym - w_risk)
                    if w_rule < 0 or w_rule > 0.20:
                        continue

                    # Compute final scores
                    final = (
                        w_ai   * ai_scores +
                        w_sym  * symptom_scores +
                        w_risk * risk_scores +
                        w_rule * rule_scores
                    )
                    final = np.clip(final, 0, 1)

                    try:
                        if metric == "auc_roc_macro":
                            score = roc_auc_score(y_true, final, average="macro")
                        elif metric == "f1_macro":
                            thresh = self.find_optimal_thresholds(y_true, final)
                            pred   = (final >= thresh).astype(int)
                            score  = f1_score(y_true, pred, average="macro", zero_division=0)
                        else:
                            score = roc_auc_score(y_true, final, average="macro")

                        results.append({
                            "w_ai":    w_ai,
                            "w_sym":   w_sym,
                            "w_risk":  w_risk,
                            "w_rule":  round(w_rule, 3),
                            "score":   round(score, 4),
                            "metric":  metric,
                        })

                        if score > best_score:
                            best_score  = score
                            best_config = {
                                "w_ai":   w_ai,
                                "w_sym":  w_sym,
                                "w_risk": w_risk,
                                "w_rule": round(w_rule, 3),
                            }
                    except Exception as e:
                        logger.debug("Grid search error at (%s,%s,%s): %s",
                                     w_ai, w_sym, w_risk, e)

                    done += 1

        # Sort by score desc
        results.sort(key=lambda x: -x["score"])

        logger.info(
            "Grid search complete — best %s=%.4f with weights %s",
            metric, best_score, best_config,
        )

        return {
            "best_weights": best_config,
            "best_score":   round(best_score, 4),
            "metric":       metric,
            "top10":        results[:10],
            "full_results": results,
        }

    # ------------------------------------------------------------------
    # Confidence threshold tuning
    # ------------------------------------------------------------------

    def tune_confidence_thresholds(
        self,
        y_true:      np.ndarray,
        y_pred_proba: np.ndarray,
        target_precision_confirmed: float = 0.90,
        target_recall_tier1:        float = 0.99,
    ) -> dict:
        """
        Find optimal confidence thresholds for CONFIRMED/PROBABLE/POSSIBLE/INCIDENTAL.

        Strategy:
          CONFIRMED  — set so precision >= target (avoid over-confident wrong calls)
          PROBABLE   — set so F1 is maximised
          POSSIBLE   — set so recall >= 0.80 (catch most positives)
          INCIDENTAL — everything below POSSIBLE

        Returns dict with recommended thresholds + calibration analysis.
        """
        # Get macro-averaged probability for each sample's top prediction
        top_probs = y_pred_proba.max(axis=1)

        # For each threshold candidate, compute precision on that subset
        candidates = np.arange(0.40, 0.95, 0.02)
        calibration = []

        for t in candidates:
            mask = top_probs >= t
            if mask.sum() < 5:
                continue
            # Among samples where we're >= t confident,
            # how often is the top prediction actually correct?
            top_pred_idx  = y_pred_proba.argmax(axis=1)
            top_true      = y_true[np.arange(len(y_true)), top_pred_idx]
            precision_at_t = top_true[mask].mean() if mask.sum() > 0 else 0.0

            calibration.append({
                "threshold":   round(float(t), 2),
                "precision":   round(float(precision_at_t), 4),
                "coverage":    round(float(mask.mean()), 4),
                "n_samples":   int(mask.sum()),
            })

        # Find CONFIRMED threshold: first t where precision >= target
        confirmed_t  = 0.80  # default from PPT
        probable_t   = 0.60
        possible_t   = 0.30

        for entry in calibration:
            if entry["precision"] >= target_precision_confirmed:
                confirmed_t = entry["threshold"]
                break

        # Find PROBABLE: where precision >= 0.75
        for entry in calibration:
            if entry["precision"] >= 0.75:
                probable_t = entry["threshold"]
                break

        # Find POSSIBLE: where precision >= 0.50
        for entry in calibration:
            if entry["precision"] >= 0.50:
                possible_t = entry["threshold"]
                break

        # Tier-1 safety check at these thresholds
        tier1_safety = self._tier1_threshold_safety(
            y_true, y_pred_proba, confirmed_t, target_recall_tier1
        )

        return {
            "recommended_thresholds": {
                "CONFIRMED":  round(confirmed_t, 2),
                "PROBABLE":   round(probable_t, 2),
                "POSSIBLE":   round(possible_t, 2),
                "INCIDENTAL": 0.00,
            },
            "original_thresholds": {
                "CONFIRMED": 0.80, "PROBABLE": 0.60,
                "POSSIBLE": 0.30,  "INCIDENTAL": 0.00,
            },
            "calibration_curve":   calibration,
            "tier1_safety":        tier1_safety,
            "note": (
                "These thresholds are tuned on validation set. "
                "Always verify on held-out test set before clinical use."
            ),
        }

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    def ablation_study(
        self,
        y_true:         np.ndarray,
        ai_proba:       np.ndarray,   # raw Pai(D) from model
        ontology_proba: np.ndarray,   # Pai after ontology (mutual excl, hierarchy)
        history_proba:  np.ndarray,   # after history deltas added
        full_proba:     np.ndarray,   # full system with rules
        thresholds:     np.ndarray = None,
    ) -> list[AblationResult]:
        """
        Run 4-variant ablation study for the paper Table.

        Variant A: AI model alone
        Variant B: AI + Ontology mapping (hierarchy + mutual exclusion)
        Variant C: AI + Ontology + History (patient symptoms/risk)
        Variant D: Full system (A + B + C + Rules)

        Returns list[AblationResult] — one row per variant.
        """
        variants = [
            ("A: AI model only",            ai_proba,
             "Raw CNN-Transformer output, no ontology or history"),
            ("B: AI + Ontology",            ontology_proba,
             "Added ontology hierarchy, mutual exclusion, triage"),
            ("C: AI + Ontology + History",  history_proba,
             "Added patient symptoms, risk factors, vitals"),
            ("D: Full system (+ Rules)",    full_proba,
             "Full pipeline with cardiologist rule engine"),
        ]

        results = []
        for name, proba, desc in variants:
            t = thresholds if thresholds is not None else \
                self.find_optimal_thresholds(y_true, proba)

            pred  = (proba >= t).astype(int)
            top1, top3 = self._topk_accuracy(y_true, proba, t)
            tier1_recall, _ = self._tier1_safety(y_true, pred)

            try:
                auc = roc_auc_score(y_true, proba, average="macro")
            except Exception:
                auc = 0.0

            f1 = f1_score(y_true, pred, average="macro", zero_division=0)

            results.append(AblationResult(
                variant=name,
                auc_roc_macro=round(auc, 4),
                f1_macro=round(f1, 4),
                top1_accuracy=round(top1, 4),
                top3_accuracy=round(top3, 4),
                tier1_recall=round(tier1_recall, 4),
                description=desc,
            ))

        return results

    # ------------------------------------------------------------------
    # Cardiologist agreement (Cohen's Kappa)
    # ------------------------------------------------------------------

    def cardiologist_agreement(
        self,
        system_labels:       list[str],    # system's top-1 prediction per patient
        cardiologist_labels: list[str],    # cardiologist's diagnosis per patient
    ) -> dict:
        """
        Compute Cohen's Kappa between system and cardiologist.
        Used in paper as qualitative validation.
        """
        kappa = cohen_kappa_score(system_labels, cardiologist_labels)
        agreement_pct = np.mean(
            [s == c for s, c in zip(system_labels, cardiologist_labels)]
        )

        interpretation = (
            "Almost perfect (κ > 0.80)" if kappa > 0.80 else
            "Substantial (0.60–0.80)"   if kappa > 0.60 else
            "Moderate (0.40–0.60)"      if kappa > 0.40 else
            "Fair (0.20–0.40)"          if kappa > 0.20 else
            "Slight (< 0.20)"
        )

        return {
            "cohens_kappa":     round(kappa, 4),
            "agreement_pct":    round(agreement_pct, 4),
            "interpretation":   interpretation,
            "n_cases":          len(system_labels),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label_metrics(
        self,
        label_id:    str,
        y_true:      np.ndarray,
        y_proba:     np.ndarray,
        y_pred:      np.ndarray,
        threshold:   float,
    ) -> LabelMetrics:
        support = int(y_true.sum())

        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = 0.0

        try:
            ap = average_precision_score(y_true, y_proba)
        except Exception:
            ap = 0.0

        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

        # Specificity = TN / (TN + FP)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return LabelMetrics(
            label_id=label_id,
            auc_roc=auc,
            average_precision=ap,
            f1=f1,
            precision=prec,
            recall=rec,
            specificity=spec,
            threshold=threshold,
            support=support,
        )

    def _topk_accuracy(
        self,
        y_true:      np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds:  np.ndarray,
        k_values:    list[int] = [1, 3],
    ) -> tuple[float, float]:
        """
        Top-K DDx accuracy:
        For each patient, is the true diagnosis in the top-K predicted labels?
        """
        N = y_true.shape[0]
        top1_hits = 0
        top3_hits = 0

        for i in range(N):
            true_labels = set(np.where(y_true[i] == 1)[0])
            if not true_labels:
                continue

            ranked = np.argsort(y_pred_proba[i])[::-1]

            if any(j in true_labels for j in ranked[:1]):
                top1_hits += 1
            if any(j in true_labels for j in ranked[:3]):
                top3_hits += 1

        n_valid = sum(1 for i in range(N) if y_true[i].sum() > 0)
        top1 = top1_hits / n_valid if n_valid > 0 else 0.0
        top3 = top3_hits / n_valid if n_valid > 0 else 0.0
        return top1, top3

    def _tier1_safety(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
    ) -> tuple[float, int]:
        """
        Tier-1 recall: among patients who truly have a Tier-1 condition,
        how many did the system detect (before ontology suppression)?
        """
        tier1_indices = [
            i for i, lid in enumerate(self.label_ids)
            if lid in self.tier1_labels
        ]
        if not tier1_indices:
            return 1.0, 0

        y_true_t1 = y_true[:, tier1_indices]
        y_pred_t1 = y_pred[:, tier1_indices]

        # Any Tier-1 condition present
        has_tier1 = y_true_t1.any(axis=1)
        detected  = (y_pred_t1 * y_true_t1).any(axis=1)

        n_tier1   = int(has_tier1.sum())
        n_detected = int((has_tier1 & detected).sum())
        n_missed  = n_tier1 - n_detected

        recall = n_detected / n_tier1 if n_tier1 > 0 else 1.0
        return recall, n_missed

    def _tier1_threshold_safety(
        self,
        y_true:       np.ndarray,
        y_pred_proba: np.ndarray,
        confirmed_t:  float,
        target_recall: float,
    ) -> dict:
        tier1_indices = [
            i for i, lid in enumerate(self.label_ids)
            if lid in self.tier1_labels
        ]
        if not tier1_indices:
            return {"safe": True, "note": "No Tier-1 labels in label set"}

        tier1_proba  = y_pred_proba[:, tier1_indices]
        tier1_true   = y_true[:, tier1_indices]
        has_tier1    = tier1_true.any(axis=1)
        detected_at_t = (tier1_proba >= confirmed_t).any(axis=1)

        recall_at_t = (
            (has_tier1 & detected_at_t).sum() / has_tier1.sum()
            if has_tier1.sum() > 0 else 1.0
        )
        safe = bool(recall_at_t >= target_recall)

        return {
            "safe":              safe,
            "tier1_recall_at_t": round(float(recall_at_t), 4),
            "target_recall":     target_recall,
            "confirmed_threshold": confirmed_t,
            "note": (
                "SAFE — Tier-1 recall meets target" if safe
                else f"UNSAFE — lower CONFIRMED threshold to improve Tier-1 recall"
            ),
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self, metrics: SystemMetrics, output_dir: str = "evaluation/") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Per-label CSV for paper table
        import csv
        with open(out / "per_label_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "label_id", "auc_roc", "average_precision",
                "f1", "precision", "recall", "specificity",
                "optimal_threshold", "support",
            ])
            writer.writeheader()
            for lm in metrics.per_label:
                writer.writerow(lm.to_dict())

        logger.info("Results saved to %s", out)

    def save_ablation(self, results: list[AblationResult], output_dir: str = "evaluation/") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out / "ablation_study.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "variant", "auc_roc_macro", "f1_macro",
                "top1_accuracy", "top3_accuracy", "tier1_recall", "description",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        logger.info("Ablation study saved to %s", out)