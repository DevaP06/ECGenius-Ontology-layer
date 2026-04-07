"""
ECGenius — evaluation/metrics.py
===================================
Evaluation metrics for both the AI model and the full
ontology-fused pipeline.

Two evaluation modes:
  1. Model-only  — standard multi-label classification metrics
                   on raw Pai(D) outputs
  2. Pipeline    — metrics on final FusionOutput confidence labels
                   (how well does the full CDSS perform end-to-end?)

Metrics computed:
  Per-label:
    AUC-ROC, AUC-PRC, F1, precision, recall, sensitivity, specificity

  Aggregate:
    macro/micro averages, Hamming loss, exact match ratio,
    ontology-weighted F1 (parent label partial credit)

  Triage-specific:
    Tier-1 sensitivity (critical — must be near 1.0)
    Tier-1 false-negative rate (clinical safety metric)

Usage:
    evaluator = PipelineEvaluator()
    report = evaluator.evaluate(predictions, ground_truth)
    evaluator.save_report(report, "evaluation/results.csv")
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        hamming_loss, multilabel_confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — some metrics unavailable.")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LabelMetrics:
    label_id: str
    auc_roc:     float = 0.0
    auc_prc:     float = 0.0
    f1:          float = 0.0
    precision:   float = 0.0
    recall:      float = 0.0
    sensitivity: float = 0.0   # = recall
    specificity: float = 0.0
    support:     int   = 0     # number of positive examples


@dataclass
class AggregateMetrics:
    macro_auc_roc:   float = 0.0
    micro_auc_roc:   float = 0.0
    macro_f1:        float = 0.0
    micro_f1:        float = 0.0
    hamming_loss:    float = 0.0
    exact_match:     float = 0.0
    tier1_sensitivity: float = 0.0   # clinical safety — must be high
    tier1_fnr:         float = 0.0   # false-negative rate for Tier-1


@dataclass
class EvaluationReport:
    per_label:   dict[str, LabelMetrics]
    aggregate:   AggregateMetrics
    n_samples:   int
    threshold:   float
    metadata:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "threshold": self.threshold,
            "aggregate": {
                "macro_auc_roc":     round(self.aggregate.macro_auc_roc, 4),
                "micro_auc_roc":     round(self.aggregate.micro_auc_roc, 4),
                "macro_f1":          round(self.aggregate.macro_f1, 4),
                "micro_f1":          round(self.aggregate.micro_f1, 4),
                "hamming_loss":      round(self.aggregate.hamming_loss, 4),
                "exact_match":       round(self.aggregate.exact_match, 4),
                "tier1_sensitivity": round(self.aggregate.tier1_sensitivity, 4),
                "tier1_fnr":         round(self.aggregate.tier1_fnr, 4),
            },
            "per_label": {
                lid: {
                    "auc_roc":     round(m.auc_roc, 4),
                    "auc_prc":     round(m.auc_prc, 4),
                    "f1":          round(m.f1, 4),
                    "precision":   round(m.precision, 4),
                    "recall":      round(m.recall, 4),
                    "specificity": round(m.specificity, 4),
                    "support":     m.support,
                }
                for lid, m in self.per_label.items()
            },
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class PipelineEvaluator:
    """
    Evaluate model-only or full pipeline predictions.

    Parameters
    ----------
    label_ids : list[str]
        Ordered list matching columns in predictions array.
    tier1_labels : list[str]
        Label IDs that are Tier-1 (life-threatening). Used for
        computing tier1_sensitivity safety metric.
    threshold : float
        Pai(D) threshold for converting probabilities to binary predictions.
    """

    def __init__(
        self,
        label_ids: list[str],
        tier1_labels: Optional[list[str]] = None,
        threshold: float = 0.50,
    ):
        self.label_ids    = label_ids
        self.tier1_labels = set(tier1_labels or [])
        self.threshold    = threshold

    def evaluate(
        self,
        y_prob: np.ndarray,    # shape (n_samples, n_labels) — sigmoid outputs
        y_true: np.ndarray,    # shape (n_samples, n_labels) — binary ground truth
    ) -> EvaluationReport:
        """
        Compute all metrics.

        Parameters
        ----------
        y_prob : np.ndarray
            Raw model probabilities or FusionOutput scores.
        y_true : np.ndarray
            Ground-truth binary labels.

        Returns
        -------
        EvaluationReport
        """
        assert y_prob.shape == y_true.shape, \
            f"Shape mismatch: y_prob={y_prob.shape} y_true={y_true.shape}"
        assert y_prob.shape[1] == len(self.label_ids), \
            f"Expected {len(self.label_ids)} labels, got {y_prob.shape[1]}"

        y_pred = (y_prob >= self.threshold).astype(int)
        n_samples = y_prob.shape[0]

        per_label = {}
        for i, lid in enumerate(self.label_ids):
            per_label[lid] = self._per_label_metrics(
                y_true[:, i], y_pred[:, i], y_prob[:, i], lid
            )

        aggregate = self._aggregate_metrics(y_true, y_pred, y_prob, per_label)

        return EvaluationReport(
            per_label=per_label,
            aggregate=aggregate,
            n_samples=n_samples,
            threshold=self.threshold,
            metadata={"label_ids": self.label_ids},
        )

    def _per_label_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        label_id: str,
    ) -> LabelMetrics:
        support = int(y_true.sum())

        if not SKLEARN_AVAILABLE:
            return LabelMetrics(label_id=label_id, support=support)

        # Handle edge case: no positive examples
        if support == 0:
            return LabelMetrics(label_id=label_id, support=0,
                                auc_roc=float("nan"), auc_prc=float("nan"))

        try:
            auc_roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc_roc = float("nan")

        try:
            auc_prc = float(average_precision_score(y_true, y_prob))
        except Exception:
            auc_prc = float("nan")

        f1  = float(f1_score(y_true, y_pred, zero_division=0))
        pre = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))

        # Specificity = TN / (TN + FP)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return LabelMetrics(
            label_id=label_id,
            auc_roc=auc_roc, auc_prc=auc_prc,
            f1=f1, precision=pre, recall=rec,
            sensitivity=rec, specificity=specificity,
            support=support,
        )

    def _aggregate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        per_label: dict[str, LabelMetrics],
    ) -> AggregateMetrics:
        if not SKLEARN_AVAILABLE:
            return AggregateMetrics()

        # Standard aggregate
        try:
            macro_auc = float(roc_auc_score(y_true, y_prob, average="macro"))
            micro_auc = float(roc_auc_score(y_true, y_prob, average="micro"))
        except Exception:
            macro_auc = micro_auc = float("nan")

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        h_loss   = float(hamming_loss(y_true, y_pred))
        exact    = float((y_pred == y_true).all(axis=1).mean())

        # Tier-1 safety metrics
        tier1_idx = [
            i for i, lid in enumerate(self.label_ids)
            if lid in self.tier1_labels
        ]
        tier1_sensitivity = tier1_fnr = float("nan")

        if tier1_idx:
            t1_true = y_true[:, tier1_idx]
            t1_pred = y_pred[:, tier1_idx]

            tp = ((t1_true == 1) & (t1_pred == 1)).sum()
            fn = ((t1_true == 1) & (t1_pred == 0)).sum()

            tier1_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            tier1_fnr         = fn / (tp + fn) if (tp + fn) > 0 else float("nan")

            if not np.isnan(tier1_fnr) and tier1_fnr > 0.05:
                logger.warning(
                    "SAFETY WARNING: Tier-1 false-negative rate = %.3f "
                    "(threshold 0.05). Review model + rules urgently.",
                    tier1_fnr,
                )

        return AggregateMetrics(
            macro_auc_roc=macro_auc,
            micro_auc_roc=micro_auc,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            hamming_loss=h_loss,
            exact_match=exact,
            tier1_sensitivity=tier1_sensitivity,
            tier1_fnr=tier1_fnr,
        )

    def save_report(self, report: EvaluationReport, output_path: str) -> None:
        """Save per-label metrics to CSV and full report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "label_id","auc_roc","auc_prc","f1",
                "precision","recall","specificity","support"
            ])
            writer.writeheader()
            for lid, m in report.per_label.items():
                writer.writerow({
                    "label_id": lid,
                    "auc_roc":     round(m.auc_roc, 4) if not np.isnan(m.auc_roc) else "nan",
                    "auc_prc":     round(m.auc_prc, 4) if not np.isnan(m.auc_prc) else "nan",
                    "f1":          round(m.f1, 4),
                    "precision":   round(m.precision, 4),
                    "recall":      round(m.recall, 4),
                    "specificity": round(m.specificity, 4),
                    "support":     m.support,
                })

        # JSON
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            f.write(report.to_json())

        logger.info("Report saved → %s  %s", csv_path, json_path)
        print(f"Saved: {csv_path}\nSaved: {json_path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(0)
    label_ids = ["AF", "NSR", "STEMI", "VF", "LVH"]
    n = 200

    y_prob = rng.random((n, len(label_ids)))
    y_true = (rng.random((n, len(label_ids))) > 0.7).astype(int)

    evaluator = PipelineEvaluator(
        label_ids=label_ids,
        tier1_labels=["STEMI", "VF"],
        threshold=0.5,
    )
    report = evaluator.evaluate(y_prob, y_true)

    print("\n=== Aggregate metrics ===")
    agg = report.aggregate
    print(f"  Macro AUC-ROC:     {agg.macro_auc_roc:.4f}")
    print(f"  Micro F1:          {agg.micro_f1:.4f}")
    print(f"  Hamming loss:      {agg.hamming_loss:.4f}")
    print(f"  Exact match:       {agg.exact_match:.4f}")
    print(f"  Tier-1 sensitivity:{agg.tier1_sensitivity:.4f}")
    print(f"  Tier-1 FNR:        {agg.tier1_fnr:.4f}")

    print("\n=== Per-label ===")
    for lid, m in report.per_label.items():
        print(f"  {lid:8s}  AUC={m.auc_roc:.3f}  F1={m.f1:.3f}  support={m.support}")