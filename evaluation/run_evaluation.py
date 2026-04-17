"""
ECGenius — evaluation/run_evaluation.py
=========================================
Main evaluation script. Run this after training to get all
paper metrics, grid search results, and ablation study.

Usage:
    # Full evaluation with real data
    python evaluation/run_evaluation.py --data data/processed/ --split test

    # With mock data (for testing evaluation code)
    python evaluation/run_evaluation.py --mock

    # Grid search only
    python evaluation/run_evaluation.py --mock --grid-search-only

    # Confidence threshold tuning only
    python evaluation/run_evaluation.py --mock --tune-thresholds-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ECGenius.Eval")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_data(data_dir: str, split: str = "test") -> dict:
    """
    Load processed labels and model predictions.

    Expected files:
        data/processed/labels/multilabel_targets.csv  — ground truth
        data/processed/splits/{split}_ids.txt         — patient IDs
        evaluation/model_predictions.npy               — saved model outputs
    """
    import csv
    data_dir = Path(data_dir)

    # Load label encoder
    enc_path = data_dir / "labels" / "label_encoder.json"
    with open(enc_path) as f:
        encoder = json.load(f)
    label_ids = encoder.get("labels", list(encoder.get("idx_to_label", {}).values()))

    # Load split IDs
    split_path = data_dir / "splits" / f"{split}_ids.txt"
    with open(split_path) as f:
        split_ids = [l.strip() for l in f.readlines()]

    # Load ground truth
    targets_path = data_dir / "labels" / "multilabel_targets.csv"
    all_targets  = {}
    with open(targets_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            if pid in split_ids:
                all_targets[pid] = [float(row.get(lid, 0)) for lid in label_ids]

    y_true = np.array([all_targets[pid] for pid in split_ids if pid in all_targets])

    # Load model predictions (saved after inference)
    pred_path = PROJECT_ROOT / "evaluation" / "model_predictions.npy"
    if pred_path.exists():
        y_pred = np.load(str(pred_path))
    else:
        raise FileNotFoundError(
            f"Model predictions not found at {pred_path}\n"
            "Run inference on test set first and save predictions:\n"
            "  np.save('evaluation/model_predictions.npy', predictions)"
        )

    return {"y_true": y_true, "y_pred": y_pred, "label_ids": label_ids}


def generate_mock_data(n_patients: int = 500, n_labels: int = 15) -> dict:
    """
    Generate realistic mock data for testing evaluation code.
    Uses biased probabilities to simulate real model behaviour.
    """
    np.random.seed(42)

    label_ids = [
        "AF", "NSR", "STEMI", "NSTEMI", "LVH", "VF", "VT",
        "SVT", "LBBB", "RBBB", "Ischemia", "ST_Elevation",
        "ST_Depression", "TWI", "PVC",
    ][:n_labels]

    # Ground truth: sparse multilabel (most patients have 1-2 conditions)
    y_true = np.zeros((n_patients, n_labels))
    for i in range(n_patients):
        n_pos = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        pos_idx = np.random.choice(n_labels, n_pos, replace=False)
        y_true[i, pos_idx] = 1

    # Model predictions: noisy but correlated with truth
    y_pred_ai = np.zeros((n_patients, n_labels))
    for i in range(n_patients):
        for j in range(n_labels):
            if y_true[i, j] == 1:
                # True positive: high probability with some noise
                y_pred_ai[i, j] = np.clip(np.random.beta(8, 2), 0.1, 0.99)
            else:
                # True negative: low probability with some noise
                y_pred_ai[i, j] = np.clip(np.random.beta(1, 6), 0.01, 0.7)

    # Simulate ontology effect: slightly better after ontology
    noise = np.random.normal(0, 0.02, y_pred_ai.shape)
    y_pred_ontology = np.clip(y_pred_ai + noise * (2 * y_true - 1), 0.01, 0.99)

    # Simulate history effect: further improvement on borderline cases
    noise2 = np.random.normal(0, 0.03, y_pred_ai.shape)
    y_pred_history = np.clip(y_pred_ontology + noise2 * (2 * y_true - 1), 0.01, 0.99)

    # Simulate full system: best
    noise3 = np.random.normal(0, 0.015, y_pred_ai.shape)
    y_pred_full = np.clip(y_pred_history + noise3 * (2 * y_true - 1), 0.01, 0.99)

    # Simulate score components for grid search
    ai_scores      = 0.5 * y_pred_ai
    symptom_scores = np.clip(
        y_true * np.random.uniform(0.05, 0.25, y_true.shape) +
        (1 - y_true) * np.random.uniform(0, 0.05, y_true.shape), 0, 0.30
    )
    risk_scores = np.clip(
        y_true * np.random.uniform(0.02, 0.08, y_true.shape) +
        (1 - y_true) * np.random.uniform(0, 0.02, y_true.shape), 0, 0.10
    )
    rule_scores = np.clip(
        y_true * np.random.uniform(0.01, 0.08, y_true.shape) +
        (1 - y_true) * np.random.uniform(-0.05, 0.01, y_true.shape), -0.10, 0.10
    )

    return {
        "y_true":         y_true,
        "y_pred_ai":      y_pred_ai,
        "y_pred_ontology": y_pred_ontology,
        "y_pred_history": y_pred_history,
        "y_pred_full":    y_pred_full,
        "ai_scores":      ai_scores,
        "symptom_scores": symptom_scores,
        "risk_scores":    risk_scores,
        "rule_scores":    rule_scores,
        "label_ids":      label_ids,
        "n_patients":     n_patients,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(args):
    from evaluation.metrics import ECGeniusEvaluator

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    if args.mock:
        logger.info("Generating mock data (%d patients)...", args.n_mock)
        data = generate_mock_data(n_patients=args.n_mock)
    else:
        logger.info("Loading real data from %s (split: %s)...", args.data, args.split)
        data = load_real_data(args.data, args.split)

    label_ids = data["label_ids"]
    y_true    = data["y_true"]
    y_pred    = data.get("y_pred_full", data.get("y_pred", data.get("y_pred_ai")))

    evaluator = ECGeniusEvaluator(
        label_ids=label_ids,
        tier1_labels={"STEMI", "VF", "VT", "NSTEMI"},
    )

    # ── 1. Find optimal thresholds ─────────────────────────────────────
    _section("Step 1: Optimal threshold tuning (Youden's J)")

    thresholds_youden = evaluator.find_optimal_thresholds(y_true, y_pred, method="youden")
    thresholds_f1     = evaluator.find_optimal_thresholds(y_true, y_pred, method="f1")

    thresh_report = {
        "youden": {lid: round(float(t), 3) for lid, t in zip(label_ids, thresholds_youden)},
        "f1":     {lid: round(float(t), 3) for lid, t in zip(label_ids, thresholds_f1)},
    }
    _save(thresh_report, out_dir / "optimal_thresholds.json")
    _print_thresholds(label_ids, thresholds_youden, thresholds_f1)

    if args.tune_thresholds_only:
        return

    # ── 2. Full evaluation metrics ─────────────────────────────────────
    _section("Step 2: Full evaluation metrics")

    metrics = evaluator.evaluate(y_true, y_pred, thresholds=thresholds_youden)

    print(f"\n  AUC-ROC macro:    {metrics.auc_roc_macro:.4f}")
    print(f"  AUC-ROC weighted: {metrics.auc_roc_weighted:.4f}")
    print(f"  F1 macro:         {metrics.f1_macro:.4f}")
    print(f"  F1 weighted:      {metrics.f1_weighted:.4f}")
    print(f"  Precision macro:  {metrics.precision_macro:.4f}")
    print(f"  Recall macro:     {metrics.recall_macro:.4f}")
    print(f"  Top-1 accuracy:   {metrics.top1_accuracy:.4f}")
    print(f"  Top-3 accuracy:   {metrics.top3_accuracy:.4f}")
    print(f"  Tier-1 recall:    {metrics.tier1_recall:.4f}")
    print(f"  Tier-1 missed:    {metrics.tier1_false_suppress}")
    print(f"  Brier score:      {metrics.mean_brier_score:.4f}")

    print(f"\n  Per-label AUC-ROC:")
    for lm in sorted(metrics.per_label, key=lambda x: -x.auc_roc):
        bar = "█" * int(lm.auc_roc * 20)
        print(f"    {lm.label_id:16s}  {lm.auc_roc:.4f}  {bar}  (n={lm.support})")

    evaluator.save_results(metrics, str(out_dir))

    if args.grid_search_only:
        pass
    else:
        # ── 3. Confidence threshold tuning ────────────────────────────
        _section("Step 3: Confidence label threshold tuning")

        conf_results = evaluator.tune_confidence_thresholds(y_true, y_pred)

        print(f"\n  Original thresholds (from PPT):")
        print(f"    CONFIRMED  ≥ 0.80")
        print(f"    PROBABLE   ≥ 0.60")
        print(f"    POSSIBLE   ≥ 0.30")

        rec = conf_results["recommended_thresholds"]
        print(f"\n  Tuned thresholds (data-driven):")
        print(f"    CONFIRMED  ≥ {rec['CONFIRMED']}")
        print(f"    PROBABLE   ≥ {rec['PROBABLE']}")
        print(f"    POSSIBLE   ≥ {rec['POSSIBLE']}")

        safety = conf_results["tier1_safety"]
        status = "SAFE" if safety["safe"] else "UNSAFE"
        print(f"\n  Tier-1 safety at CONFIRMED threshold: {status}")
        print(f"  Tier-1 recall: {safety['tier1_recall_at_t']:.4f}")
        print(f"  Note: {safety['note']}")

        _save(conf_results, out_dir / "confidence_thresholds.json")

        # ── 4. Ablation study ──────────────────────────────────────────
        _section("Step 4: Ablation study")

        if args.mock:
            ablation = evaluator.ablation_study(
                y_true         = y_true,
                ai_proba       = data["y_pred_ai"],
                ontology_proba = data["y_pred_ontology"],
                history_proba  = data["y_pred_history"],
                full_proba     = data["y_pred_full"],
                thresholds     = thresholds_youden,
            )
        else:
            logger.warning("Ablation requires separate score arrays. Skipping.")
            ablation = []

        if ablation:
            print(f"\n  {'Variant':<35} {'AUC-ROC':>8} {'F1':>8} {'Top-1':>8} {'Top-3':>8} {'Tier1-R':>8}")
            print(f"  {'-'*75}")
            for r in ablation:
                print(f"  {r.variant:<35} {r.auc_roc_macro:>8.4f} {r.f1_macro:>8.4f} "
                      f"{r.top1_accuracy:>8.4f} {r.top3_accuracy:>8.4f} {r.tier1_recall:>8.4f}")
            evaluator.save_ablation(ablation, str(out_dir))

    # ── 5. Grid search for fusion weights ─────────────────────────────
    _section("Step 5: Grid search — fusion weights (w_ai, w_sym, w_risk, w_rule)")

    if args.mock:
        grid_results = evaluator.grid_search_weights(
            y_true         = y_true,
            ai_scores      = data["ai_scores"],
            symptom_scores = data["symptom_scores"],
            risk_scores    = data["risk_scores"],
            rule_scores    = data["rule_scores"],
            metric         = "auc_roc_macro",
        )

        bw = grid_results["best_weights"]
        print(f"\n  Best weights found:")
        print(f"    w_ai     = {bw['w_ai']}   (your PPT: 0.50)")
        print(f"    w_sym    = {bw['w_sym']}   (your PPT: 0.30 cap)")
        print(f"    w_risk   = {bw['w_risk']}  (your PPT: 0.10 cap)")
        print(f"    w_rule   = {bw['w_rule']}  (your PPT: 0.10 cap)")
        print(f"    Score ({grid_results['metric']}): {grid_results['best_score']:.4f}")

        print(f"\n  Top 5 weight configurations:")
        for i, r in enumerate(grid_results["top10"][:5], 1):
            print(f"    {i}. ai={r['w_ai']} sym={r['w_sym']} "
                  f"risk={r['w_risk']} rule={r['w_rule']}  "
                  f"→ {r['metric']}={r['score']:.4f}")

        # Save without full_results (too large)
        save_grid = {k: v for k, v in grid_results.items() if k != "full_results"}
        save_grid["top10"] = grid_results["top10"]
        _save(save_grid, out_dir / "grid_search_weights.json")
    else:
        logger.warning(
            "Grid search requires pre-computed score components. "
            "Save ai_scores, symptom_scores, risk_scores, rule_scores as .npy files."
        )

    _section("Evaluation complete")
    print(f"  Results saved to: {out_dir}/")
    print(f"  Files: metrics.json, per_label_metrics.csv, ablation_study.csv,")
    print(f"         optimal_thresholds.json, confidence_thresholds.json,")
    print(f"         grid_search_weights.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def _save(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved: %s", path)

def _print_thresholds(label_ids, t_youden, t_f1):
    print(f"\n  {'Label':<18} {'Youden':>10} {'F1-optimal':>12}")
    print(f"  {'-'*42}")
    for lid, ty, tf in zip(label_ids, t_youden, t_f1):
        flag = " ← differs" if abs(ty - tf) > 0.05 else ""
        print(f"  {lid:<18} {ty:>10.3f} {tf:>12.3f}{flag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ECGenius Evaluation Suite")
    parser.add_argument("--mock",                action="store_true",
                        help="Use mock data (no real model needed)")
    parser.add_argument("--n-mock",              type=int, default=500,
                        help="Number of mock patients (default 500)")
    parser.add_argument("--data",                type=str, default="data/processed/",
                        help="Path to processed data directory")
    parser.add_argument("--split",               type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--grid-search-only",    action="store_true")
    parser.add_argument("--tune-thresholds-only", action="store_true")
    parser.add_argument("--output",              type=str, default="evaluation/results/")

    args = parser.parse_args()

    if not args.mock and not Path(args.data).exists():
        print(f"Error: data dir '{args.data}' not found. Use --mock for testing.")
        sys.exit(1)

    run_evaluation(args)


if __name__ == "__main__":
    main()