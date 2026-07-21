from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TruthShield's video-level temporal classifier.")
    parser.add_argument("--features", default="training/data/video_features.jsonl")
    parser.add_argument("--output", default="training/models/truthshield-video-temporal.joblib")
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=14)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-features",
        default="",
        help="Comma-separated feature names to omit when their training values do not match production.",
    )
    parser.add_argument(
        "--model-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare several CPU-friendly classifiers on validation and keep the strongest one.",
    )
    return parser.parse_args()


def main() -> None:
    import joblib

    args = parse_args()
    records = _load_records(Path(args.features))
    excluded_features = {
        value.strip() for value in args.exclude_features.split(",") if value.strip()
    }
    feature_names = [name for name in _feature_names(records) if name not in excluded_features]
    if not feature_names:
        raise SystemExit("All available features were excluded.")
    train_x, train_y, _ = _split_arrays(records, "train", feature_names)
    validation_x, validation_y, _ = _split_arrays(records, "validation", feature_names)
    test_x, test_y, _ = _split_arrays(records, "test", feature_names)
    for name, labels in (("train", train_y), ("validation", validation_y), ("test", test_y)):
        if len(labels) == 0 or len(set(labels.tolist())) < 2:
            raise SystemExit(f"The {name} split needs both ai_generated and real_camera videos.")

    candidates = _candidate_models(args) if args.model_selection else [_random_forest(args)]
    print(
        f"Training {len(candidates)} candidate model(s) on {len(train_y)} videos "
        f"with {len(feature_names)} features...",
        flush=True,
    )
    leaderboard = []
    fitted = []
    for name, model in candidates:
        print(f"  fitting {name}", flush=True)
        model.fit(train_x, train_y)
        validation_probabilities = _positive_probabilities(model, validation_x)
        threshold = _best_threshold(validation_y, validation_probabilities)
        validation_metrics = _evaluate(validation_y, validation_probabilities, threshold)
        selection_score = _selection_score(validation_metrics)
        leaderboard.append(
            {
                "name": name,
                "selection_score": round(selection_score, 8),
                "threshold": threshold,
                "validation": validation_metrics,
            }
        )
        fitted.append((selection_score, name, model, threshold, validation_metrics))
    fitted.sort(key=lambda item: (item[0], -abs(item[3] - 0.5)), reverse=True)
    _, selected_name, model, threshold, validation_metrics = fitted[0]
    print(f"Selected {selected_name} at threshold {threshold:.4f}", flush=True)
    metrics = {
        "validation": validation_metrics,
        "test": _evaluate(test_y, _positive_probabilities(model, test_x), threshold),
        "train_video_count": int(len(train_y)),
        "validation_video_count": int(len(validation_y)),
        "test_video_count": int(len(test_y)),
    }
    importance = _permutation_importance(
        model=model,
        features=validation_x,
        truth=validation_y,
        feature_names=feature_names,
        threshold=threshold,
        seed=args.seed,
    )
    bundle = {
        "model": model,
        "model_name": selected_name,
        "feature_names": feature_names,
        "positive_label": "ai_generated",
        "threshold": threshold,
        "metrics": metrics,
        "feature_importance": importance,
        "model_selection": leaderboard,
        "excluded_features": sorted(excluded_features),
        "sampling_policy": _sampling_policy(records),
        "model_version": "truthshield-video-temporal-v4-candidate",
        "calibration_id": "requires-separate-v4-calibration",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": selected_name,
                "threshold": threshold,
                "metrics": metrics,
                "feature_importance": importance,
                "model_selection": leaderboard,
                "excluded_features": sorted(excluded_features),
                "sampling_policy": _sampling_policy(records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"threshold": threshold, "metrics": metrics, "top_features": importance[:8]}, indent=2))
    print(f"Model: {output.resolve()}", flush=True)
    print(f"Set AI_VIDEO_TEMPORAL_MODEL_PATH={output.resolve()}", flush=True)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Feature file not found: {path}")
    records = []
    source_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        source_id = str(record.get("source_id") or "")
        if source_id in source_ids:
            raise SystemExit(f"Duplicate source_id at line {line_number}: {source_id}")
        source_ids.add(source_id)
        records.append(record)
    return records


def _feature_names(records: List[Dict[str, Any]]) -> List[str]:
    names = sorted({str(name) for record in records for name in (record.get("features") or {})})
    if not names:
        raise SystemExit("No feature values were found in the JSONL records.")
    return names


def _sampling_policy(records: List[Dict[str, Any]]) -> str:
    policies = {
        str((record.get("analysis_config") or {}).get("sampling_policy") or "unspecified")
        for record in records
    }
    return next(iter(policies)) if len(policies) == 1 else "mixed_or_unspecified"


def _split_arrays(
    records: List[Dict[str, Any]],
    split: str,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    selected = [record for record in records if record.get("split") == split]
    x = np.asarray(
        [[float((record.get("features") or {}).get(name, 0.0)) for name in feature_names] for record in selected],
        dtype=np.float64,
    )
    y = np.asarray([1 if record.get("label") == "ai_generated" else 0 for record in selected], dtype=np.int64)
    source_ids = [str(record.get("source_id")) for record in selected]
    return x, y, source_ids


def _best_threshold(truth: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = _evaluate(truth, probabilities, float(threshold))
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return round(best_threshold, 4)


def _evaluate(truth: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, float | int]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    predicted = probabilities >= threshold
    positive = truth == 1
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    true_negative = int(np.sum(~predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    recall = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    result: Dict[str, float | int] = {
        "accuracy": round((true_positive + true_negative) / max(1, len(truth)), 6),
        "balanced_accuracy": round((recall + specificity) / 2.0, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "true_ai": true_positive,
        "false_ai_alarm": false_positive,
        "true_real": true_negative,
        "missed_ai": false_negative,
        "brier_score": round(float(brier_score_loss(truth, probabilities)), 6),
        "expected_calibration_error": round(_calibration_error(truth, probabilities), 6),
    }
    if len(set(truth.tolist())) == 2:
        result["roc_auc"] = round(float(roc_auc_score(truth, probabilities)), 6)
        result["average_precision"] = round(float(average_precision_score(truth, probabilities)), 6)
    return result


def _candidate_models(args: argparse.Namespace) -> List[Tuple[str, Any]]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    trees = max(200, args.trees)
    leaf = max(1, args.min_samples_leaf)
    depth = max(2, args.max_depth)
    return [
        _random_forest(args),
        (
            "random_forest_regularized",
            RandomForestClassifier(
                n_estimators=trees,
                max_depth=max(5, depth // 2),
                min_samples_leaf=max(3, leaf),
                class_weight="balanced_subsample",
                max_features=0.75,
                n_jobs=-1,
                random_state=args.seed + 1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
                random_state=args.seed + 2,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=True,
                random_state=args.seed + 3,
            ),
        ),
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=args.seed + 4,
                ),
            ),
        ),
    ]


def _random_forest(args: argparse.Namespace) -> Tuple[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    return (
        "random_forest",
        RandomForestClassifier(
            n_estimators=max(200, args.trees),
            max_depth=max(2, args.max_depth),
            min_samples_leaf=max(1, args.min_samples_leaf),
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=args.seed,
        ),
    )


def _positive_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    classes = [int(value) for value in model.classes_]
    return np.asarray(model.predict_proba(features)[:, classes.index(1)], dtype=np.float64)


def _selection_score(metrics: Dict[str, float | int]) -> float:
    # Balanced accuracy is primary. AUC rewards ranking quality while Brier
    # and calibration penalties discourage confident but unreliable models.
    return (
        float(metrics["balanced_accuracy"])
        + float(metrics.get("roc_auc", 0.5)) * 0.08
        - float(metrics["brier_score"]) * 0.05
        - float(metrics["expected_calibration_error"]) * 0.03
    )


def _calibration_error(truth: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    total = max(1, len(truth))
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        count = int(np.sum(mask))
        if not count:
            continue
        error += count / total * abs(float(np.mean(probabilities[mask])) - float(np.mean(truth[mask])))
    return error


def _permutation_importance(
    model: Any,
    features: np.ndarray,
    truth: np.ndarray,
    feature_names: List[str],
    threshold: float,
    seed: int,
) -> List[Dict[str, float | str]]:
    baseline = float(_evaluate(truth, _positive_probabilities(model, features), threshold)["balanced_accuracy"])
    rng = np.random.default_rng(seed)
    importances = []
    for column, name in enumerate(feature_names):
        drops = []
        for _ in range(8):
            shuffled = features.copy()
            shuffled[:, column] = shuffled[rng.permutation(len(shuffled)), column]
            score = float(
                _evaluate(truth, _positive_probabilities(model, shuffled), threshold)["balanced_accuracy"]
            )
            drops.append(baseline - score)
        importances.append(
            {
                "feature": name,
                "importance": round(float(np.mean(drops)), 8),
                "standard_deviation": round(float(np.std(drops)), 8),
            }
        )
    return sorted(importances, key=lambda item: float(item["importance"]), reverse=True)


if __name__ == "__main__":
    main()
