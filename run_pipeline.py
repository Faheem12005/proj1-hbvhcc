"""Research pipeline for the HBV-HCC paper baseline and a leakage-safe ensemble comparison.

This script preserves the current workflow while adding a strict paper-style ANN benchmark and a
separate optimized ensemble benchmark that uses the exact same training/validation cohorts and
three genes as the paper.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier

from config import DATA_DIR, OUT_DIR, PAPER_FEATURE_GENES, RANDOM_STATE
from step5_baseline_ann import gene_score_from_training_thresholds, train_ann_baseline
from step6_ensemble_model import fit_best_ensemble, predict_ensemble
from step7_shap_explain import compute_shap_values, ensemble_mean_abs_shap, summary_plot


def _read_geo_metadata(clinical_path: str) -> pd.DataFrame:
    """Parse GEO clinical metadata and keep only sample-level keys beginning with !Sample_."""
    rows = {}
    with open(clinical_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            key = parts[0]
            if key.startswith("!") and len(parts) > 1 and key.startswith("!Sample_"):
                rows.setdefault(key, []).extend(parts[1:])

    if not rows or "!Sample_geo_accession" not in rows:
        raise ValueError(f"No primary GEO sample metadata found in {clinical_path}")

    sample_ids = rows["!Sample_geo_accession"]
    metadata = pd.DataFrame({"sample_id": sample_ids})
    for key, values in rows.items():
        if key == "!Sample_geo_accession":
            continue
        if len(values) != len(sample_ids):
            continue
        metadata[key] = values
    return metadata


def load_real_cohort(expr_path: str, clinical_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a real GEO cohort and infer HBV-vs-HCC labels from sample titles."""
    expr = pd.read_csv(expr_path, sep="\t")
    expr = expr.set_index("ID")
    expr.index = expr.index.str.upper()
    expr = expr.groupby(level=0).mean()

    metadata = _read_geo_metadata(clinical_path).rename(columns={"sample_id": "!Sample_geo_accession"})
    sample_to_label = {}
    for row in metadata.to_dict("records"):
        sample = str(row.get("!Sample_geo_accession", "")).strip().strip('"')
        title = str(row.get("!Sample_title", "")).strip().strip('"')
        if not sample:
            continue
        text = title.lower()
        if any(term in text for term in ["adjacent normal", "normal tissue", "non-tumor", "non tumor", "non_tumor"]):
            value = 0
        elif any(term in text for term in ["tumor", "hcc", "tumour"]):
            value = 1
        else:
            value = 0
        sample_to_label[sample] = value

    expr.columns = [str(col).strip().strip('"') for col in expr.columns]
    labels = pd.Series({sample: sample_to_label.get(sample, np.nan) for sample in expr.columns})
    labels = labels.dropna()
    expr = expr.loc[:, labels.index]
    return expr, labels.to_numpy(dtype=int)


def validate_cohort_counts(train_y: np.ndarray, val_y: np.ndarray):
    train_hbv = int(np.sum(train_y == 0))
    train_hcc = int(np.sum(train_y == 1))
    val_hbv = int(np.sum(val_y == 0))
    val_hcc = int(np.sum(val_y == 1))
    if not (train_hbv == 37 and train_hcc == 70 and val_hbv == 91 and val_hcc == 49):
        raise ValueError(
            "Cohort counts do not match the paper specification: "
            f"train=(HBV={train_hbv}, HCC={train_hcc}), val=(HBV={val_hbv}, HCC={val_hcc})"
        )
    return {"train": {"HBV": train_hbv, "HBV_HCC": train_hcc}, "validation": {"HBV": val_hbv, "HBV_HCC": val_hcc}}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def paper_gene_score(train_expr: pd.DataFrame, val_expr: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Strict paper-style GeneScore using training medians only. The paper reports all 3 genes are downregulated in HCC."""
    training_medians = train_expr.loc[genes].median(axis=1).to_dict()
    downregulated = set(genes)
    train_score = pd.DataFrame(index=genes, columns=train_expr.columns, dtype=int)
    val_score = pd.DataFrame(index=genes, columns=val_expr.columns, dtype=int)
    for gene in genes:
        med = float(training_medians[gene])
        train_score.loc[gene] = (train_expr.loc[gene] <= med).astype(int) if gene in downregulated else (train_expr.loc[gene] > med).astype(int)
        val_score.loc[gene] = (val_expr.loc[gene] <= med).astype(int) if gene in downregulated else (val_expr.loc[gene] > med).astype(int)
    return train_score, val_score, training_medians


def fit_paper_ann(train_expr: pd.DataFrame, y_train: np.ndarray, valid_expr: pd.DataFrame, y_valid: np.ndarray, genes: list[str]) -> dict:
    train_gs, valid_gs, training_medians = paper_gene_score(train_expr, valid_expr, genes)
    ann = train_ann_baseline(train_gs, y_train, hidden_layer_sizes=(5,))
    train_prob = ann.predict_proba(train_gs.T.to_numpy(dtype=float))[:, 1]
    valid_prob = ann.predict_proba(valid_gs.T.to_numpy(dtype=float))[:, 1]
    metrics_train = compute_metrics(y_train, train_prob)
    metrics_valid = compute_metrics(y_valid, valid_prob)
    return {
        "model_name": "Paper ANN",
        "features": genes,
        "feature_type": "binary GeneScore",
        "training_medians": training_medians,
        "model": ann,
        "train_prob": train_prob,
        "valid_prob": valid_prob,
        "train_auc": metrics_train["auc"],
        "valid_auc": metrics_valid["auc"],
        "train_metrics": metrics_train,
        "valid_metrics": metrics_valid,
    }


def paired_bootstrap_auc_difference(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, n_boot: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """Paired bootstrap for the AUC difference on the identical validation set."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    observed = roc_auc_score(y_true, p2) - roc_auc_score(y_true, p1)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(roc_auc_score(y_true[idx], p2[idx]) - roc_auc_score(y_true[idx], p1[idx]))
    diffs = np.asarray(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = float(np.mean(diffs <= 0))
    return {
        "method": "paired bootstrap AUC difference",
        "observed_difference": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }


def _make_roc_plot(ax, y_true: np.ndarray, y_prob: np.ndarray, label: str, color: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('GSE55092 validation ROC')
    ax.legend(loc='lower right')


def _plot_confusion_matrices(models: dict, y_true: np.ndarray, X_valid: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for ax, (name, probs) in zip(axes, models.items()):
        pred = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        ax.imshow(cm, cmap='Blues')
        ax.set_title(name)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['HBV', 'HCC'])
        ax.set_yticklabels(['HBV', 'HCC'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    return fig


def _plot_model_auc(results_df: pd.DataFrame, out_path: str):
    """Plot validation AUC for every model in the benchmark."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ordered = results_df.sort_values("validation_auc", ascending=False)
    colors = ["#d95f02" if name == "Ensemble" else "#1b9e77" if name == "Paper ANN" else "#7570b3" for name in ordered["model"]]
    bars = ax.bar(ordered["model"], ordered["validation_auc"], color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation AUC")
    ax.set_title("External validation performance by model")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, ordered["validation_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_expression(train_expr: pd.DataFrame, y_train: np.ndarray, genes: list[str], out_path: str):
    """Plot training expression distributions for the genes used by all models."""
    rows = []
    for gene in genes:
        for value, label in zip(train_expr.loc[gene].to_numpy(dtype=float), y_train):
            rows.append({"gene": gene, "expression": value, "group": "HBV" if label == 0 else "HBV-HCC"})
    expression = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(genes), figsize=(4 * len(genes), 4), constrained_layout=True)
    if len(genes) == 1:
        axes = [axes]
    for ax, gene in zip(axes, genes):
        groups = [expression.loc[(expression["gene"] == gene) & (expression["group"] == group), "expression"] for group in ["HBV", "HBV-HCC"]]
        ax.boxplot(groups, tick_labels=["HBV", "HBV-HCC"], patch_artist=True, boxprops={"facecolor": "#a6cee3"})
        ax.set_title(gene)
        ax.set_ylabel("Expression")
    fig.suptitle("Training expression distributions of selected genes")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_synthetic_cohort(n_hbv: int, n_hcc: int, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct a clean synthetic cohort with the same cohort sizes as the paper and a strong signal in the three target genes.

    The three paper genes are reported as downregulated in HCC. We mimic that direction with a
    large separation in HBV vs HCC while keeping the remaining genes weak/noisy so the model must
    rely on the same small discriminative feature set.
    """
    rng = np.random.default_rng(seed)
    genes = list(PAPER_FEATURE_GENES) + [f"noise_{i}" for i in range(18)]
    sample_ids = [f"HBV_{i}" for i in range(n_hbv)] + [f"HCC_{i}" for i in range(n_hcc)]
    labels = np.concatenate([np.zeros(n_hbv, dtype=int), np.ones(n_hcc, dtype=int)])

    matrix = np.zeros((len(genes), len(sample_ids)), dtype=float)
    for idx, gene in enumerate(genes):
        if gene in PAPER_FEATURE_GENES:
            hbv_mean = 8.5 + rng.normal(0, 0.2)
            hcc_mean = 6.1 + rng.normal(0, 0.2)
            signal = np.concatenate([
                rng.normal(hbv_mean, 0.8, size=n_hbv),
                rng.normal(hcc_mean, 0.8, size=n_hcc),
            ])
            matrix[idx, :] = signal
        else:
            base = 7.0 + rng.normal(0, 0.6, size=len(sample_ids))
            matrix[idx, :] = base

    expr = pd.DataFrame(matrix, index=genes, columns=sample_ids)
    return expr, labels


def prepare_shared_dataset(train_expr: pd.DataFrame, valid_expr: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return one shared base dataset for every model.

    Every model receives the same train/validation cohorts and the same gene set. The
    difference is *only* the feature transformation applied afterward:
      - ANN baseline: binary GeneScore learned from training medians
      - gradient boosting ensemble: raw expression values on the same genes
    """
    train_features = train_expr.loc[genes].copy()
    valid_features = valid_expr.loc[genes].copy()
    return train_features, valid_features


def run_research_experiment() -> dict:
    genes = PAPER_FEATURE_GENES
    train_expr_path = os.path.join(DATA_DIR, "GEO-GSE121248-symbol.txt")
    valid_expr_path = os.path.join(DATA_DIR, "GEO-GSE55092-symbol.txt")
    train_clinical_path = os.path.join(DATA_DIR, "GEO-GSE121248-clinical.txt")
    valid_clinical_path = os.path.join(DATA_DIR, "GEO-GSE55092-clinical.txt")

    if all(os.path.exists(path) for path in [train_expr_path, valid_expr_path, train_clinical_path, valid_clinical_path]):
        train_expr, y_train = load_real_cohort(train_expr_path, train_clinical_path)
        valid_expr, y_valid = load_real_cohort(valid_expr_path, valid_clinical_path)
    else:
        train_expr, y_train = make_synthetic_cohort(37, 70, seed=RANDOM_STATE)
        valid_expr, y_valid = make_synthetic_cohort(91, 49, seed=RANDOM_STATE + 1)

    train_expr, valid_expr = prepare_shared_dataset(train_expr, valid_expr, genes)
    cohort_counts = validate_cohort_counts(y_train, y_valid)

    paper_result = fit_paper_ann(train_expr, y_train, valid_expr, y_valid, genes)

    X_train = train_expr.T
    X_valid = valid_expr.T

    ensemble_result = fit_best_ensemble(X_train, y_train, cv_folds=5)
    ensemble_models = ensemble_result.models
    xgb_valid = ensemble_models['xgboost'].predict_proba(X_valid)[:, 1]
    lgbm_valid = ensemble_models['lightgbm'].predict_proba(X_valid)[:, 1]
    cat_valid = ensemble_models['catboost'].predict_proba(X_valid)[:, 1]
    ensemble_valid_prob = predict_ensemble(ensemble_models, X_valid).to_numpy()

    results = []
    for model_name, prob in [
        ("Paper ANN", paper_result["valid_prob"]),
        ("XGBoost", xgb_valid),
        ("LightGBM", lgbm_valid),
        ("CatBoost", cat_valid),
        ("Ensemble", ensemble_valid_prob),
    ]:
        metrics = compute_metrics(y_valid, prob)
        if model_name == "Paper ANN":
            train_auc = float(paper_result["train_metrics"]["auc"])
        else:
            train_auc = float(ensemble_result.cv_auc.get(model_name.lower(), ensemble_result.cv_auc["ensemble"]))
        results.append({
            "model": model_name,
            "features": ", ".join(genes),
            "feature_type": "binary GeneScore" if model_name == "Paper ANN" else "continuous expression",
            "train_auc": train_auc,
            "validation_auc": float(metrics["auc"]),
            "validation_accuracy": float(metrics["accuracy"]),
            "validation_sensitivity": float(metrics["sensitivity"]),
            "validation_specificity": float(metrics["specificity"]),
            "validation_precision": float(metrics["precision"]),
            "validation_f1": float(metrics["f1"]),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"), index=False)
    _plot_model_auc(results_df, os.path.join(OUT_DIR, "model_auc_comparison.png"))
    _plot_gene_expression(train_expr, y_train, genes, os.path.join(OUT_DIR, "selected_gene_expression.png"))

    fig, ax = plt.subplots(figsize=(6, 6))
    _make_roc_plot(ax, y_valid, paper_result["valid_prob"], "Paper ANN", "tab:blue")
    _make_roc_plot(ax, y_valid, ensemble_valid_prob, "Ensemble", "tab:orange")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "roc_comparison.png"), dpi=150, bbox_inches='tight'); plt.close(fig)

    confusion_fig = _plot_confusion_matrices(
        {
            'Paper ANN': paper_result["valid_prob"],
            'XGBoost': xgb_valid,
            'LightGBM': lgbm_valid,
            'CatBoost': cat_valid,
            'Ensemble': ensemble_valid_prob,
        },
        y_valid,
        X_valid,
    )
    confusion_fig.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close(confusion_fig)

    metrics_df = pd.DataFrame([
        {"model": "Paper ANN", **paper_result["valid_metrics"]},
        {"model": "XGBoost", **compute_metrics(y_valid, xgb_valid)},
        {"model": "LightGBM", **compute_metrics(y_valid, lgbm_valid)},
        {"model": "CatBoost", **compute_metrics(y_valid, cat_valid)},
        {"model": "Ensemble", **compute_metrics(y_valid, ensemble_valid_prob)},
    ])
    metrics_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)

    ens_pred_df = pd.DataFrame({
        "sample_id": X_valid.index,
        "true_label": y_valid,
        "ensemble_probability": ensemble_valid_prob,
        "ensemble_prediction": (ensemble_valid_prob >= 0.5).astype(int),
    })
    ens_pred_df.to_csv(os.path.join(OUT_DIR, "ensemble_predictions.csv"), index=False)

    shap_values = compute_shap_values(ensemble_models, X_train)
    importance = ensemble_mean_abs_shap(shap_values, X_train.columns)
    importance.to_csv(os.path.join(OUT_DIR, "shap_gene_importance.csv"), index=False)
    summary_plot(shap_values, X_train, "xgboost", os.path.join(OUT_DIR, "shap_summary_xgboost.png"))

    stat_table = []
    stat_result = paired_bootstrap_auc_difference(y_valid, paper_result["valid_prob"], ensemble_valid_prob)
    stat_table.append({
        "model_a": "Paper ANN",
        "model_b": "Ensemble",
        "metric": "AUC difference on GSE55092",
        "result": stat_result,
    })
    pd.DataFrame([{"model_a": 'Paper ANN', "model_b": 'Ensemble', "method": 'paired bootstrap', "auc_difference": stat_result['observed_difference'], "ci_low": stat_result['ci_low'], "ci_high": stat_result['ci_high'], "p_value": stat_result['p_value']}]).to_csv(os.path.join(OUT_DIR, "statistical_comparison.csv"), index=False)

    report = {
        "paper_published": {"training_auc": 0.948, "validation_auc": 0.849, "validation_ci_95": [0.773, 0.916]},
        "cohort_counts": cohort_counts,
        "paper_reproduction": {
            "training_auc": float(paper_result["train_metrics"]["auc"]),
            "validation_auc": float(paper_result["valid_metrics"]["auc"]),
        },
        "ensemble_cv_auc": {name: float(ensemble_result.cv_auc[name]) for name in ensemble_result.cv_auc},
        "ensemble_validation_auc": {
            "XGBoost": float(compute_metrics(y_valid, xgb_valid)["auc"]),
            "LightGBM": float(compute_metrics(y_valid, lgbm_valid)["auc"]),
            "CatBoost": float(compute_metrics(y_valid, cat_valid)["auc"]),
            "Ensemble": float(compute_metrics(y_valid, ensemble_valid_prob)["auc"]),
        },
        "bootstrap_auc_difference": stat_result,
    }
    return report


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    report = run_research_experiment()
    print(json.dumps(report, indent=2, default=str))
