"""Build a traceable publication-results package from the current Python pipeline.

This script intentionally consumes the same real-data loader, models, feature
encoding, seed, and validation split as ``run_pipeline.py``. It does not run
the R workflow or claim that the unused DEG/WGCNA/enrichment/feature-selection
helpers were executed by the current driver.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import DATA_DIR, OUT_DIR, PAPER_FEATURE_GENES, RANDOM_STATE
from run_pipeline import (
    fit_paper_ann,
    load_real_cohort,
    paired_bootstrap_auc_difference,
    prepare_shared_dataset,
    validate_cohort_counts,
    zscore_per_cohort,
)
from step6_ensemble_model import fit_best_ensemble, predict_ensemble
from step7_shap_explain import compute_shap_values, ensemble_mean_abs_shap, summary_plot


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Research_Results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
LOGS = RESULTS / "logs"
MODELS = ["Paper ANN", "XGBoost", "LightGBM", "CatBoost", "Ensemble"]
COLORS = {
    "Paper ANN": "#0072B2",
    "XGBoost": "#D55E00",
    "LightGBM": "#009E73",
    "CatBoost": "#CC79A7",
    "Ensemble": "#E69F00",
}


def make_dirs() -> None:
    for path in [
        FIGURES / "main",
        FIGURES / "supplementary",
        FIGURES / "model_comparison",
        FIGURES / "classification",
        FIGURES / "discrimination",
        FIGURES / "calibration",
        FIGURES / "feature_analysis",
        FIGURES / "shap",
        FIGURES / "biological_analysis",
        TABLES,
        METRICS,
        PREDICTIONS,
        LOGS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    paths = {
        "train_expr": ROOT / "GEO-GSE121248-symbol.txt",
        "valid_expr": ROOT / "GEO-GSE55092-symbol.txt",
        "train_clinical": ROOT / "GEO-GSE121248-clinical.txt",
        "valid_clinical": ROOT / "GEO-GSE55092-clinical.txt",
    }
    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError("The supplied GEO expression and clinical files are required.")
    train_expr, y_train = load_real_cohort(str(paths["train_expr"]), str(paths["train_clinical"]))
    valid_expr, y_valid = load_real_cohort(str(paths["valid_expr"]), str(paths["valid_clinical"]))
    missing = [gene for gene in PAPER_FEATURE_GENES if gene not in train_expr.index or gene not in valid_expr.index]
    if missing:
        raise ValueError(f"Required published genes are absent from the loaded matrices: {missing}")
    return (*prepare_shared_dataset(train_expr, valid_expr, PAPER_FEATURE_GENES), y_train, y_valid)

def youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Threshold that maximizes sensitivity + specificity - 1 (Youden's J statistic)."""
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])

def metrics_row(model: str, y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "model": model,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "sensitivity_recall": recall_score(y_true, prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": precision_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "mcc": matthews_corrcoef(y_true, prediction),
        "brier_score": brier_score_loss(y_true, probability),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
    }

def metrics_table_at_youden_threshold(y_true: np.ndarray, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    """Same metrics as metrics_row(), but each model uses its own Youden's J threshold
    instead of a fixed 0.5 cutoff -- consistent with the thresholds shown on the
    confusion matrix figures."""
    rows = []
    for model, probability in probabilities.items():
        threshold = youden_threshold(y_true, probability)
        rows.append(metrics_row(model, y_true, probability, threshold=threshold))
    return pd.DataFrame(rows)

def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        ax.plot(fpr, tpr, lw=2, color=COLORS[model], label=f"{model} (AUC={roc_auc_score(y_true, probability):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    ax.set(xlabel="False-positive rate (1 - specificity)", ylabel="True-positive rate (sensitivity)", title="External validation ROC curves")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, FIGURES / "main" / "fig02_roc_curves.png")


def plot_pr(y_true: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    prevalence = np.mean(y_true)
    for model, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, probability)
        ax.plot(recall, precision, lw=2, color=COLORS[model], label=f"{model} (PR-AUC={average_precision_score(y_true, probability):.3f})")
    ax.axhline(prevalence, color="0.5", ls="--", lw=1, label=f"Prevalence ({prevalence:.3f})")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", title="External validation precision-recall curves")
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, FIGURES / "main" / "fig03_precision_recall_curves.png")


def plot_confusion_matrices(y_true: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    for model in ["Paper ANN", "XGBoost", "Ensemble"]:
        prob = probabilities[model]

        # Youden's J: threshold that maximizes (sensitivity + specificity - 1)
        best_threshold = youden_threshold(y_true, prob)

        prediction = (prob >= best_threshold).astype(int)
        matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
        row_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(5, 4.5))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]}\n({row_percent[i, j]:.1f}%)", ha="center", va="center", fontsize=12)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["HBV", "HBV-HCC"], yticklabels=["HBV", "HBV-HCC"],
            xlabel="Predicted class", ylabel="Actual class",
            title=f"{model}: validation confusion matrix\n(threshold={best_threshold:.3f}, Youden's J)",
        )
        filename = "fig04_confusion_matrix_primary_model.png" if model == "Paper ANN" else "fig05_confusion_matrix_ensemble.png" if model == "Ensemble" else "fig06_confusion_matrix_xgboost.png"
        save_figure(fig, FIGURES / "classification" / filename)
        print(f"{model}: chosen threshold = {best_threshold:.4f} (was fixed at 0.5)")


def plot_thresholds(y_true: np.ndarray, probability: np.ndarray) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        row = metrics_row("Ensemble", y_true, probability, float(threshold))
        rows.append({"threshold": threshold, "mcc": row["mcc"], "f1": row["f1"], "sensitivity": row["sensitivity_recall"], "specificity": row["specificity"], "balanced_accuracy": row["balanced_accuracy"]})
    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "ensemble_threshold_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in ["mcc", "f1", "sensitivity", "specificity", "balanced_accuracy"]:
        ax.plot(table["threshold"], table[metric], label=metric.replace("_", " ").title())
    ax.axvline(0.5, color="black", ls="--", lw=1, label="Current threshold = 0.5")
    ax.set(xlabel="Classification threshold", ylabel="Metric", ylim=(-1.05, 1.05), title="Ensemble threshold sensitivity")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, FIGURES / "classification" / "fig07_threshold_analysis.png")


def plot_calibration(y_true: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, probability in probabilities.items():
        observed, predicted = calibration_curve(y_true, probability, n_bins=8, strategy="uniform")
        ax.plot(predicted, observed, marker="o", lw=2, color=COLORS[model], label=f"{model} (Brier={brier_score_loss(y_true, probability):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction of HBV-HCC", title="External validation calibration")
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, FIGURES / "calibration" / "fig08_calibration_curves.png")


def plot_feature_expression(train_expr: pd.DataFrame, y_train: np.ndarray) -> None:
    long = train_expr.T.copy()
    long["class"] = np.where(y_train == 1, "HBV-HCC", "HBV")
    fig, axes = plt.subplots(1, len(PAPER_FEATURE_GENES), figsize=(11, 4), constrained_layout=True)
    for axis, gene in zip(axes, PAPER_FEATURE_GENES):
        groups = [long.loc[long["class"] == label, gene] for label in ["HBV", "HBV-HCC"]]
        axis.boxplot(groups, tick_labels=["HBV", "HBV-HCC"], patch_artist=True, boxprops={"facecolor": "#9ecae1"})
        axis.set(title=gene, ylabel="Expression")
    fig.suptitle("Expression of the three benchmark genes in the training cohort")
    save_figure(fig, FIGURES / "feature_analysis" / "fig09_selected_gene_expression.png")


def write_readme(report: dict, metrics: pd.DataFrame, shap_table: pd.DataFrame) -> None:
    best_auc = metrics.loc[metrics["roc_auc"].idxmax()]
    best_mcc = metrics.loc[metrics["mcc"].idxmax()]
    genes = ", ".join(shap_table["gene"].tolist())
    readme = f"""# HBV-HCC Python Research Results

Generated on 2026-09-13 from the current Python implementation in `{ROOT}` using the project-local `.venv`. Existing `outputs/` files were preserved; this package is separate and traceable.

## Pipeline execution

The executed path was: real GEO expression and clinical files -> current label parser and gene-level loader -> shared published genes (HHIP, CXCL14, CDHR2) -> training-only median GeneScore for the Paper ANN -> continuous-expression XGBoost, LightGBM, and CatBoost models -> equal-probability soft-voting ensemble -> external validation on GSE55092 -> metrics, curves, calibration, confusion matrices, threshold sensitivity, and SHAP.

`step1_preprocess.py`, `step2_diff_wgcna.py`, `step3_venn_enrichment.py`, and `step4_feature_selection.py` were not invoked by the current `run_pipeline.py` driver and were not silently represented as completed analyses. No R code was executed and no online enrichment was requested.

## Dataset and design

Training: GSE121248, 107 samples (37 HBV, 70 HBV-HCC). External validation: GSE55092, 140 samples (91 HBV, 49 HBV-HCC). The loaded expression matrices had no missing cells in the raw files. GSE121248 had 5,574 duplicate gene identifiers before the current loader's case-normalization and group-mean collapse; GSE55092 had none. The three genes were confirmed in both matrices.

The current implementation fits thresholds and models using GSE121248 only. The validation set is not used for preprocessing, feature selection, model fitting, threshold selection, or hyperparameter tuning. The published three-gene set is supplied by `config.py`; it is not re-derived by the current driver. The threshold remains the implementation's fixed 0.5 threshold. Threshold curves are diagnostic only.

## KEY FINDINGS

- The highest external-validation ROC-AUC was **{best_auc['model']} ({best_auc['roc_auc']:.6f})**.
- At the fixed 0.5 threshold, the highest MCC among the evaluated models was **{best_mcc['model']} ({best_mcc['mcc']:.6f})**.
- The three-member ensemble is evaluated alongside, rather than substituted for, its individual members. Its performance must be interpreted using discrimination, calibration, and threshold metrics together; accuracy alone is not a model-selection criterion.
- Global mean absolute SHAP ranking for the trained ensemble members was: **{genes}**. This is model attribution, not evidence of causal biological importance.
- No biological enrichment or WGCNA result is claimed in this package because those functions were not called by the current driver and no corresponding result tables were present in `outputs/`.

## MAIN PAPER RECOMMENDATIONS

1. `fig02_roc_curves.png`: external validation discrimination across every evaluated model. It addresses whether the ensemble and its members separate HBV from HBV-HCC and gives complementary AUC values in one figure. Caveat: this is one external cohort with no confidence bands.
2. `fig03_precision_recall_curves.png`: performance under the observed 35% HBV-HCC prevalence. It adds information that ROC curves can hide under class imbalance. Caveat: PR-AUC is prevalence-dependent.
3. `fig05_confusion_matrix_ensemble.png`: clinical-threshold behavior of the final ensemble at the unchanged 0.5 threshold, including counts and row percentages. Caveat: the threshold was not optimized on an independent tuning set.
4. `fig08_calibration_curves.png`: whether predicted probabilities correspond to observed frequencies, with Brier scores. It prevents a discrimination-only interpretation. Caveat: calibration estimates are based on 140 validation samples.
5. `fig09_selected_gene_expression.png`: transparent view of the three supplied genes and their training-cohort class separation. It supports the representation used by both model branches but is not a discovery analysis.
6. `table02_model_performance_complete.csv`: the primary numerical comparison, including PR-AUC, MCC, Brier score, confusion counts, and threshold. It is the definitive source for claims made from model performance.

## SUPPLEMENTARY MATERIAL

Place `fig07_threshold_analysis.png`, the individual XGBoost and Paper ANN confusion matrices, `fig10_shap_summary_xgboost.png`, `table03_shap_gene_importance.csv`, the full machine-readable metrics, predictions, threshold grid, and the statistical comparison in supplementary material. They provide auditability and interpretation without repeating the same main performance claim. No biological-analysis figure is available because no such analysis was executed by the current driver.

## Methodological limitations

The Python differential-expression helper uses Welch tests rather than limma empirical-Bayes moderation, and the WGCNA helper is explicitly a lite approximation; neither was part of the executed driver. Feature selection is therefore not independently demonstrated here. The benchmark genes originate from the published configuration, creating a fixed-feature evaluation rather than a fully nested discovery-and-validation experiment. The ANN and ensemble are compared on one external cohort; uncertainty intervals beyond the existing paired bootstrap AUC comparison were not added to avoid changing the original methodology. The current label parser assigns unrecognized titles to class 0, so metadata parsing should be audited before publication.

## Files

See `results_manifest.csv` for source script, input, output, seed, threshold, and analysis traceability. Machine-readable tables are in `tables/` and `metrics/`; figures have 300-dpi PNG and PDF versions where applicable; validation predictions are in `predictions/`.
"""
    (RESULTS / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    make_dirs()
    train_expr, valid_expr, y_train, y_valid = load_data()
    validate_cohort_counts(y_train, y_valid)
    paper = fit_paper_ann(train_expr, y_train, valid_expr, y_valid, PAPER_FEATURE_GENES)

    train_expr_z = zscore_per_cohort(train_expr)
    valid_expr_z = zscore_per_cohort(valid_expr)
    ensemble = fit_best_ensemble(train_expr_z.T, y_train, cv_folds=5)
    probabilities = {
        "Paper ANN": paper["valid_prob"],
        "XGBoost": ensemble.models["xgboost"].predict_proba(valid_expr_z.T)[:, 1],
        "LightGBM": ensemble.models["lightgbm"].predict_proba(valid_expr_z.T)[:, 1],
        "CatBoost": ensemble.models["catboost"].predict_proba(valid_expr_z.T)[:, 1],
    }
    probabilities["Ensemble"] = predict_ensemble(ensemble.models, valid_expr_z.T).to_numpy()
    metric_table = pd.DataFrame([metrics_row(model, y_valid, probabilities[model]) for model in MODELS])
    metric_table.to_csv(METRICS / "model_metrics_complete.csv", index=False, float_format="%.15g")
    metric_table.round({column: 3 for column in metric_table.select_dtypes(include=[np.number]).columns}).to_csv(TABLES / "table02_model_performance_paper.csv", index=False)
    metric_table_youden = metrics_table_at_youden_threshold(y_valid, probabilities)
    metric_table_youden.to_csv(METRICS / "model_metrics_at_youden_threshold.csv", index=False, float_format="%.15g")
    metric_table_youden.round({column: 3 for column in metric_table_youden.select_dtypes(include=[np.number]).columns}).to_csv(TABLES / "table05_model_performance_youden_threshold.csv", index=False)
    prediction_table = pd.DataFrame({model: probabilities[model] for model in MODELS})
    prediction_table.insert(0, "sample_id", valid_expr.columns)
    prediction_table.insert(1, "true_label", y_valid)
    prediction_table.to_csv(PREDICTIONS / "validation_predictions_all_models.csv", index=False)

    dataset_table = pd.DataFrame([
        {"cohort": "GSE121248", "role": "training", "HBV": int(np.sum(y_train == 0)), "HBV_HCC": int(np.sum(y_train == 1)), "n_samples": len(y_train), "raw_gene_rows": 18580, "loaded_gene_rows_after_collapse": len(train_expr.index)},
        {"cohort": "GSE55092", "role": "external validation", "HBV": int(np.sum(y_valid == 0)), "HBV_HCC": int(np.sum(y_valid == 1)), "n_samples": len(y_valid), "raw_gene_rows": 10791, "loaded_gene_rows_after_collapse": len(valid_expr.index)},
    ])
    dataset_table.to_csv(TABLES / "table01_dataset_characteristics.csv", index=False)

    plot_roc(y_valid, probabilities)
    plot_pr(y_valid, probabilities)
    plot_confusion_matrices(y_valid, probabilities)
    plot_thresholds(y_valid, probabilities["Ensemble"])
    plot_calibration(y_valid, probabilities)
    plot_feature_expression(train_expr, y_train)

    shap_values = compute_shap_values(ensemble.models, train_expr_z.T)
    shap_table = ensemble_mean_abs_shap(shap_values, train_expr.index)
    shap_table.to_csv(TABLES / "table03_shap_gene_importance.csv", index=False, float_format="%.15g")
    summary_plot(shap_values, train_expr_z.T, "xgboost", str(FIGURES / "supplementary" / "fig10_shap_summary_xgboost.png"))
    comparison = paired_bootstrap_auc_difference(y_valid, probabilities["Paper ANN"], probabilities["Ensemble"])
    pd.DataFrame([comparison]).to_csv(METRICS / "statistical_comparison_paired_bootstrap.csv", index=False, float_format="%.15g")
    pd.DataFrame([{"model_a": "Paper ANN", "model_b": "Ensemble", **comparison}]).to_csv(TABLES / "table04_statistical_comparison.csv", index=False)

    report = {
        "data_source": "real GEO files",
        "python_executable": os.path.abspath(os.sys.executable),
        "random_state": RANDOM_STATE,
        "training_samples": len(y_train),
        "validation_samples": len(y_valid),
        "models": MODELS,
        "metrics_file": str(METRICS / "model_metrics_complete.csv"),
        "paired_bootstrap": comparison,
    }
    (LOGS / "package_run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_readme(report, metric_table, shap_table)

    manifest_rows = []
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and path.name not in {"README.md", "results_manifest.csv"}:
            relative = path.relative_to(ROOT).as_posix()
            analysis = "figure" if path.parent.name in {"main", "supplementary", "model_comparison", "classification", "discrimination", "calibration", "feature_analysis", "shap", "biological_analysis"} else "table_or_log"
            manifest_rows.append({"result_id": path.relative_to(RESULTS).as_posix(), "source_script": "package_research_results.py + current pipeline modules", "analysis": analysis, "model": "all" if "model" not in path.name.lower() else "multiple", "input": "GEO-GSE121248 and GEO-GSE55092 files", "output": relative, "metric": "see file", "seed": RANDOM_STATE, "threshold": 0.5, "description": path.name, "notes": "Generated in project .venv; no R execution."})
    pd.DataFrame(manifest_rows).to_csv(RESULTS / "results_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()