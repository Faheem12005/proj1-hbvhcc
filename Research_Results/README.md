# HBV-HCC Python Research Results

Generated on 2026-09-13 from the current Python implementation in `/Users/ananya/Downloads/VIT/proj1-hbvhcc` using the project-local `.venv`. Existing `outputs/` files were preserved; this package is separate and traceable.

## Pipeline execution

The executed path was: real GEO expression and clinical files -> current label parser and gene-level loader -> shared published genes (HHIP, CXCL14, CDHR2) -> training-only median GeneScore for the Paper ANN -> continuous-expression XGBoost, LightGBM, and CatBoost models -> equal-probability soft-voting ensemble -> external validation on GSE55092 -> metrics, curves, calibration, confusion matrices, threshold sensitivity, and SHAP.

`step1_preprocess.py`, `step2_diff_wgcna.py`, `step3_venn_enrichment.py`, and `step4_feature_selection.py` were not invoked by the current `run_pipeline.py` driver and were not silently represented as completed analyses. No R code was executed and no online enrichment was requested.

## Dataset and design

Training: GSE121248, 107 samples (37 HBV, 70 HBV-HCC). External validation: GSE55092, 140 samples (91 HBV, 49 HBV-HCC). The loaded expression matrices had no missing cells in the raw files. GSE121248 had 5,574 duplicate gene identifiers before the current loader's case-normalization and group-mean collapse; GSE55092 had none. The three genes were confirmed in both matrices.

The current implementation fits thresholds and models using GSE121248 only. The validation set is not used for preprocessing, feature selection, model fitting, threshold selection, or hyperparameter tuning. The published three-gene set is supplied by `config.py`; it is not re-derived by the current driver. The threshold remains the implementation's fixed 0.5 threshold. Threshold curves are diagnostic only.

## KEY FINDINGS

- The highest external-validation ROC-AUC was **XGBoost (0.959408)**.
- At the fixed 0.5 threshold, the highest MCC among the evaluated models was **LightGBM (0.375072)**.
- The three-member ensemble is evaluated alongside, rather than substituted for, its individual members. Its performance must be interpreted using discrimination, calibration, and threshold metrics together; accuracy alone is not a model-selection criterion.
- Global mean absolute SHAP ranking for the trained ensemble members was: **CXCL14, CDHR2, HHIP**. This is model attribution, not evidence of causal biological importance.
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
