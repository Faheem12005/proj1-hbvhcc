# HBV-HCC diagnostic model: R → Python port + gradient-boosting upgrade

This ports the Zhang et al. 2023 (_Tumour Virus Research_) pipeline from R
to Python, then replaces their single 3-input ANN with a soft-voting
ensemble of XGBoost + LightGBM + CatBoost, explained with SHAP.

## Status

All 8 stages run end-to-end (`src/run_pipeline.py` demo confirmed working).
Stages 1-4 (preprocessing → DEGs/WGCNA → enrichment → feature selection)
are direct ports of your R scripts and are ready to point at your actual
`GSE121248`/`GSE55092` files — **no real GEO data was uploaded**, so
stages 1-4 haven't been run against real numbers yet. Stages 5-8 (GeneScore
baseline, the new ensemble, SHAP, comparison) were smoke-tested on
synthetic data that mimics the paper's cohort sizes and effect direction,
just to prove the plumbing works — swap in your real expression matrix
before trusting any AUC number.

## File-by-file mapping

| R script                                  | Python module                                             | What changed                                                                                          |
| ----------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `1.Extraction Matrix.txt`                 | `src/step1_preprocess.py`                                 | `avereps` → pandas `groupby(...).mean()`                                                              |
| `2_af_Diff_wgcna.txt` (DEG half)          | `src/step2_diff_wgcna.py::differential_expression`        | `limma` eBayes moderated t-test → Welch's t-test + BH FDR (`statsmodels`). See caveat below.          |
| `2_af_Diff_wgcna.txt` (WGCNA half)        | `src/step2_diff_wgcna.py::wgcna_lite`                     | Reimplemented adjacency/TOM/clustering/module-trait-correlation in numpy/scipy. See caveat below.     |
| `3.1.venn.txt`                            | `src/step3_venn_enrichment.py::venn_intersection`         | `venn` R package → plain set intersection                                                             |
| `3.2.DO/GO/KEGG.txt`                      | `src/step3_venn_enrichment.py::run_enrichr` / `local_ora` | `clusterProfiler` → `gseapy` (online) or a hand-rolled hypergeometric test (offline, same statistics) |
| `4.randomForest.txt`                      | `src/step4_feature_selection.py::random_forest_features`  | `randomForest` → `sklearn.ensemble.RandomForestClassifier`                                            |
| `4.SVM-REF.txt`                           | `src/step4_feature_selection.py::svm_rfe_features`        | `e1071`/`sigFeature` → `sklearn.feature_selection.RFECV` with linear SVM                              |
| `4.lasso.txt`                             | `src/step4_feature_selection.py::lasso_features`          | `glmnet` → `sklearn.linear_model.LogisticRegressionCV(penalty='l1')`                                  |
| `5.neuralgeneScore.txt`                   | `src/step5_baseline_ann.py::gene_score`                   | Direct port (identical median-split rule)                                                             |
| `5.neuralNet.txt`                         | `src/step5_baseline_ann.py::train_ann_baseline`           | `neuralnet(hidden=5)` → `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(5,))`               |
| `5.neuralROC.txt` / `5.neuraltestROC.txt` | `src/step5_baseline_ann.py::roc_with_bootstrap_ci`        | `pROC::ci.auc(method='bootstrap')` → manual bootstrap of `roc_auc_score`                              |
| — (new)                                   | `src/step6_ensemble_model.py`                             | XGBoost + LightGBM + CatBoost soft-voting ensemble, replacing the single ANN                          |
| — (new)                                   | `src/step7_shap_explain.py`                               | SHAP `TreeExplainer` — per-gene, per-sample attribution the paper doesn't have                        |
| — (new)                                   | `src/step8_compare_eval.py`                               | Head-to-head AUC table + ROC overlay, baseline vs ensemble                                            |

### Fidelity caveats (read before trusting exact numbers)

- **limma's eBayes** shrinks per-gene variance estimates across the whole
  dataset before testing; the Python port here is an ordinary Welch t-test,
  which will be _close_ but not numerically identical, especially for
  genes with small within-group variance. For exact parity, run limma via
  `rpy2` and treat that as ground truth for the DEG list.
- **WGCNA** has no faithful line-for-line Python port. `wgcna_lite()` does
  the same conceptual steps (soft-power selection by scale-free fit →
  adjacency → TOM → hierarchical clustering → module eigengenes →
  module-trait correlation) but will not produce identical module
  boundaries to R's `dynamicTreeCut`. If your paper's contribution
  isn't about the WGCNA step itself, consider just reusing the paper's
  published candidate gene list (or rerun the real R WGCNA once and treat
  its output as a fixed input to your Python pipeline) rather than
  re-deriving modules in Python.
- **GO/KEGG/DO enrichment**: `gseapy.enrichr` hits a live web API and uses
  Enrichr's curated libraries, which are not byte-identical to
  `clusterProfiler`'s GO.db/KEGG.db snapshots — expect similar but not
  identical term lists. `local_ora()` is provided for a fully offline,
  statistically-equivalent alternative if you supply your own `.gmt` files.

## The actual improvement (what to write up)

```
GEO data → preprocess → DEGs ∩ WGCNA module → RF/SVM-RFE/LASSO → 3 genes
                                                                    │
                                        ┌───────────────────────────┴────────────────────────────┐
                                        │                                                         │
                                paper baseline                                          your contribution
                                (GeneScore + 1 ANN)                              (raw expression → XGB+LGBM+CatBoost)
                                        │                                                         │
                                   ROC / AUC  ──────────────────  compare on same train/test split ──────  SHAP explanation
```

Two independent levers to report on:

1. **Model architecture**: does a small gradient-boosting ensemble
   out-perform the paper's single ANN on the _same_ 3 genes, same
   train/test split (GSE121248 → GSE55092)? This isolates the modeling
   contribution.
2. **Explainability**: SHAP gives a ranked, signed, per-patient
   contribution table straight from the deployed model — the paper only
   argues gene relevance indirectly (GeneMANIA/GSEA on the genes in
   isolation, not on the classifier's actual decision function).

Note the ensemble also drops the paper's manual 0/1 GeneScore
binarization — gradient-boosted trees split on raw expression values
directly, so you keep continuous information the ANN's discretization step
throws away. That's worth calling out explicitly as a second, smaller
methodological improvement.

## Running it

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py        # runs on synthetic data, proves the pipeline works
```

To run on real data:

1. Download `GSE121248` and `GSE55092` series matrices + the `GPL570-55999`
   platform annotation from GEO.
2. Run `step1_preprocess.extract_matrix()` on each.
3. Run `step2_diff_wgcna` for DEGs + WGCNA modules on the training cohort.
4. Intersect with `step3_venn_enrichment.venn_intersection()`, run
   enrichment for the write-up's figures.
5. Run `step4_feature_selection` (RF ∩ SVM-RFE ∩ LASSO) to get your
   feature gene set (may or may not reproduce HHIP/CXCL14/CDHR2 exactly —
   that's fine and worth reporting either way).
6. Feed the resulting expression matrix into `run_pipeline.run_diagnostic_comparison()`
   in place of `make_synthetic_cohort()`.

## Detailed methodology and system design

### Research objective

The system classifies samples as HBV (label 0) or HBV-associated hepatocellular
carcinoma, HBV-HCC (label 1), using three candidate genes from the paper:
HHIP, CXCL14, and CDHR2. The paper's single hidden-layer ANN is retained as a
reproducible baseline. The proposed model uses raw continuous expression and a
soft-voting ensemble of XGBoost, LightGBM, and CatBoost.

### Data flow

```text
GEO probe matrix + clinical metadata
             |
             v
  probe-to-gene collapsing and label parsing
             |
             v
  training cohort: GSE121248 / validation cohort: GSE55092
             |
             +--> differential expression + WGCNA-lite
             |          |
             |          +--> candidate intersection and enrichment
             |          +--> RF / SVM-RFE / LASSO feature selection
             |
             v
     shared three-gene benchmark dataset
             |
       +-----+------------------+
       |                        |
       v                        v
  median GeneScore          raw expression
       |                        |
       v                        v
  paper ANN baseline       XGBoost + LightGBM + CatBoost
                                |
                                v
                       average probability ensemble
                                |
                 ROC, metrics, bootstrap test, SHAP
```

### Methodology

1. `step1_preprocess.py` maps platform probe IDs to gene symbols, converts
   expression values to numeric values, and averages duplicate probes.
2. `step2_diff_wgcna.py` performs Welch differential tests with
   Benjamini-Hochberg FDR correction and a lightweight WGCNA-style network
   analysis. The Python DEG calculation approximates, but does not numerically
   reproduce, R limma empirical-Bayes moderation.
3. `step3_venn_enrichment.py` intersects gene lists and performs either online
   Enrichr analysis or offline hypergeometric over-representation analysis.
4. `step4_feature_selection.py` applies random forest importance, linear SVM
   recursive feature elimination, and LASSO logistic regression.
5. `run_pipeline.py` creates one shared train/validation dataset and one shared
   three-gene feature set. The ANN uses training-only median thresholds for its
   binary GeneScore; the ensemble uses the corresponding raw expression values.
6. `step6_ensemble_model.py` trains shallow, regularized boosting models with
   stratified out-of-fold predictions on the training cohort, then refits on all
   training samples before external validation.
7. `step7_shap_explain.py` computes TreeExplainer attributions for the ensemble
   members and produces a ranked gene-importance table.

### Architecture

The design has four layers:

- **Input layer:** GEO expression matrices and sample-level clinical metadata.
- **Analysis layer:** preprocessing, differential expression, network analysis,
  enrichment, and feature selection.
- **Model layer:** a paper-faithful ANN baseline and a three-member boosting
  ensemble. Both use the same cohorts and genes; only the feature encoding and
  model architecture differ.
- **Evaluation layer:** validation metrics, ROC plots, confusion matrices,
  bootstrap AUC comparison, SHAP explanations, and CSV artifacts.

### Graph outputs

Every successful `run_pipeline.py` execution writes these graph files to
`outputs/`:

- `roc_comparison.png`: validation ROC curves for ANN and ensemble.
- `confusion_matrices.png`: threshold-0.5 confusion matrices for every model.
- `model_auc_comparison.png`: validation AUC bar chart for ANN, each boosting
  model, and the ensemble.
- `selected_gene_expression.png`: HBV versus HBV-HCC expression distributions
  for HHIP, CXCL14, and CDHR2.
- `shap_summary_xgboost.png`: per-sample feature-attribution summary for the
  XGBoost member of the ensemble.

### Verified smoke-test results

Because the repository does not include the real GEO matrices, the default run
uses synthetic cohorts with the paper's sample counts: 37 HBV and 70 HBV-HCC
training samples, followed by 91 HBV and 49 HBV-HCC validation samples. The
latest verified run produced:

| Model                | Validation AUC |
| -------------------- | -------------: |
| Paper ANN            |         0.9408 |
| XGBoost              |         0.9843 |
| LightGBM             |         0.9637 |
| CatBoost             |         0.9788 |
| Soft-voting ensemble |         0.9781 |

The ensemble's paired AUC improvement over the ANN was 0.0373, with a bootstrap
95% interval of 0.0037 to 0.0784 and p=0.014 on this synthetic benchmark.
These numbers demonstrate that the implementation runs; they are not claims
about real-world diagnostic performance. Real GEO files must be supplied and
the complete analysis rerun before drawing biological or clinical conclusions.
