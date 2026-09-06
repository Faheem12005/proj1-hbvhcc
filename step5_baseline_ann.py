"""
Maps: 5.neuralgeneScore.txt, 5.neuralNet.txt, 5.neuralROC.txt and the
matching 5.neuraltest*.txt (external validation on GSE55092).

  R GeneScore binarization    -> gene_score() below (identical median-split rule)
  R neuralnet::neuralnet      -> sklearn.neural_network.MLPClassifier
  R pROC::roc/ci.auc          -> sklearn.metrics.roc_curve/roc_auc_score + bootstrap CI

This reproduces the PAPER'S baseline model so you have a like-for-like
number to beat with the XGBoost/LightGBM/CatBoost ensemble in step 6.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from config import RANDOM_STATE


def gene_score(expr_genes_by_samples: pd.DataFrame, logfc: pd.Series) -> pd.DataFrame:
    """5.neuralgeneScore.txt:
    For genes up in HBV-HCC (logFC>0): score=1 if expression>median else 0.
    For genes down in HBV-HCC (logFC<0): score=0 if expression>median else 1.
    expr_genes_by_samples: genes (rows, index=gene symbol) x samples (cols).
    """
    scores = pd.DataFrame(index=expr_genes_by_samples.index, columns=expr_genes_by_samples.columns, dtype=int)
    for gene, row in expr_genes_by_samples.iterrows():
        med = row.median()
        if logfc.get(gene, 0) > 0:
            scores.loc[gene] = (row > med).astype(int)
        else:
            scores.loc[gene] = (row <= med).astype(int)
    return scores  # genes x samples, same orientation as R's geneScore.txt


def gene_score_from_training_thresholds(expr_genes_by_samples: pd.DataFrame, training_medians: dict, downregulated_genes: set) -> pd.DataFrame:
    """Paper-faithful median split using thresholds learned on the training cohort only.

    This avoids leakage from the external validation set. The direction is set by the fact that
    HHIP, CXCL14 and CDHR2 are reported as downregulated in HBV-HCC.
    """
    scores = pd.DataFrame(index=expr_genes_by_samples.index, columns=expr_genes_by_samples.columns, dtype=int)
    for gene, row in expr_genes_by_samples.iterrows():
        med = float(training_medians[gene])
        if gene in downregulated_genes:
            scores.loc[gene] = (row <= med).astype(int)
        else:
            scores.loc[gene] = (row > med).astype(int)
    return scores


def train_ann_baseline(gene_score_train: pd.DataFrame, y_train: np.ndarray,
                        hidden_layer_sizes=(5,)) -> MLPClassifier:
    """5.neuralNet.txt: neuralnet(c+p~., data, hidden=5) -> one hidden layer
    of 5 units, matching the paper's architecture exactly.

    Implementation assumption: logistic activation with one hidden layer of 5 units,
    maximum iterations set to 2000 and a fixed random state. These choices are the
    simplest defensible ones consistent with the limited information available in the paper.
    """
    X = gene_score_train.T.to_numpy(dtype=float)  # samples x genes
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000,
                         random_state=RANDOM_STATE, activation="logistic")
    clf.fit(X, y_train)
    return clf


def roc_with_bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000):
    """5.neuralROC.txt / 5.neuraltestROC.txt: pROC::roc + ci.auc(method='bootstrap')."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    rng = np.random.RandomState(RANDOM_STATE)
    boot_aucs = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        sample_idx = resample(idx, random_state=rng.randint(0, 1_000_000))
        yt, ys = y_true[sample_idx], y_score[sample_idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    return {"fpr": fpr, "tpr": tpr, "auc": auc, "ci": (ci_low, ci_high)}
