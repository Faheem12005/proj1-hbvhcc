"""
Maps: 4.lasso.txt, 4.randomForest.txt, 4.SVM-REF.txt

  R randomForest              -> sklearn.ensemble.RandomForestClassifier
  R glmnet (LASSO, alpha=1)   -> sklearn.linear_model.LogisticRegressionCV(penalty='l1')
  R e1071 + sigFeature (SVM-RFE) -> sklearn.feature_selection.RFECV with an SVC(kernel='linear')

All three take X = samples x genes (already restricted to the candidate
gene pool from step 3) and a binary y (0=HBV, 1=HBV-HCC), matching the R
scripts' `group=c(rep("0",ncon),rep("1",ntreat))`.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from config import RANDOM_STATE


def random_forest_features(X: pd.DataFrame, y: np.ndarray, n_trees: int = 500,
                            importance_threshold: float = 1.5) -> list:
    """4.randomForest.txt: fit RF, take genes with importance > threshold.
    sklearn's feature_importances_ is on a different scale than R's
    MeanDecreaseGini, so we rank-normalize to a comparable 0-100 scale
    before applying the same style of cutoff (top genes by importance)."""
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=RANDOM_STATE, oob_score=True)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    scaled = importance / importance.max() * 100
    selected = scaled[scaled > importance_threshold].sort_values(ascending=False)
    return selected.index.tolist()


def svm_rfe_features(X: pd.DataFrame, y: np.ndarray, cv_folds: int = 10) -> list:
    """4.SVM-REF.txt: SVM recursive feature elimination with cross-validation,
    picking the feature count that minimizes CV error (mirrors msvmRFE.R)."""
    svc = SVC(kernel="linear")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(estimator=svc, step=1, cv=cv, scoring="accuracy", min_features_to_select=1)
    selector.fit(X, y)
    selected = X.columns[selector.support_].tolist()
    return selected


def lasso_features(X: pd.DataFrame, y: np.ndarray, cv_folds: int = 10) -> list:
    """4.lasso.txt: LASSO logistic regression (glmnet alpha=1), 10-fold CV
    to pick lambda.min, keep genes with non-zero coefficients."""
    model = LogisticRegressionCV(
        Cs=100, cv=cv_folds, penalty="l1", solver="liblinear",
        scoring="neg_log_loss", random_state=RANDOM_STATE, max_iter=5000,
    )
    model.fit(X, y)
    coef = pd.Series(model.coef_.ravel(), index=X.columns)
    return coef[coef != 0].index.tolist()


def intersect_feature_genes(rf_genes: list, svm_genes: list, lasso_genes: list) -> list:
    """4.*.txt final step: Venn intersection of the three algorithms' outputs."""
    return sorted(set(rf_genes) & set(svm_genes) & set(lasso_genes))
