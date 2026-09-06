"""
NEW STEP — this is the actual improvement over the paper, not a straight
R->Python port. The paper's diagnostic model is a single 3-input, 1-hidden-
layer ANN (input=3 GeneScores, hidden=5 units). We replace it with a
soft-voting ensemble of three gradient-boosting models, each of which
handles small-n / few-feature tabular data at least as well as a tiny MLP,
and each of which SHAP can explain natively and efficiently (TreeExplainer).

Design choices, and why:
  * Only 3 features (HHIP, CXCL14, CDHR2) -> deliberately shallow trees
    (max_depth 2-3) and heavy regularization to avoid overfitting a model
    with far more capacity than 3 inputs need.
  * Soft voting (average predicted probability) rather than a stacked
    meta-learner, because with ~100-140 total samples a second-level
    learner would overfit; simple averaging is the safer default. Swap in
    `StackingClassifier` if you have a larger validation cohort to fit the
    meta-learner on.
  * Cross-validated AUC is used for model selection/comparison, mirroring
    the paper's train/validate split (GSE121248 train -> GSE55092 test).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from config import RANDOM_STATE


def _default_models() -> dict:
    return {
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=1.0, reg_lambda=2.0,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=2, num_leaves=4, learning_rate=0.05,
            subsample=0.8, reg_lambda=2.0, random_state=RANDOM_STATE, verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300, depth=2, learning_rate=0.05, l2_leaf_reg=4.0,
            random_state=RANDOM_STATE, verbose=False,
        ),
    }


@dataclass
class EnsembleResult:
    models: dict = field(default_factory=dict)
    oof_proba: pd.DataFrame = None
    ensemble_oof_proba: np.ndarray = None
    cv_auc: dict = field(default_factory=dict)
    model_proba_by_name: dict = field(default_factory=dict)


def fit_best_ensemble(X: pd.DataFrame, y: np.ndarray, cv_folds: int = 5) -> EnsembleResult:
    """Leakage-safe training on GSE121248 only.

    We compute out-of-fold predictions with stratified CV, then refit each model on the full
    training cohort before external validation on GSE55092. No validation data are used for tuning.
    """
    models = _default_models()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    oof = {}
    for name, model in models.items():
        proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]
        oof[name] = proba
        model.fit(X, y)

    oof_df = pd.DataFrame(oof, index=X.index)
    ensemble_oof = oof_df.mean(axis=1).to_numpy()
    cv_auc = {name: float(roc_auc_score(y, oof_df[name])) for name in oof_df.columns}
    cv_auc["ensemble"] = float(roc_auc_score(y, ensemble_oof))

    model_proba_by_name = {name: {"oof_prob": oof_df[name].to_numpy()} for name in oof_df.columns}

    return EnsembleResult(
        models=models,
        oof_proba=oof_df,
        ensemble_oof_proba=ensemble_oof,
        cv_auc=cv_auc,
        model_proba_by_name=model_proba_by_name,
    )


def predict_ensemble(models: dict, X: pd.DataFrame) -> pd.Series:
    """Average predicted probability of class 1 (HBV-HCC) across the three fitted models."""
    probs = np.column_stack([m.predict_proba(X)[:, 1] for m in models.values()])
    return pd.Series(probs.mean(axis=1), index=X.index, name="ensemble_proba")
