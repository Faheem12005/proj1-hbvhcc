"""
NEW STEP — reproduces the paper's train/test AUC table (Fig. 7B/7C: 0.948
train, 0.849 test) alongside the ensemble's numbers, on the same split, so
the comparison is apples-to-apples.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def compare_models(y_train, ann_train_scores, ens_train_scores,
                    y_test, ann_test_scores, ens_test_scores, out_path: str) -> pd.DataFrame:
    rows = [
        {"model": "ANN (paper baseline)", "cohort": "train (GSE121248)", "auc": roc_auc_score(y_train, ann_train_scores)},
        {"model": "ANN (paper baseline)", "cohort": "test (GSE55092)", "auc": roc_auc_score(y_test, ann_test_scores)},
        {"model": "XGB+LGBM+CatBoost ensemble", "cohort": "train (GSE121248)", "auc": roc_auc_score(y_train, ens_train_scores)},
        {"model": "XGB+LGBM+CatBoost ensemble", "cohort": "test (GSE55092)", "auc": roc_auc_score(y_test, ens_test_scores)},
    ]
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, y_true, scores in [
        ("ANN test", y_test, ann_test_scores),
        ("Ensemble test", y_test, ens_test_scores),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("External validation (GSE55092): baseline vs ensemble")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return table
