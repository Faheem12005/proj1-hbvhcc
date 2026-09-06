"""
NEW STEP — the paper has no explainability analysis for its ANN (it relies
on external GeneMANIA / GSEA to argue biological relevance of HHIP, CXCL14,
CDHR2 after the fact). SHAP gives per-sample, per-gene attribution directly
from the diagnostic model itself, which is a genuine methodological
improvement to highlight in your write-up.

TreeExplainer is exact and fast for XGBoost/LightGBM/CatBoost (no sampling
approximation needed, unlike KernelExplainer, which you'd have had to use
for the paper's MLP).
"""
import numpy as np
import pandas as pd
import shap


def compute_shap_values(models: dict, X: pd.DataFrame) -> dict:
    """Returns {model_name: shap.Explanation} for each ensemble member."""
    out = {}
    for name, model in models.items():
        explainer = shap.TreeExplainer(model)
        out[name] = explainer(X)
    return out


def ensemble_mean_abs_shap(shap_dict: dict, feature_names) -> pd.DataFrame:
    """Average |SHAP value| per gene across the three models, giving one
    ranked table of gene contribution to the ensemble's diagnosis -- the
    direct analogue of the paper's RF 'importance' plot (Fig. 5B), but
    computed from the actual deployed diagnostic model instead of a
    separate feature-selection run."""
    stacked = np.stack([np.abs(exp.values).mean(axis=0) for exp in shap_dict.values()])
    mean_importance = stacked.mean(axis=0)
    return (
        pd.DataFrame({"gene": feature_names, "mean_abs_shap": mean_importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def summary_plot(shap_dict: dict, X: pd.DataFrame, model_name: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_dict[model_name], X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def waterfall_for_sample(shap_dict: dict, model_name: str, sample_iloc: int, out_path: str):
    """Per-patient explanation: why the model called *this* sample HBV-HCC
    vs HBV, gene by gene -- useful for a case-study figure in the paper."""
    import matplotlib.pyplot as plt
    shap.plots.waterfall(shap_dict[model_name][sample_iloc], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
