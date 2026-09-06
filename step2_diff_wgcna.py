"""
Maps: 2_af_Diff_wgcna.txt (R / limma + WGCNA)

Two halves of that R script, ported separately below:

  A) differential_expression()  <- limma::lmFit/eBayes/topTable
  B) wgcna_lite()                <- the WGCNA package (pickSoftThreshold,
                                     adjacency, TOM, hierarchical clustering,
                                     dynamicTreeCut, module eigengenes,
                                     module-trait correlation)

Notes on fidelity:
  * limma's eBayes() shrinks per-gene variances via empirical Bayes before
    the t-test. There is no drop-in Python port; statsmodels/scipy give you
    an ordinary Welch t-test, which is a reasonable approximation for
    moderately sized microarray cohorts (n~30-90/group as in this paper) but
    will be slightly less powerful/stable for lowly-expressed genes than
    real limma. For exact numeric parity, run limma via rpy2, or use
    `pydeseq2` if you switch to RNA-seq counts instead of microarray data.
  * WGCNA has no exact Python port either. `PyWGCNA` (pip install PyWGCNA)
    wraps the same math and is the closest match if you want full fidelity
    (soft power search, TOM, dynamic tree cut). The implementation below is
    a compact, dependency-light reimplementation of the same steps so the
    pipeline runs without extra native dependencies; swap in PyWGCNA if you
    need it to match the paper's module composition more closely.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.multitest import multipletests

from config import LOGFC_CUTOFF, FDR_CUTOFF


# ---------------------------------------------------------------------------
# A) Differential expression (limma lmFit/eBayes/topTable equivalent)
# ---------------------------------------------------------------------------
@dataclass
class DEGResult:
    table: pd.DataFrame          # full result, one row per gene
    sig_genes: list               # genes passing |log2FC|>1 & FDR<0.05


def differential_expression(expr: pd.DataFrame, con_samples: list, treat_samples: list) -> DEGResult:
    """expr: genes x samples, already log2-scale (as in the R script's
    `if(max(rt)>50) rt=log2(rt+1)` guard). Returns logFC, t, p, adj.p per gene."""
    con = expr[con_samples].to_numpy(dtype=float)
    treat = expr[treat_samples].to_numpy(dtype=float)

    mean_con = con.mean(axis=1)
    mean_treat = treat.mean(axis=1)
    logfc = mean_treat - mean_con  # data already log2 -> difference of means = log2FC

    t_stat, p_val = stats.ttest_ind(treat, con, axis=1, equal_var=False, nan_policy="omit")
    _, p_adj, _, _ = multipletests(np.nan_to_num(p_val, nan=1.0), method="fdr_bh")

    table = pd.DataFrame({
        "gene": expr.index,
        "logFC": logfc,
        "t": t_stat,
        "P.Value": p_val,
        "adj.P.Val": p_adj,
    }).set_index("gene").sort_values("logFC")

    sig = table[(table["adj.P.Val"] < FDR_CUTOFF) & (table["logFC"].abs() > LOGFC_CUTOFF)]
    return DEGResult(table=table, sig_genes=sig.index.tolist())


# ---------------------------------------------------------------------------
# B) WGCNA-lite (adjacency -> TOM -> clustering -> module eigengenes -> trait corr)
# ---------------------------------------------------------------------------
def pick_soft_power(expr_samples_by_genes: pd.DataFrame, powers=range(1, 21), r2_cutoff=0.9) -> int:
    """expr_samples_by_genes: samples x genes (WGCNA's datExpr0 orientation).
    Approximates WGCNA::pickSoftThreshold's scale-free topology search."""
    corr = np.abs(np.corrcoef(expr_samples_by_genes.to_numpy().T))
    np.fill_diagonal(corr, 0)
    best_power, best_r2 = powers[0], -np.inf
    for p in powers:
        adj = corr ** p
        k = adj.sum(axis=1)  # connectivity per gene
        if k.std() == 0:
            continue
        hist, edges = np.histogram(k, bins=min(20, len(k)))
        centers = (edges[:-1] + edges[1:]) / 2
        mask = hist > 0
        if mask.sum() < 3:
            continue
        log_k, log_p = np.log10(centers[mask] + 1e-9), np.log10(hist[mask] / hist[mask].sum() + 1e-9)
        slope, intercept, r, _, _ = stats.linregress(log_k, log_p)
        r2 = r ** 2 * np.sign(slope)
        if r2 >= r2_cutoff:
            return p
        if r2 > best_r2:
            best_r2, best_power = r2, p
    return best_power


def wgcna_lite(expr_samples_by_genes: pd.DataFrame, trait: pd.Series,
               min_module_size: int = 30, merge_cut_height: float = 0.25):
    """
    expr_samples_by_genes: samples x genes.
    trait: 1/0 series indexed the same as expr_samples_by_genes (e.g. is_HCC).

    Returns (module_assignment: Series[gene->module_id],
             module_eigengenes: DataFrame[sample x module],
             module_trait_corr: DataFrame[module x {r, p}])
    """
    genes = expr_samples_by_genes.columns
    corr = np.corrcoef(expr_samples_by_genes.to_numpy().T)
    power = pick_soft_power(expr_samples_by_genes)
    adjacency = np.abs(corr) ** power
    np.fill_diagonal(adjacency, 0)

    # Topological overlap matrix (TOM)
    k = adjacency.sum(axis=1)
    a_sq = adjacency @ adjacency
    denom = np.minimum.outer(k, k) + 1 - adjacency
    with np.errstate(divide="ignore", invalid="ignore"):
        tom = (a_sq + adjacency) / denom
    np.fill_diagonal(tom, 1)
    tom = np.nan_to_num(tom, nan=0.0)
    diss_tom = 1 - tom

    # Hierarchical clustering + module cut (dynamic-tree-cut approximation)
    condensed = diss_tom[np.triu_indices_from(diss_tom, k=1)]
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=merge_cut_height, criterion="distance")

    module_assignment = pd.Series(labels, index=genes, name="module")
    counts = module_assignment.value_counts()
    keep_modules = counts[counts >= min_module_size].index
    module_assignment = module_assignment.where(module_assignment.isin(keep_modules), other=0)  # 0 = grey/unassigned

    # Module eigengenes = first PC of each module's expression
    eig = {}
    for m in sorted(module_assignment.unique()):
        if m == 0:
            continue
        cols = module_assignment[module_assignment == m].index
        sub = expr_samples_by_genes[cols].to_numpy()
        sub = sub - sub.mean(axis=0)
        u, s, vt = np.linalg.svd(sub, full_matrices=False)
        eig[f"ME{m}"] = u[:, 0] * s[0]
    module_eigengenes = pd.DataFrame(eig, index=expr_samples_by_genes.index)

    # Module-trait correlation
    rows = []
    t = trait.reindex(module_eigengenes.index).to_numpy(dtype=float)
    for col in module_eigengenes.columns:
        r, p = stats.pearsonr(module_eigengenes[col], t)
        rows.append({"module": col, "r": r, "p": p})
    module_trait_corr = pd.DataFrame(rows).set_index("module").sort_values("r", ascending=False)

    return module_assignment, module_eigengenes, module_trait_corr


def highly_correlated_module_genes(module_assignment: pd.Series, module_trait_corr: pd.DataFrame,
                                    top_n_modules: int = 1) -> list:
    """Genes belonging to the module(s) most correlated with the trait (the
    R script's 'blue module' selection step)."""
    best_modules = module_trait_corr.index[:top_n_modules]
    best_ids = [int(m.replace("ME", "")) for m in best_modules]
    return module_assignment[module_assignment.isin(best_ids)].index.tolist()
