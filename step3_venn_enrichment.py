"""
Maps: 3.1.venn.txt, 3.2.DO.txt, 3.2.GO.txt, 3.2.GO circle.txt, 3.2.KEGG.txt

  R `venn` package             -> matplotlib_venn / plain set intersection
  R `clusterProfiler::enrichGO`  -> gseapy.enrich (Enrichr GO_Biological_Process_2021 etc.)
  R `clusterProfiler::enrichKEGG`-> gseapy.enrich (Enrichr KEGG_2021_Human) or gseapy.gsea
  R `DOSE::enrichDO`             -> gseapy.enrich against a Disease Ontology / DisGeNET gene set

gseapy calls Enrichr's web API, so these functions need network access and a
gene-set library name. If you're offline, swap `run_enrichr` for a local
hypergeometric test against a downloaded .gmt file (see `local_ora` below,
which reimplements exactly what enrichGO/enrichKEGG do statistically: a
one-sided Fisher's exact / hypergeometric test per gene set, BH-corrected).
"""
from typing import Iterable

import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def venn_intersection(gene_sets: dict) -> set:
    """gene_sets: {"WGCNA": [...], "DIFF": [...]} -> intersection of all sets.
    Direct port of 3.1.venn.txt's Reduce(intersect, geneList)."""
    sets = [set(g) for g in gene_sets.values()]
    return set.intersection(*sets) if sets else set()


def run_enrichr(gene_list: Iterable[str], gene_sets: str = "GO_Biological_Process_2021",
                 organism: str = "human") -> pd.DataFrame:
    """Thin wrapper around gseapy.enrichr (needs internet access).
    gene_sets examples: 'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
    'GO_Molecular_Function_2021', 'KEGG_2021_Human', 'DisGeNET' (proxy for DO)."""
    import gseapy as gp
    enr = gp.enrichr(gene_list=list(gene_list), gene_sets=gene_sets, organism=organism, outdir=None)
    return enr.results.sort_values("Adjusted P-value")


def local_ora(gene_list: Iterable[str], gene_set_library: dict, background_size: int) -> pd.DataFrame:
    """Offline over-representation analysis identical in spirit to
    clusterProfiler::enrichGO/enrichKEGG/enrichDO: hypergeometric test per
    term, BH-adjusted p-values.

    gene_set_library: {term_name: [gene, gene, ...]}  (load from a .gmt file)
    background_size: total number of genes considered "detectable" (universe)
    """
    query = set(gene_list)
    rows = []
    for term, members in gene_set_library.items():
        members = set(members)
        overlap = query & members
        if not overlap:
            continue
        # hypergeometric survival function: P(X >= overlap) with
        # population=background_size, successes=len(members), draws=len(query)
        pval = hypergeom.sf(len(overlap) - 1, background_size, len(members), len(query))
        rows.append({
            "term": term,
            "overlap": len(overlap),
            "term_size": len(members),
            "genes": "/".join(sorted(overlap)),
            "pvalue": pval,
        })
    if not rows:
        return pd.DataFrame(columns=["term", "overlap", "term_size", "genes", "pvalue", "qvalue"])
    out = pd.DataFrame(rows).sort_values("pvalue")
    _, qval, _, _ = multipletests(out["pvalue"], method="fdr_bh")
    out["qvalue"] = qval
    return out.reset_index(drop=True)


def load_gmt(path: str) -> dict:
    """Load a .gmt gene-set file (MSigDB format: name\\tdescription\\tgene1\\tgene2...)."""
    library = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            library[parts[0]] = parts[2:]
    return library
