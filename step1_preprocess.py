"""
Maps: 1.Extraction Matrix.txt (R / limma::avereps)

R version:
    ann = read.table(GPL annotation)[, c(probe_id, symbol_col)]
    GSE = read.table(series_matrix)
    merge GSE with ann by probe ID
    avereps()                 # average rows that share the same gene symbol
    write GSE55092.txt

Python equivalent below. `avereps` (average expression of duplicate probes
mapping to the same gene symbol) has no 1:1 sklearn/pandas function, so it's
reimplemented as a groupby-mean, which is exactly what avereps does for the
default (unweighted) case.
"""
import pandas as pd


def load_platform_annotation(ann_path: str, probe_col: int = 0, symbol_col: int = 10) -> pd.DataFrame:
    """GPL annotation file: probe ID -> gene symbol. `symbol_col` defaults to
    column index 10 (R's kcol=11, 1-indexed -> 10 in 0-indexed pandas)."""
    ann = pd.read_csv(ann_path, sep="\t", header=None, dtype=str, low_memory=False)
    ann = ann.iloc[:, [probe_col, symbol_col]]
    ann.columns = ["probe_id", "symbol"]
    return ann


def load_series_matrix(gse_path: str) -> pd.DataFrame:
    """GEO series_matrix.txt with probes in the first column, samples as
    the remaining columns (already stripped of the GEO header/footer)."""
    gse = pd.read_csv(gse_path, sep="\t", header=0, dtype=str, low_memory=False)
    gse = gse.rename(columns={gse.columns[0]: "probe_id"})
    return gse


def collapse_probes_to_genes(gse: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """Merge probe->symbol mapping, coerce to numeric, and average duplicate
    probes per gene symbol (avereps equivalent). Returns genes x samples."""
    merged = gse.merge(ann, on="probe_id", how="inner")
    merged = merged.drop(columns=["probe_id"])
    sample_cols = [c for c in merged.columns if c != "symbol"]
    merged[sample_cols] = merged[sample_cols].apply(pd.to_numeric, errors="coerce")
    expr = merged.groupby("symbol", as_index=True)[sample_cols].mean()
    expr = expr.dropna(how="all")
    return expr  # rows = gene symbol, columns = samples


def extract_matrix(gse_path: str, ann_path: str, out_path: str,
                    probe_col: int = 0, symbol_col: int = 10) -> pd.DataFrame:
    ann = load_platform_annotation(ann_path, probe_col, symbol_col)
    gse = load_series_matrix(gse_path)
    expr = collapse_probes_to_genes(gse, ann)
    expr.to_csv(out_path, sep="\t")
    return expr


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Collapse a GEO series matrix to gene-level expression.")
    p.add_argument("--gse", required=True)
    p.add_argument("--ann", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    extract_matrix(args.gse, args.ann, args.out)
