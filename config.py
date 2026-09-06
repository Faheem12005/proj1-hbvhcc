"""
Shared configuration for the HBV-HCC pipeline.

Defaults to the imported project files in this workspace so the real GEO
matrices are used automatically when present. Set HBV_HCC_DATA or HBV_HCC_OUT
if you want to override the locations.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("HBV_HCC_DATA", BASE_DIR)
OUT_DIR = os.environ.get("HBV_HCC_OUT", os.path.join(BASE_DIR, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

# Training / validation cohorts used in the paper
TRAIN_GSE = "GSE121248"   # 37 HBV vs 70 HBV-HCC
VALID_GSE = "GSE55092"    # 91 HBV vs 49 HBV-HCC

# DEG thresholds (2_af_Diff_wgcna.txt: |log2FC|>1 & FDR<0.05)
LOGFC_CUTOFF = 1.0
FDR_CUTOFF = 0.05

# The 3 feature genes the paper converged on. Your pipeline should
# re-derive candidates from data; this is only the paper's published
# result, kept here so downstream steps have a sane default to test with.
PAPER_FEATURE_GENES = ["HHIP", "CXCL14", "CDHR2"]

RANDOM_STATE = 42
