#!/usr/bin/env python
import sys
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from rich.progress import Progress
from scipy.stats import iqr

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.misc.colors import bcolors as c

np.random.seed(0)

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
params = {
    "bandwidth_method": "adaptive",
    "adaptive_grid": True,
    "kde_grid_points": 550,
    "bw_adjust": True,
    "bw_adjust_factor": 1.25,
    # You can set "gmm_components": 1 to force a single component,
    # or "auto" to automatically pick among (1,2,3).
    "gmm_components": "auto",
    "gmm_selection_criterion": "BIC",
    "tail_trim_quantile": 0.94,
    "tail_trim_fraction": 0.1,
    "peak_prominence": "auto",
    "prominence_adjust": True,
    "prominence_factor": 0.1,
    "plot_gmm_pdf": True,
    "gmm_pdf_points": 500,
    "equalize_ranges": False,
}

TARGETS = [
    "AEI1_MONO_NEG", "AEI1_BI_NEG",
    "AEI2_MONO_NEG", "AEI2_BI_NEG",
    "AEI_KI67_MONO_NEG", "AEI_KI67_BI_NEG",
    "EFJ806_KI67_MONO_NEG", "EFJ806_KI67_BI_NEG",
    "EFJ831_MONO_NEG", "EFJ831_BI_NEG",
    "EFJ832_MONO_NEG", "EFJ832_BI_NEG",
]

FEATURES = ['volume_microns', 'integrated_intensity', 'energy'] # could also include 'energy' if you like

# We'll store final per-specimen stats (2N% etc.) in a global dictionary keyed by feature.
PLOIDY_RESULTS = {feat: [] for feat in FEATURES}

# Where we save final plots & xlsx
OUT_BASE = (
    "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/"
    "LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/single_curve_plots"
)

# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------
def gather_excels(base_dirs):
    excel_files = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith("annotated.xlsx"):
                    # if f not in ['AEI1_dapi_annotated.xlsx', 'AEI1_001_dapi_annotated.xlsx', 'AEI2_dapi_annotated.xlsx',
                    #              'AEI2_001_dapi_annotated.xlsx', 'AEI2_002_dapi_annotated.xlsx',
                    #              'AEI2_003_dapi_annotated.xlsx']:

                    if f in ['efj832_2_004_dapi_annotated.xlsx', 'efj832_2_003_dapi_annotated.xlsx',
                             'efj832_2_002_dapi_annotated.xlsx', 'efj832_2_001_dapi_annotated.xlsx']:
                        continue
                    excel_files.append(os.path.join(root, f))
    return excel_files

def load_and_merge_annotated(files):
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        if 'cohort' not in df:
            if 'WT' in f.upper():
                df['cohort'] = 'WT'
            elif 'MYC' in f.upper():
                df['cohort'] = 'MYC'
            else:
                df['cohort'] = 'Unknown'
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(data):
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    q1, q3 = np.percentile(data, [20, 75])
    iqr_val = q3 - q1
    filtered = data[(data >= q1 - 2 * iqr_val) & (data <= q3 + 2 * iqr_val)]
    mad = np.median(np.abs(filtered - np.median(filtered)))
    if mad < 1e-6:
        mad = np.std(filtered)
    standardized = (filtered - np.median(filtered)) / mad
    return standardized, lambda x: x * mad + np.median(filtered)

def select_bw(data_std):
    n = len(data_std)
    sigma = np.std(data_std)
    iq = iqr(data_std)
    bw = 0.9 * min(sigma, iq / 1.34) * n ** -0.2
    return bw * params['bw_adjust_factor']

# ---------------------------------------------------------------------------
# GMM logic
# ---------------------------------------------------------------------------
def compute_gmm_info(data_std):
    """
    Try GMM for n_components in (1,2,3) if params['gmm_components'] == 'auto'.
    Return best_k plus the BIC scores for each k=1..3.
    If user forced 'gmm_components': 1 => best_k=1, do one GMM.
    """
    forced_k = params.get("gmm_components", "auto")
    X = data_std.reshape(-1, 1)

    # If user forced a single GMM:
    if isinstance(forced_k, int) and forced_k == 1:
        # Just do k=1
        bic_scores = [GaussianMixture(n_components=1, random_state=0).fit(X).bic(X)]
        return 1, {f"BIC_{1}": bic_scores[0], "BIC_2": np.nan, "BIC_3": np.nan}

    # Otherwise gather BIC or AIC for k=1..3
    crit = params["gmm_selection_criterion"].upper()  # BIC or AIC
    scores = []
    for k in [1,2,3]:
        gm = GaussianMixture(n_components=k, random_state=0).fit(X)
        if crit == "BIC":
            val = gm.bic(X)
        else:
            val = gm.aic(X)
        scores.append(val)

    # Store them in dict
    info = {f"{crit}_{k}": sc for k, sc in zip([1,2,3], scores)}
    # pick best k
    best_k = np.argmin(scores) + 1  # index plus 1
    # If forced but numeric, e.g. 2 or 3 => override
    if isinstance(forced_k, int) and forced_k in [1,2,3]:
        best_k = forced_k

    return best_k, info

def build_gmm_pdf(data_std, inv, best_k):
    """
    Build final GMM PDF with best_k components, no tail trimming here, just PDF.
    """
    X = data_std.reshape(-1, 1)
    gmm = GaussianMixture(n_components=best_k, random_state=0).fit(X)
    x_grid = np.linspace(data_std.min(), data_std.max(), params["gmm_pdf_points"])
    pdf = np.exp(gmm.score_samples(x_grid.reshape(-1, 1)))
    return inv(x_grid), pdf

# If you want tail trimming (like you had in your old code),
# we can do so in a function that picks best_k but also re-fits
# with trimming. Omitted here for brevity.

# ---------------------------------------------------------------------------
# Summaries: 2N%, 2NMean, 2N#, etc.
# ---------------------------------------------------------------------------
def annotate_and_collect(ax, grid, pdf, data, label, feature, best_k, bic_dict):
    """
    Identify up to 3 peaks => 2N,4N,8N. Then store columns:
     - 2N%,2N#,2NMean, 4N*, 8N*
     - K = best_k
     - BIC_1, BIC_2, BIC_3
    in PLOIDY_RESULTS[feature].
    """
    data = np.array(data)
    # detect peaks
    prom_val = 0.1 * (pdf.max() - pdf.min()) * 0.01
    peaks, _ = find_peaks(pdf, prominence=prom_val)
    # label them
    peak_labels = ["2N","4N","8N"]
    # store them
    peak_positions = []
    for i, pidx in enumerate(peaks[:3]):
        x = grid[pidx]
        y = pdf[pidx]
        peak_positions.append(x)
        ax.plot(x, y, 'ro')
        ax.text(x, y+0.01, f"{peak_labels[i]}\n({x:.1f})", ha='center', color='red', fontsize=9)

    # define boundaries from these peak positions
    sorted_peaks = np.sort(peak_positions)
    cuts = []
    if len(sorted_peaks) >= 2:
        cut1 = 0.5*(sorted_peaks[0] + sorted_peaks[1])
        cuts.append(cut1)
    if len(sorted_peaks) >= 3:
        cut2 = 0.5*(sorted_peaks[1] + sorted_peaks[2])
        cuts.append(cut2)

    boundaries = [data.min()] + cuts + [data.max()]

    # We'll store the row as a dict
    row_dict = {
        "Specimen_Label": label,
        "K": best_k
    }
    # incorporate BIC_1, BIC_2, BIC_3 from bic_dict
    # For safety, we handle either BIC or AIC, but let's just store them as is
    for key, val in bic_dict.items():
        # e.g. 'BIC_1', 'BIC_2', 'BIC_3'
        row_dict[key] = val

    # For each peak label, fill in %/mean/# if we have an interval
    for i, pklabel in enumerate(peak_labels):
        if i < len(boundaries)-1:
            lb = boundaries[i]
            ub = boundaries[i+1]
            # data in [lb, ub]
            mask = (data >= lb) & (data < ub) if i < len(boundaries)-2 else (data >= lb) & (data <= ub)
            row_dict[f"{pklabel}%"] = 100.0 * np.sum(mask) / len(data)
            row_dict[f"{pklabel}#"] = np.sum(mask)
            # pk mean
            if np.sum(mask) > 0:
                row_dict[f"{pklabel}Mean"] = data[mask].mean()
            else:
                row_dict[f"{pklabel}Mean"] = np.nan
        else:
            # no interval => 0 or nan
            row_dict[f"{pklabel}%"] = 0.0
            row_dict[f"{pklabel}#"] = 0
            row_dict[f"{pklabel}Mean"] = np.nan

    row_dict["Mean"] = np.mean(data)
    row_dict["Median"] = np.median(data)
    row_dict["Std"] = np.std(data)
    # add row_dict to PLOIDY_RESULTS[feature]
    PLOIDY_RESULTS[feature].append(row_dict)

def plot_curve(data, label, feature, out_path):
    """
    - Preprocess data
    - GMM pick best_k (unless forced)
    - Collect BIC_1,BIC_2,BIC_3
    - Plot GMM PDF, detect 2N,4N,8N => store results in PLOIDY_RESULTS
    """
    data_std, inv = preprocess(data)
    # pick best_k, gather BIC_x
    best_k, bic_info = compute_gmm_info(data_std)
    # build GMM PDF
    if params["plot_gmm_pdf"]:
        x_gmm, pdf_gmm = build_gmm_pdf(data_std, inv, best_k)
    else:
        x_gmm, pdf_gmm = np.array([]), np.array([])

    # KDE
    bw = select_bw(data_std)
    grid_std = np.linspace(data_std.min(), data_std.max(), params["kde_grid_points"])
    pdf_kde = np.exp(KernelDensity(bandwidth=bw).fit(data_std.reshape(-1,1)).score_samples(grid_std.reshape(-1,1)))
    x_kde = inv(grid_std)

    # plot
    fig, ax = plt.subplots(figsize=(10,4))
    if len(x_gmm) > 0:
        ax.plot(x_gmm, pdf_gmm, label='GMM', color='blue')
        ax.fill_between(x_gmm, pdf_gmm, alpha=0.3, color='blue')
        # annotate & collect
        annotate_and_collect(ax, x_gmm, pdf_gmm, inv(data_std), label, feature, best_k, bic_info)

    ax.plot(x_kde, pdf_kde, '--', color='black', label='KDE')
    sns.rugplot(inv(data_std), ax=ax, height=0.05, color='gray', alpha=0.5)
    ax.set_title(label)
    ax.set_yticks([])
    if len(x_kde) > 0:
        xticks = np.arange(0, np.max(x_kde)+1, 50)
        ax.set_xticks(xticks)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_violin(data, label, feature, out_path, global_min, global_max):
    """
    Creates a single violin plot for the entire array `data`.
    The y-limit is set from global_min to global_max so that
    all mice share the same y-range.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 6))
    # We'll just place them at x=label or something:
    # simpler is to supply a dummy x array
    df_plot = pd.DataFrame({"group":[label]*len(data), "value":data})
    sns.violinplot(x="group", y="value", data=df_plot, color='lightblue', cut=0, ax=ax)
    sns.swarmplot(x="group", y="value", data=df_plot, color='black', alpha=0.5, ax=ax, size=3)
    plt.axhline(np.mean(data), color='red', linestyle='--', label='Mean')
    plt.text(0, np.mean(data)+0.1, f"Mean: {np.mean(data):.2f}", color='red', fontsize=10)
    ax.set_ylim([global_min, global_max])
    ax.set_title(f"{label} - {feature} Violin")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def extract_specimen_prefix(filename):
    """
    As in your original code:
    parse the name and decide the prefix
    """
    name = filename.upper()
    for prefix in ["AEI1", "AEI2", "AEI_KI67", "EFJ806_KI67", "EFJ831", "EFJ832"]:
        if prefix in name:
            return prefix
    return filename.split("_")[0].upper()

def main():
    base_dirs = [
        "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/"
        "RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/ADULT MYC",
        "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/"
        "RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/ADULT WT"
    ]
    out_base = OUT_BASE
    excel_files = gather_excels(base_dirs)
    df = load_and_merge_annotated(excel_files)

    df['group_label'] = df['cohort'].str.upper() + '_' + df['nuclei_type'].str.upper() + '_' + df['brdu_status'].str.upper()
    df['specimen'] = df['source_file'].apply(extract_specimen_prefix)

    feature_ranges = {}
    for feat in FEATURES:
        col_data = df[feat].dropna()
        if len(col_data) > 0:
            # feature_ranges[feat] = (col_data.min(), col_data.max())
            if feat == 'volume_microns':
                feature_ranges[feat] = (-10, 700)
            elif feat == 'integrated_intensity' or feat == 'energy':
                feature_ranges[feat] = (
                    np.percentile(col_data, .5),
                    np.percentile(col_data, 98)
                )
                print(f"Feature range {feat}: {feature_ranges[feat]}")
        else:
            feature_ranges[feat] = (0, 1)  # fallback

    merged = df.groupby(['specimen','group_label'])

    for (specimen, label), group in merged:
        key = f"{specimen}_{label.split('_',1)[1]}"
        print(f"Processing {key}...")
        if key.upper() not in TARGETS:
            continue
        for feat in FEATURES:
            try:
                values = group[feat].dropna().values
                if len(values)<5:
                    continue
                subdir = os.path.join(out_base, key)
                out_path = os.path.join(subdir, f"{feat}.png")
                print(f"Generating {out_path}...")
                plot_curve(values, key, feat, out_path)

                (ymin, ymax) = feature_ranges[feat]
                violin_out = out_path.replace(".png", "_violin.png")
                print(f"Generating violin => {violin_out} ...")
                values_std, inv = preprocess(values)
                plot_violin(inv(values_std), key, feat, violin_out, ymin, ymax)
            except Exception as e:
                print(f"Error processing {key} for {feat}: {e}")
                continue

    # Write final results
    for feat in FEATURES:
        results_df = pd.DataFrame(PLOIDY_RESULTS[feat])
        out_xlsx = os.path.join(out_base, f"{feat}_ploidy.xlsx")
        results_df.to_excel(out_xlsx, index=False)
        print(f"Saved {feat} ploidy results to {out_xlsx}")

if __name__=="__main__":
    main()
