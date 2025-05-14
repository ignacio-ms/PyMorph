#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
from rich.progress import Progress
from scipy.stats import ttest_ind, iqr, skew, kurtosis, ks_2samp
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import warnings
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.misc.colors import bcolors as c

np.random.seed(0)
PLOIDY_RATIOS = np.array([2, 4, 8])
MIN_PEAK_PROMINENCE = 0.05

# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================
params = {
    "bandwidth_method": "adaptive",  # Options: "silverman", "scott", "adaptive"
    "adaptive_grid": True,  # Use quantile-based adaptive grid
    "kde_grid_points": 550,  # Number of grid points for KDE
    "auto_skew_adjust": False,
    "skewness_threshold": 1.0,
    "auto_kurt_adjust": False,
    "kurtosis_threshold": 3.0,
    "bw_adjust": True,
    "bw_adjust_factor": 1.25,
    "prominence_adjust": True,
    "prominence_factor": 0.1,
    "gmm_components": 3,  # Fixed to 3 components here
    "gmm_selection_criterion": "BIC",  # Options: "BIC" or "AIC"
    "tail_trim_quantile": 0.94,
    "tail_trim_fraction": 0.1,
    "smooth_kde": False,
    "smoothing_method": "savitzky_golay",  # Options: "gaussian", "moving_average", "savitzky_golay"
    "savgol_window": 51,
    "savgol_order": 3,
    "smoothing_sigma": 6.0,
    "smoothing_window": 100,
    "peak_prominence": "auto",
    "peak_ratio_tolerance": 0.95,
    "plot_gmm_pdf": True,  # If True, use GMM PDF as density estimator
    "gmm_pdf_points": 500,
    "equalize_ranges": False,  # Not used in the normalized version
    "SHOW_RUGS": False  # Toggle to show/hide rug plots
}

# Colors for groups (WT: gray, MYC: dark blue)
WT_COLOR = "#797979"
MYC_COLOR = "#0532ff"


# =============================================================================
# Data Loading and Summary Functions
# =============================================================================
def gather_annotated_excels(base_dir_list):
    """Recursively collect all Excel files from specified directories."""
    excel_files = []
    for bd in base_dir_list:
        for root, _, files in os.walk(bd):
            for f in files:
                if f.endswith("annotated.xlsx"):
                    # if f not in ['AEI1_dapi_annotated.xlsx', 'AEI1_001_dapi_annotated.xlsx', 'AEI2_dapi_annotated.xlsx', 'AEI2_001_dapi_annotated.xlsx', 'AEI2_002_dapi_annotated.xlsx', 'AEI2_003_dapi_annotated.xlsx']:
                    if f in ['efj832_2_004_dapi_annotated.xlsx', 'efj832_2_003_dapi_annotated.xlsx',
                             'efj832_2_002_dapi_annotated.xlsx', 'efj832_2_001_dapi_annotated.xlsx']:
                        continue
                    excel_files.append(os.path.join(root, f))
    return excel_files


def load_and_merge_annotated(excel_files):
    """Load and merge Excel files with group labeling."""
    dfs = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Loading files...", total=len(excel_files))
        for f in excel_files:
            try:
                df = pd.read_excel(f)
                if 'cohort' not in df.columns:
                    if "ADULT WT" in f.upper() or "WT" in f.upper():
                        df['cohort'] = "WT"
                    elif "ADULT MYC" in f.upper() or "MYC" in f.upper():
                        df['cohort'] = "MYC"
                    else:
                        df['cohort'] = "Unknown"
                df['source_file'] = os.path.basename(f)
                dfs.append(df)
                progress.update(task, advance=1)
            except Exception as e:
                print(f"{c.WARNING}Error reading {f}: {e}{c.ENDC}")
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined.dropna(subset=['nuclei_type', 'brdu_status', 'cohort'])


def generate_summary_files(df, out_dir):
    """
    Generates two summary CSV files:
      1. group_counts.csv: [group_label, count]
      2. pairwise_ttests.csv: [feature, group1, group2, p_value]
    """
    if 'group_label' not in df.columns:
        df['group_label'] = df['cohort'] + '_' + df['nuclei_type'] + '_' + df['brdu_status']
    group_counts = df['group_label'].value_counts().reset_index()
    group_counts.columns = ['group_label', 'count']
    group_counts.to_csv(os.path.join(out_dir, 'group_counts.csv'), index=False)
    print(f"{c.OKGREEN}group_counts.csv generated{c.ENDC}")
    pairs = [
        ('WT_BI_NEG', 'MYC_BI_NEG'),
        # ('WT_BI_NEG', 'MYC_MONO_NEG'),
        # ('WT_MONO_NEG', 'MYC_BI_NEG'),
        ('WT_MONO_NEG', 'MYC_MONO_NEG'),
    ]
    results = []
    feature = 'volume_microns'
    for (g1, g2) in pairs:
        d1 = df.loc[df['group_label'] == g1, feature].dropna()
        d2 = df.loc[df['group_label'] == g2, feature].dropna()
        if len(d1) > 1 and len(d2) > 1:
            _, p_val = ks_2samp(d1, d2)
            results.append([feature, g1, g2, p_val])
    pairwise_df = pd.DataFrame(results, columns=['feature', 'group1', 'group2', 'p_value'])
    pairwise_df.to_csv(os.path.join(out_dir, 'pairwise_ks_test.csv'), index=False)
    print(f"{c.OKGREEN}pairwise_ks_test.csv generated{c.ENDC}")


# =============================================================================
# Pre-processing for Density Estimation
# =============================================================================
def preprocess_data(data, label, iqr_factor=2):
    data = np.array(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("No valid data found. (Full NaN array)")
    q1, q3 = np.quantile(data, [.1, .9])
    if label == 'WT_MONO_NEG' or label == 'WT_BI_NEG':
        q1, q3 = np.quantile(data, [.1, .75])
    # if label == 'WT_BI_NEG':
    #     q1, q3 = np.quantile(data, [.25, .75])
    # if label == 'MYC_MONO_NEG':
    #     q1, q3 = np.quantile(data, [.20, .95])
    # if label == 'MYC_BI_NEG':
    #     q1, q3 = np.quantile(data, [.20, .85])
    iqr_val = q3 - q1
    filtered = data[(data >= q1 - iqr_factor * iqr_val) & (data <= q3 + iqr_factor * iqr_val)]
    mad = np.median(np.abs(filtered - np.median(filtered)))
    if mad < 1e-6:
        mad = np.std(filtered)
    standardized = (filtered - np.median(filtered)) / mad

    def inverse_transform(x):
        return x * mad + np.median(filtered)

    return standardized, inverse_transform


# =============================================================================
# Bandwidth Selection and Adaptive Grid
# =============================================================================
def select_bandwidth(data, method="silverman"):
    n = len(data)
    if n < 10:
        return max(0.1, np.ptp(data) / 10)
    sigma = np.std(data)
    iqr_val = iqr(data)
    if method == "silverman":
        bw = 0.9 * min(sigma, iqr_val / 1.34) * (n ** (-0.2))
    elif method == "scott":
        bw = 1.06 * sigma * (n ** (-0.2))
    elif method == "adaptive":
        bw = 0.9 * min(sigma, iqr_val / 1.34) * (n ** (-0.2))
    else:
        bw = 0.9 * min(sigma, iqr_val / 1.34) * (n ** (-0.2))
    if params.get("bw_adjust", False):
        print(f"{c.WARNING}Adjusting bandwidth by factor{c.ENDC}: {params.get('bw_adjust_factor', 1.5)}")
        bw *= params.get("bw_adjust_factor", 1.5)
    print(f"{c.OKGREEN}Selected bandwidth{c.ENDC}: {bw:.4f}")
    return bw


def create_ploidy_grid(data, bw, n_points=750, adaptive=False):
    if adaptive:
        quantiles = np.linspace(0.0, 1.0, n_points)
        grid = np.quantile(data, quantiles)
        return grid
    else:
        return np.linspace(data.min(), data.max(), n_points)


# =============================================================================
# GMM Functions
# =============================================================================
def gmm_peak_enhancement(data, n_components=None, tail_trim=True):
    X = data.reshape(-1, 1)
    if n_components is None or n_components == "auto":
        criterion = params.get("gmm_selection_criterion", "BIC").upper()
        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
        gmm3 = GaussianMixture(n_components=3, random_state=0).fit(X)
        bic1, bic2, bic3 = gmm1.bic(X), gmm2.bic(X), gmm3.bic(X)
        n_components = 2 if bic2 < bic3 else 3
    print(f"{c.OKGREEN}Selected GMM components{c.ENDC}: {n_components}")
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    labels = gmm.fit_predict(X)
    trimmed_data = []
    for comp in range(n_components):
        comp_data = data[labels == comp]
        if tail_trim and comp_data.size > 0:
            tq = params.get("tail_trim_quantile", 0.99)
            threshold = np.quantile(comp_data, tq)
            fraction = params.get("tail_trim_fraction", 0.1)
            comp_keep = comp_data[comp_data <= threshold]
            comp_tail = comp_data[comp_data > threshold]
            if comp_tail.size > 0:
                keep_indices = np.random.rand(comp_tail.size) < fraction
                comp_tail = comp_tail[keep_indices]
            trimmed_comp = np.concatenate([comp_keep, comp_tail])
            trimmed_data.append(trimmed_comp)
        else:
            trimmed_data.append(comp_data)
    return np.concatenate(trimmed_data)


def compute_gmm_pdf_values(data_std, inv, n_components=None):
    X = data_std.reshape(-1, 1)
    if n_components == "auto" or n_components is None:
        criterion = params.get("gmm_selection_criterion", "BIC").upper()
        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
        gmm3 = GaussianMixture(n_components=3, random_state=0).fit(X)
        bic1, bic2, bic3 = gmm1.bic(X), gmm2.bic(X), gmm3.bic(X)
        n_components = 2 if bic2 < bic3 else 3

    print(f"{c.OKGREEN}[GMM PDF] Using {n_components} components{c.ENDC}")
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    n_points = params.get("gmm_pdf_points", 500)
    x_grid = np.linspace(data_std.min(), data_std.max(), n_points)
    log_probs = gmm.score_samples(x_grid.reshape(-1, 1))
    pdf = np.exp(log_probs)
    x_grid_orig = inv(x_grid)
    return x_grid_orig, pdf, gmm


# =============================================================================
# Peak detection (plot markers only)
# =============================================================================
def annotate_ploidy_peaks(ax, grid_orig, dens, color):
    if params.get("peak_prominence", "auto") == "auto":
        prom_val = 0.1 * (dens.max() - dens.min())
    else:
        prom_val = float(params.get("peak_prominence"))
    if params.get("prominence_adjust", False):
        prom_val *= params.get("prominence_factor", 2.0)
    peaks, _ = find_peaks(dens) # , prominence=prom_val
    if len(peaks) == 0:
        return None
    peak_vals = grid_orig[peaks]
    sorted_idx = np.argsort(peak_vals)
    peak_vals = peak_vals[sorted_idx]
    for xpos in peak_vals:
        y_val = np.interp(xpos, grid_orig, dens)
        ax.plot(xpos, y_val, 'o', color=color)
    return peak_vals


# =============================================================================
# Core analysis / Plotting Function
# =============================================================================
def kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file, gmm_enhancement=True,
                       show_rugs=params["SHOW_RUGS"]):
    """
    Produce a density plot with a separate rug axis below.
    If comparing WT vs MYC, the MYC curve is scaled so its 2N peak aligns with the WT 2N peak.
    Rug plots are drawn on the lower axis using sns.rugplot.
    """
    # Combine data with any additional positive (POS) values if provided
    data1 = np.concatenate([data1, pos1]) if pos1 is not None else data1
    data2 = np.concatenate([data2, pos2]) if pos2 is not None else data2

    data1_std, inv1 = preprocess_data(data1, label1)
    data2_std, inv2 = preprocess_data(data2, label2)

    n_comp_1, n_comp_2 = params.get("gmm_components", "auto"), params.get("gmm_components", "auto")
    if label1 == 'MYC_BI_NEG' or label1 == 'MYC_MONO_NEG':
        n_comp_1 = 2
    if label2 == 'MYC_BI_NEG' or label2 == 'MYC_MONO_NEG':
        n_comp_2 = 2

    if gmm_enhancement and feature == 'volume_microns':
        data1_enhanced = gmm_peak_enhancement(data1_std, n_components=n_comp_1)
        data2_enhanced = gmm_peak_enhancement(data2_std, n_components=n_comp_2)
    else:
        data1_enhanced = data1_std
        data2_enhanced = data2_std

    # Compute density estimates
    if params.get("plot_gmm_pdf", False):
        grid1_orig, dens1, gmm1 = compute_gmm_pdf_values(data1_std if not gmm_enhancement else data1_enhanced, inv1, n_components=n_comp_1)
        grid2_orig, dens2, gmm2 = compute_gmm_pdf_values(data2_std if not gmm_enhancement else data2_enhanced, inv2, n_components=n_comp_2)
    else:
        bw1 = select_bandwidth(data1_std if not gmm_enhancement else data1_enhanced, method=params["bandwidth_method"])
        bw2 = select_bandwidth(data2_std if not gmm_enhancement else data2_enhanced, method=params["bandwidth_method"])
        kde1 = KernelDensity(bandwidth=bw1, algorithm='kd_tree', kernel='cosine', metric='euclidean').fit(
            (data1_std if not gmm_enhancement else data1_enhanced).reshape(-1, 1))
        kde2 = KernelDensity(bandwidth=bw2, algorithm='kd_tree', kernel='cosine', metric='euclidean').fit(
            (data2_std if not gmm_enhancement else data2_enhanced).reshape(-1, 1))
        grid1 = create_ploidy_grid(data1_std if not gmm_enhancement else data1_enhanced, bw1,
                                   n_points=params.get("kde_grid_points", 550),
                                   adaptive=params.get("adaptive_grid", False))
        grid2 = create_ploidy_grid(data2_std if not gmm_enhancement else data2_enhanced, bw2,
                                   n_points=params.get("kde_grid_points", 550),
                                   adaptive=params.get("adaptive_grid", False))
        grid1 = grid1[~np.isnan(grid1)]
        grid2 = grid2[~np.isnan(grid2)]
        grid1_orig = inv1(grid1)
        grid2_orig = inv2(grid2)
        dens1 = np.exp(kde1.score_samples(grid1.reshape(-1, 1)))
        dens2 = np.exp(kde2.score_samples(grid2.reshape(-1, 1)))

    # Determine which group is WT and which is MYC by checking label strings
    if "WT" in label1.upper():
        wt_grid, wt_dens, wt_inv = grid1_orig, dens1, inv1
        myc_grid, myc_dens, myc_inv = grid2_orig, dens2, inv2
        wt_color = WT_COLOR
        myc_color = MYC_COLOR
    elif "WT" in label2.upper():
        wt_grid, wt_dens, wt_inv = grid2_orig, dens2, inv2
        myc_grid, myc_dens, myc_inv = grid1_orig, dens1, inv1
        wt_color = WT_COLOR
        myc_color = MYC_COLOR
    else:
        wt_grid, wt_dens, wt_inv = grid1_orig, dens1, inv1
        myc_grid, myc_dens, myc_inv = grid2_orig, dens2, inv2
        wt_color = "gray"
        myc_color = "darkblue"

    # Normalize MYC curve: align its 2N peak to WT 2N peak
    wt_peaks = find_peaks(wt_dens, prominence=MIN_PEAK_PROMINENCE * np.max(wt_dens))[0]
    myc_peaks = find_peaks(myc_dens, prominence=MIN_PEAK_PROMINENCE * np.max(myc_dens))[0]
    if wt_peaks.size > 0 and myc_peaks.size > 0:
        wt_2N = np.sort(wt_grid[wt_peaks])[0]
        myc_2N = np.sort(myc_grid[myc_peaks])[0]
        scale_factor = wt_2N / myc_2N
        myc_grid_aligned = myc_grid * scale_factor
    else:
        myc_grid_aligned = myc_grid
        scale_factor = 1.0

    # Create a figure with two rows: main density axis on top and a rug axis below
    fig, (ax_main) = plt.subplots(
        1, 1, sharex=True, figsize=(10, 6),
        # gridspec_kw={'height_ratios': [1, 0.05, 0.05]}
    )

    # Plot density curves on the main axis (no legends or extra labels)
    ax_main.plot(wt_grid, wt_dens, color=wt_color, linewidth=2)
    ax_main.fill_between(wt_grid, wt_dens, alpha=0.25, color=wt_color)
    ax_main.plot(myc_grid_aligned, myc_dens, color=myc_color, linewidth=2)
    ax_main.fill_between(myc_grid_aligned, myc_dens, alpha=0.25, color=myc_color)

    xmin, xmax = ax_main.get_xlim()
    xticks = np.arange(np.floor(xmin/50)*50, np.ceil(xmax/50)*50, 50)
    ax_main.set_xticks(xticks)
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    # for spine in ax_main.spines.values():
    #     spine.set_visible(False)

    # Plot rug plots on the rug axis using sns.rugplot.
    # For NEG data, use the original data; for POS, use red.
    # WT NEG rug:
    # if params.get("SHOW_RUGS", False):
    #     # sns.rugplot(wt_inv(data1_std), ax=ax_rug_wt, color=wt_color, height=1, alpha=0.3)
    #     # # MYC NEG rug (scale by scale_factor)
    #     # sns.rugplot(myc_inv(data2_std) * scale_factor, ax=ax_rug_myc, color=myc_color, height=1, alpha=0.3)
    #     for x in wt_inv(data1_std):
    #         ax_rug_wt.axvline(x, color=wt_color, linewidth=1.2, alpha=0.2)
    #     for x in myc_inv(data2_std):
    #         ax_rug_myc.axvline(x, color=myc_color, linewidth=1.2, alpha=0.2)

    # For POS data (retrieved from group_data)
    def get_pos_data(label, feature, group_data):
        pos_label = label.replace("NEG", "POS")
        try:
            return group_data.get(pos_label, pd.DataFrame())[feature].dropna().values
        except Exception:
            return None

#     ax_rug_wt.set_xticks([])
#     ax_rug_myc.set_xticks([])
#     ax_rug_wt.set_yticks([])
#     ax_rug_myc.set_yticks([])
#     for spine in ax_rug_wt.spines.values():
#         spine.set_visible(False)
#     for spine in ax_rug_myc.spines.values():
#         spine.set_visible(False)

    # Remove any extra labels from both axes
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    # ax_rug_wt.set_xlabel("")
    # ax_rug_myc.set_xlabel("")

    x_limits = ax_main.get_xlim()

    # Save the figure with high quality and _rug.png suffix
    if not out_file.endswith("_rug.png") and params.get("SHOW_RUGS", False):
        out_file = os.path.splitext(out_file)[0] + "_rug.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"{c.OKGREEN}Saved plot to {out_file}{c.ENDC}")
    plt.close()
    return x_limits


def kde_combined_plot(data1, data2, label1, label2, feature, out_file, group_data):
    """Wrapper to call kde_multi_subplots."""

    def get_pos_data(label, feature, group_data):
        pos_label = label.replace("NEG", "POS")
        try:
            return group_data.get(pos_label, pd.DataFrame())[feature].dropna().values
        except Exception:
            return None

    pos1 = get_pos_data(label1, feature, group_data)
    pos2 = get_pos_data(label2, feature, group_data)
    # Append _rug.png to filename if not already

    x_lims = kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file,
                       gmm_enhancement=params.get("plot_gmm_pdf", False))

    return x_lims


# =============================================================================
# Group Analysis Workflow
# =============================================================================
def group_analysis(df, output_dir):
    features = ['volume_microns', 'energy', 'integrated_intensity'] #, 'energy', 'integrated_intensity']
    if 'group_label' not in df.columns:
        df['group_label'] = df['cohort'] + '_' + df['nuclei_type'] + '_' + df['brdu_status']
    group_data = {glab: df[df['group_label'] == glab] for glab in sorted(df['group_label'].unique())}
    valid_pairs = []
    for i in range(len(group_data)):
        for j in range(i + 1, len(group_data)):
            g1 = list(group_data.keys())[i]
            g2 = list(group_data.keys())[j]
            if (("WT" in g1.upper() and "MYC" in g2.upper()) or ("WT" in g2.upper() and "MYC" in g1.upper())):
                if ("NEG" in g1.upper() and "NEG" in g2.upper()):
                    valid_pairs.append((g1, g2))

    with Progress() as progress:
        main_task = progress.add_task("[cyan]Analyzing groups...", total=len(valid_pairs) * len(features))

        def pick_color(group_label):
            return "#797979" if "WT" in group_label.upper() else "#0532ff"

        for group1, group2 in valid_pairs:
            data1 = group_data.get(group1, pd.DataFrame())
            data2 = group_data.get(group2, pd.DataFrame())
            if data1.empty or data2.empty:
                progress.update(main_task, advance=len(features))
                print(f"{c.WARNING}No data found for {group1} vs {group2}.{c.ENDC}")
                continue

            for feature in features:
                feature_dir = os.path.join(output_dir, feature.replace("_", ""))
                os.makedirs(feature_dir, exist_ok=True)
                print(f"{c.OKBLUE}Processing {feature} for {group1} vs {group2}...{c.ENDC}")

                d1 = data1[feature].dropna().values
                d2 = data2[feature].dropna().values
                if len(d1) < 5 or len(d2) < 5:
                    progress.update(main_task, advance=1)
                    continue

                base_name = f"{group1}_vs_{group2}"
                out_path = os.path.join(feature_dir, f"{base_name}.png")

                # 1) Generate the main PDF/density plot and grab its x-limits:
                try:
                    x_limits = kde_combined_plot(d1, d2, group1, group2, feature, out_path, group_data)
                except Exception as e:
                    print(f"{c.FAIL}Error generating plot for {group1} vs {group2}:{c.ENDC} {e}")
                    progress.update(main_task, advance=1)
                    continue

                d1, inv1 = preprocess_data(d1, group1)
                d2, inv2 = preprocess_data(d2, group2)
                d1, d2 = inv1(d1), inv2(d2)

                # 2) Boxplot for group1 (same x-limits as PDF)
                color1 = pick_color(group1)
                fig, ax = plt.subplots(figsize=(8, 1))
                sns.boxplot(x=d1, color=color1, orient='h', ax=ax)

                # Remove any labels/titles:
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_yticks([])

                # Force same x-range:
                ax.set_xlim(x_limits)

                # Tick marks at multiples of 50, but no numeric labels:
                xmin, xmax = x_limits
                tick_min = np.floor(xmin / 50) * 50
                tick_max = np.ceil(xmax / 50) * 50
                ax.set_xticks(np.arange(tick_min, tick_max + 1, 50))
                ax.set_xticklabels([])

                boxplot_path1 = out_path.replace(".png", f"_{group1}_box.png")
                plt.savefig(boxplot_path1, dpi=300, bbox_inches='tight')
                plt.close(fig)

                # 3) Boxplot for group2 (same x-limits)
                color2 = pick_color(group2)
                fig, ax = plt.subplots(figsize=(8, 1))
                sns.boxplot(x=d2, color=color2, orient='h', ax=ax)

                # Remove any labels/titles:
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_yticks([])

                # Force same x-range:
                ax.set_xlim(x_limits)

                # Tick marks at multiples of 50, but no numeric labels:
                xmin, xmax = x_limits
                tick_min = np.floor(xmin / 50) * 50
                tick_max = np.ceil(xmax / 50) * 50
                ax.set_xticks(np.arange(tick_min, tick_max + 1, 50))
                ax.set_xticklabels([])

                boxplot_path2 = out_path.replace(".png", f"_{group2}_box.png")
                plt.savefig(boxplot_path2, dpi=300, bbox_inches='tight')
                plt.close(fig)

                progress.update(main_task, advance=1)


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """
    1) Collect annotated Excel files from both 'ADULT WT' and 'ADULT MYC'.
    2) Merge them into one DataFrame.
    3) Generate summary CSV files.
    4) Perform group analysis (NEG only, WT vs MYC) and save density plots with separate rug axis.
    """
    bas_path = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/'
    # bas_path = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/'
    base_dirs = [f"{bas_path}ADULT WT", f"{bas_path}ADULT MYC"]
    output_dir = f"{bas_path}final_results"
    os.makedirs(output_dir, exist_ok=True)
    excel_files = gather_annotated_excels(base_dirs)
    print(f"Found {len(excel_files)} annotated Excel files.")
    df = load_and_merge_annotated(excel_files)
    df.to_excel('combined_measurements.xlsx', index=False)
    if df.empty:
        print(f"{c.WARNING}No annotated data found. Exiting...{c.ENDC}")
        return
    generate_summary_files(df, output_dir)
    group_analysis(df, output_dir)


if __name__ == "__main__":
    main()