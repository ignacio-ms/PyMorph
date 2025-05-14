import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
from rich.progress import Progress
from scipy.stats import ttest_ind, iqr, skew, kurtosis
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

##############################################################################
# GLOBAL PARAMS
##############################################################################
params = {
    "bandwidth_method": "adaptive",   # Options: "silverman", "scott", "adaptive"
    "adaptive_grid": True,           # True uses quantile-based adaptive grid
    "kde_grid_points": 550,          # Number of grid points for KDE (fixed grid if adaptive_grid False)
    "auto_skew_adjust": False,
    "skewness_threshold": 1.0,
    "auto_kurt_adjust": False,
    "kurtosis_threshold": 3.0,
    "bw_adjust": True,
    "bw_adjust_factor": 1.25,
    "prominence_adjust": True,
    "prominence_factor": 0.1,
    "gmm_components": "auto",        # "auto" to select automatically between 2 and 3 components
    "gmm_selection_criterion": "BIC", # Options: "BIC" or "AIC"
    "tail_trim_quantile": 0.94,      # Quantile for tail trimming
    "tail_trim_fraction": 0.1,       # Fraction of tail points to keep
    "smooth_kde": False,             # If True, apply smoothing to KDE curve after estimation
    "smoothing_method": "savitzky_golay",  # Options: "gaussian", "moving_average", "savitzky_golay"
    "savgol_window": 51,             # Window size for Savitzky-Golay smoothing
    "savgol_order": 3,               # Polynomial order for Savitzky-Golay smoothing
    "smoothing_sigma": 6.0,          # Sigma for Gaussian smoothing
    "smoothing_window": 100,         # Window size for moving average smoothing
    "peak_prominence": "auto",       # "auto" or fixed numeric value
    "peak_ratio_tolerance": 0.95,
    "plot_gmm_pdf": True,            # If True, use GMM PDF as the density estimate (and skip KDE)
    "gmm_pdf_points": 500,           # Number of points for GMM PDF evaluation
    "equalize_ranges": False,        # Unify x-range (and y-axis) if True
}

# NEW: We store one row for each distribution in these columns:
# [Specimen_Label, K, BIC_1, BIC_2, BIC_3, 2N%, 2N#, 2NMean, 4N%, 4N#, 4NMean, 8N%, 8N#, 8NMean].
ANNOTATION_ROWS = []

##############################################################################
# Data Loading and Preparation
##############################################################################
def gather_annotated_excels(base_dir_list):
    """Recursively collect all Excel files from specified directories"""
    excel_files = []
    for bd in base_dir_list:
        for root, _, files in os.walk(bd):
            for f in files:
                if f.endswith("annotated.xlsx"):
                    # if f not in ['AEI1_dapi_annotated.xlsx', 'AEI1_001_dapi_annotated.xlsx',
                    #              'AEI2_dapi_annotated.xlsx','AEI2_001_dapi_annotated.xlsx',
                    #              'AEI2_002_dapi_annotated.xlsx','AEI2_003_dapi_annotated.xlsx']:
                    if f in ['efj832_2_004_dapi_annotated.xlsx', 'efj832_2_003_dapi_annotated.xlsx',
                             'efj832_2_002_dapi_annotated.xlsx', 'efj832_2_001_dapi_annotated.xlsx']:
                        continue
                    excel_files.append(os.path.join(root, f))
    return excel_files

def load_and_merge_annotated(excel_files):
    """Load and merge Excel files with group labeling"""
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
    1) group_counts.csv: [group_label, count]
    2) pairwise_ttests.csv: [feature, group1, group2, p_value]
    """
    if 'group_label' not in df.columns:
        df['group_label'] = df['cohort'] + '_' + df['nuclei_type'] + '_' + df['brdu_status']

    # 1) group_counts.csv
    group_counts = df['group_label'].value_counts().reset_index()
    group_counts.columns = ['group_label','count', 'mean_volume']
    group_counts['mean_volume'] = df.groupby('group_label')['volume_microns'].mean().values
    group_counts.to_csv(os.path.join(out_dir, 'group_counts.csv'), index=False)
    print(f"{c.OKGREEN}group_counts.csv generated{c.ENDC}")

    # 2) pairwise_ttests.csv
    pairs = [
        ('WT_BI_NEG','MYC_BI_NEG'),
        # ('WT_BI_NEG','MYC_MONO_NEG'),
        # ('WT_MONO_NEG','MYC_BI_NEG'),
        ('WT_MONO_NEG','MYC_MONO_NEG'),
    ]
    results = []
    feature = 'volume_microns'  # as per your example
    for (g1, g2) in pairs:
        d1 = df.loc[df['group_label']==g1, feature].dropna()
        d2 = df.loc[df['group_label']==g2, feature].dropna()
        if len(d1) > 1 and len(d2) > 1:
            _, p_val = ttest_ind(d1, d2, equal_var=False)
            results.append([feature, g1, g2, p_val])
    pairwise_df = pd.DataFrame(results, columns=['feature','group1','group2','p_value'])
    pairwise_df.to_csv(os.path.join(out_dir, 'pairwise_ttests.csv'), index=False)
    print(f"{c.OKGREEN}pairwise_ttests.csv generated{c.ENDC}")

##############################################################################
# Pre-processing
##############################################################################
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
    if label == 'MYC_MONO_NEG':
        q1, q3 = np.quantile(data, [.20, .95])
    # if label == 'MYC_BI_NEG':
    #     q1, q3 = np.quantile(data, [.20, .85])
    iqr_val = q3 - q1
    filtered = data[(data >= q1 - iqr_factor * iqr_val) & (data <= q3 + iqr_factor * iqr_val)]
    # MAD standardization
    mad = np.median(np.abs(filtered - np.median(filtered)))
    if mad < 1e-6:
        mad = np.std(filtered)
    standardized = (filtered - np.median(filtered)) / mad

    def inverse_transform(x):
        return x * mad + np.median(filtered)

    return standardized, inverse_transform

##############################################################################
# Bandwidth Selection
##############################################################################
def select_bandwidth(data, method="silverman"):
    n = len(data)
    if n < 10:
        return max(0.1, np.ptp(data) / 10)
    sigma = np.std(data)
    iqr_val = iqr(data)
    if method == "silverman":
        bw = 0.9 * min(sigma, iqr_val / 1.34) * (n ** -0.2)
    elif method == "scott":
        bw = 1.06 * sigma * (n ** -0.2)
    elif method == "adaptive":
        bw = 0.9 * min(sigma, iqr_val / 1.34) * (n ** -0.2)
    else:
        bw = 0.9 * min(sigma, iqr_val / 1.34) * (n ** -0.2)

    if params.get("bw_adjust", False):
        print(f"{c.WARNING}Adjusting bandwidth by factor{c.ENDC}: {params.get('bw_adjust_factor', 1.5)}")
        bw *= params.get('bw_adjust_factor', 1.5)
    print(f"{c.OKGREEN}Selected bandwidth{c.ENDC}: {bw:.4f}")
    return bw

##############################################################################
# Adaptive grid generation
##############################################################################
def create_ploidy_grid(data, bw, n_points=750, adaptive=False):
    if adaptive:
        quantiles = np.linspace(0.0, 1.0, n_points)
        grid = np.quantile(data, quantiles)
        return grid
    else:
        return np.linspace(data.min(), data.max(), n_points)

##############################################################################
# GMM-Based Downsampling (Tail Trimming)
##############################################################################
def gmm_peak_enhancement(data, n_components=None, label=None, tail_trim=True):
    X = data.reshape(-1, 1)
    if n_components is None or n_components == "auto":
        criterion = params.get("gmm_selection_criterion", "BIC").upper()
        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
        gmm3 = GaussianMixture(n_components=3, random_state=0).fit(X)
        gmm4 = GaussianMixture(n_components=4, random_state=0).fit(X)
        if criterion == "BIC":
            bic1, bic2, bic3 = gmm1.bic(X), gmm2.bic(X), gmm3.bic(X)
            n_components = 2 if bic2 < bic3 else 3
        elif criterion == "AIC":
            aic1, aic2, aic3, aic4 = gmm1.aic(X), gmm2.aic(X), gmm3.aic(X), gmm4.aic(X)
            n_components = 2 if aic2 < aic3 else 3 if aic3 < aic4 else 4
        else:
            n_components = 2
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

            # if label == 'MYC_BI_NEG':
            #     tq, fraction = 0.95, 0.05

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

##############################################################################
# GMM PDF Computation (returns density values)
# MODIFIED: returns (x_grid_orig, pdf, gmm, [bic1,bic2,bic3], best_n)
##############################################################################
def compute_gmm_pdf_values(data_std, inv, best_n=None):
    X = data_std.reshape(-1, 1)

    # We gather BIC for k=1..3 for real columns BIC_1,BIC_2,BIC_3
    criterion = params.get("gmm_selection_criterion", "BIC").upper()
    gm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
    gm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
    gm3 = GaussianMixture(n_components=3, random_state=0).fit(X)

    if best_n is None or best_n == "auto":
        if criterion=="BIC":
            bic1, bic2, bic3 = gm1.bic(X), gm2.bic(X), gm3.bic(X)
        else:
            bic1, bic2, bic3 = gm1.aic(X), gm2.aic(X), gm3.aic(X)

        # Decide best k
        if params.get("gmm_components","auto")=="auto":
            # pick best from gm1..gm3
            scores = [bic1,bic2,bic3]
            best_idx = np.argmin(scores)
            best_n = best_idx+1
            if best_n<1: best_n=1
        else:
            # user forced e.g. 2 or 3
            best_n = int(params["gmm_components"])

        print(f"{c.OKGREEN}[GMM PDF] Using {best_n} components{c.ENDC}")
        if best_n == 1:
            gmm = gm1
        elif best_n == 2:
            gmm = gm2
        else:
            gmm = gm3

        n_points = params.get("gmm_pdf_points", 500)
        x_grid = np.linspace(data_std.min(), data_std.max(), n_points)
        log_probs = gmm.score_samples(x_grid.reshape(-1, 1))
        pdf = np.exp(log_probs)
        x_grid_orig = inv(x_grid)
        return x_grid_orig, pdf, gmm, (bic1, bic2, bic3), best_n

    print(f"{c.OKGREEN}[GMM PDF] Using {best_n} components{c.ENDC}")
    if best_n==1:
        gmm = gm1
    elif best_n==2:
        gmm = gm2
    else:
        gmm = gm3

    n_points = params.get("gmm_pdf_points", 500)
    x_grid = np.linspace(data_std.min(), data_std.max(), n_points)
    log_probs = gmm.score_samples(x_grid.reshape(-1, 1))
    pdf = np.exp(log_probs)
    x_grid_orig = inv(x_grid)
    return x_grid_orig, pdf, gmm, (None, None, None), best_n

##############################################################################
# Peak detection and annotation
##############################################################################
def annotate_ploidy_peaks(ax, grid_orig, dens, color, data_orig):
    if params.get("peak_prominence", "auto") == "auto":
        prom_val = 0.1 * (dens.max() - dens.min())
    else:
        prom_val = float(params.get("peak_prominence"))
    if params.get("prominence_adjust", False):
        print(f"{c.WARNING}Adjusting peak prominence by factor{c.ENDC}: {params.get('prominence_factor', 2.0)}")
        prom_val *= params.get("prominence_factor", 2.0)
    print(f"{c.OKGREEN}Peak prominence{c.ENDC}: {prom_val:.4f}")
    peaks, _ = find_peaks(dens)
    if len(peaks) == 0:
        return
    peak_vals = grid_orig[peaks]
    sorted_idx = np.argsort(peak_vals)
    peak_vals = peak_vals[sorted_idx]
    labels = ["2N", "4N", "8N"]
    for i, p in enumerate(peak_vals[:3]):
        y_val = np.interp(p, grid_orig, dens)
        ax.plot(p, y_val, 'o', color=color)
        annotation = f"{labels[i]}\n({p:.2f})"
        ax.text(p, y_val * 1.05, annotation, ha='center', color=color, fontsize=10)

##############################################################################
# Core analysis / Plotting
##############################################################################
def kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file, gmm_enhancement=True):
    data1 = np.concatenate([data1, pos1]) if pos1 is not None else data1
    data2 = np.concatenate([data2, pos2]) if pos2 is not None else data2

    data1_std, inv1 = preprocess_data(data1, label1)
    data2_std, inv2 = preprocess_data(data2, label2)

    n_comp_1, n_comp_2 = params.get("gmm_components", "auto"), params.get("gmm_components", "auto")
    if label1 == 'MYC_BI_NEG':
        n_comp_1 = 2
    if label2 == 'MYC_BI_NEG':
        n_comp_2 = 2

    if gmm_enhancement and feature == 'volume_microns':
        data1_enhanced = gmm_peak_enhancement(data1_std, n_components=n_comp_1, label=label1)
        data2_enhanced = gmm_peak_enhancement(data2_std, n_components=n_comp_2, label=label2)
    else:
        data1_enhanced = data1_std
        data2_enhanced = data2_std

    fig, (ax_box1, ax_box2, ax_dist, ax_rug1, ax_rug2) = plt.subplots(
        5, 1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios': (0.2, 0.2, 1, 0.05, 0.05)},
        figsize=(10, 8)
    )

    ax_overlay = fig.add_subplot(ax_dist.get_position(), sharex=ax_dist)
    ax_overlay.tick_params(labelleft=False, labelbottom=False)
    for spine in ax_overlay.spines.values():
        spine.set_visible(False)

    # GMM-based PDF
    if params.get("plot_gmm_pdf", False):
        grid1_orig, dens1, gmm1, bic_info1, k1 = compute_gmm_pdf_values(data1_enhanced, inv1, best_n=n_comp_1)
        grid2_orig, dens2, gmm2, bic_info2, k2 = compute_gmm_pdf_values(data2_enhanced, inv2, best_n=n_comp_2)
    else:
        # if not plotting GMM, do empty
        grid1_orig, dens1, gmm1, bic_info1, k1 = np.array([]), np.array([]), None, (np.nan,np.nan,np.nan), 1
        grid2_orig, dens2, gmm2, bic_info2, k2 = np.array([]), np.array([]), None, (np.nan,np.nan,np.nan), 1

    bw1 = select_bandwidth(data1_enhanced, method=params["bandwidth_method"])
    bw2 = select_bandwidth(data2_enhanced, method=params["bandwidth_method"])
    kde_params = {'algorithm': 'kd_tree', 'kernel': 'cosine', 'metric': 'euclidean'}
    kde1 = KernelDensity(bandwidth=bw1, **kde_params).fit(data1_enhanced.reshape(-1, 1))
    kde2 = KernelDensity(bandwidth=bw2, **kde_params).fit(data2_enhanced.reshape(-1, 1))

    n_points1 = 350 if len(data1) < 300 else params.get("kde_grid_points", 550)
    n_points2 = 350 if len(data2) < 300 else params.get("kde_grid_points", 550)
    grid1_std = create_ploidy_grid(data1_enhanced, bw1, n_points=n_points1,
                                   adaptive=params.get("adaptive_grid", False))
    grid2_std = create_ploidy_grid(data2_enhanced, bw2, n_points=n_points2,
                                   adaptive=params.get("adaptive_grid", False))
    dens1_kde = np.exp(kde1.score_samples(grid1_std.reshape(-1, 1)))
    dens2_kde = np.exp(kde2.score_samples(grid2_std.reshape(-1, 1)))
    grid1_orig_kde = inv1(grid1_std)
    grid2_orig_kde = inv2(grid2_std)

    # Box
    def plot_box(ax, data, color, inv):
        original_data = inv(data)
        sns.boxplot(x=original_data, ax=ax, color=color, orient='h', width=0.6)
        ax.axvline(np.median(original_data), color=f'dark{color}', linestyle='--')
        ax.set(xticks=[], yticks=[])

    plot_box(ax_box1, data1_std, 'blue', inv1)
    plot_box(ax_box2, data2_std, 'green', inv2)

    ax_dist.plot(grid1_orig, dens1, label=label1, color='blue')
    ax_dist.fill_between(grid1_orig, dens1, alpha=0.25, color='blue')
    ax_dist.plot(grid2_orig, dens2, label=label2, color='green')
    ax_dist.fill_between(grid2_orig, dens2, alpha=0.25, color='green')

    ax_dist.plot(grid1_orig_kde, dens1_kde, label=label1, color='blue', linestyle='--')
    ax_dist.fill_between(grid1_orig_kde, dens1_kde, alpha=0.25, color='blue')
    ax_dist.plot(grid2_orig_kde, dens2_kde, label=label2, color='green', linestyle='--')
    ax_dist.fill_between(grid2_orig_kde, dens2_kde, alpha=0.25, color='green')

    data1_orig = inv1(data1_std)
    data2_orig = inv2(data2_std)

    wt_valley_indices = find_peaks(-dens1)[0]
    if wt_valley_indices.size > 0:
        wt_valleys = np.sort(grid1_orig[wt_valley_indices])
        for x in wt_valleys:
            ax_dist.axvline(x, color='blue', linestyle='--', linewidth=1)
    else:
        wt_valleys = np.array([])

    wt_region_bounds = [data1_orig.min()] + wt_valleys.tolist() + [data1_orig.max()]
    for i in range(len(wt_region_bounds) - 1):
        lb = wt_region_bounds[i]
        ub = wt_region_bounds[i + 1]
        if i < len(wt_region_bounds) - 2:
            mask = (data1_orig >= lb) & (data1_orig < ub)
        else:
            mask = (data1_orig >= lb) & (data1_orig <= ub)
        pct = 100 * np.sum(mask) / len(data1_orig)
        center = (lb + ub) / 2
        y_min, y_max = ax_dist.get_ylim()
        ax_dist.text(center, y_min + 0.05 * (y_max - y_min), f"{pct:.1f}%", color='blue',
                     ha='center', va='bottom', fontsize=10, alpha=0.8)

    myc_valley_indices = find_peaks(-dens2)[0]
    if myc_valley_indices.size > 0:
        myc_valleys = np.sort(grid2_orig[myc_valley_indices])
        for x in myc_valleys:
            ax_dist.axvline(x, color='green', linestyle='--', linewidth=1)
    else:
        myc_valleys = np.array([])
    myc_region_bounds = [data2_orig.min()] + myc_valleys.tolist() + [data2_orig.max()]
    for i in range(len(myc_region_bounds) - 1):
        lb = myc_region_bounds[i]
        ub = myc_region_bounds[i + 1]
        if i < len(myc_region_bounds) - 2:
            mask = (data2_orig >= lb) & (data2_orig < ub)
        else:
            mask = (data2_orig >= lb) & (data2_orig <= ub)
        pct = 100 * np.sum(mask) / len(data2_orig)
        center = (lb + ub) / 2
        y_min, y_max = ax_dist.get_ylim()
        ax_dist.text(center, y_min + 0.10 * (y_max - y_min), f"{pct:.1f}%", color='green',
                     ha='center', va='bottom', fontsize=10, alpha=0.8)

    annotate_ploidy_peaks(ax_dist, grid1_orig, dens1, 'blue', data1_std)
    annotate_ploidy_peaks(ax_dist, grid2_orig, dens2, 'green', data2_std)

    ax_dist.set(ylabel='Density', xlabel='', yticks=[])
    ax_dist.legend()
    ax_dist.patch.set_alpha(0)
    ax_overlay.set(ylabel='Density', xlabel='', yticks=[])
    ax_overlay.legend()
    ax_overlay.patch.set_alpha(0)

    def plot_rug(ax, data, pos_data, color, inv):
        sns.rugplot(inv(data), ax=ax, color=color, alpha=0.3, height=1)
        if pos_data is not None and len(pos_data) > 0:
            filtered = np.array([p for p in pos_data if p <= np.max(inv(data))])
            sns.rugplot(filtered, ax=ax, color='red', height=1.5)
        ax.set(yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plot_rug(ax_rug1, data1_std, pos1, 'blue', inv1)
    plot_rug(ax_rug2, data2_std, pos2, 'green', inv2)

    plt.suptitle(
        f"{label1} (n={len(data1)}) vs {label2} (n={len(data2)})"
        f"\n{feature.replace('_', ' ').title()} Distribution"
        f"\np-value: {ttest_ind(data1, data2, equal_var=False)[1]:.1e} (t-test)"
    )
    plt.savefig(out_file)
    print(f"{c.OKGREEN}Saved plot to{c.ENDC} {out_file}")
    plt.close()

    # NEW: gather real peak-based intervals for 2N,4N,8N
    # We'll replicate the logic from "annotate_ploidy_peaks" to find up to 3 peaks for each distribution.
    # Then define intervals from consecutive peaks => measure data portion => real stats.

    def compute_ploidy_stats(grid, dens, raw_data):
        # detect up to 3 peaks => 2N,4N,8N
        peaks,_ = find_peaks(dens)
        if len(peaks)==0:
            return (np.nan,)*9  # no peaks => no stats
        peak_vals = grid[peaks]
        peak_vals = np.sort(peak_vals)[:3]  # up to 3
        # define intervals e.g. [lowest, midpoint(peak1,peak2)] => 2N region
        # we do up to 3 intervals.
        intervals = []
        sorted_peaks = list(peak_vals)
        # boundaries: [min, mid(1,2), mid(2,3), max]
        cuts = []
        if len(sorted_peaks)>=2:
            cuts.append(0.5*(sorted_peaks[0]+sorted_peaks[1]))
        if len(sorted_peaks)>=3:
            cuts.append(0.5*(sorted_peaks[1]+sorted_peaks[2]))
        boundaries = [raw_data.min()] + cuts + [raw_data.max()]
        # measure each region
        def region_stats(lb,ub):
            mask=(raw_data>=lb)&(raw_data<ub)
            # last region inclusive
            if ub==raw_data.max():
                mask=(raw_data>=lb)&(raw_data<=ub)
            pct=100*np.sum(mask)/len(raw_data)
            n_ = np.sum(mask)
            mean_ = raw_data[mask].mean() if n_>0 else np.nan
            return (pct,n_,mean_)

        # up to 3 intervals => 2N,4N,8N
        region_out = []
        for i in range(3):
            if i < len(boundaries)-1:
                stats_ = region_stats(boundaries[i], boundaries[i+1])
            else:
                stats_ = (0.0,0.0,np.nan)
            region_out.append(stats_)

        # region_out is [ (2N% ,2N#, 2NMean), (4N%..), (8N%..) ]
        # flatten them
        twoN_ = region_out[0]
        fourN_ = region_out[1] if len(region_out)>=2 else (0.0,0.0,np.nan)
        eightN_= region_out[2] if len(region_out)>=3 else (0.0,0.0,np.nan)
        return twoN_+fourN_+eightN_ # => (2N%,2N#,2NMean, 4N%,4N#,4NMean, 8N%,8N#,8NMean)

    # compute for data1, data2
    if len(grid1_orig)>0 and len(dens1)>0:
        data1_ploidy=compute_ploidy_stats(grid1_orig,dens1,inv1(data1_std))
    else:
        data1_ploidy=(np.nan,)*9
    if len(grid2_orig)>0 and len(dens2)>0:
        data2_ploidy=compute_ploidy_stats(grid2_orig,dens2,inv2(data2_std))
    else:
        data2_ploidy=(np.nan,)*9

    # gather BIC_1,BIC_2,BIC_3 from bic_info1, k from k1
    # keep in mind we used partial, so we have bic1,bic2,bic3
    (bic1_1,bic2_1,bic3_1)=bic_info1
    (bic1_2,bic2_2,bic3_2)=bic_info2

    # if label1 or label2 in {WT_BI_NEG, MYC_BI_NEG, WT_MONO_NEG, MYC_MONO_NEG}
    # store row => "Specimen_Label","K","BIC_1","BIC_2","BIC_3","2N%","2N#","2NMean",...
    SPECIMENS = {"WT_BI_NEG","MYC_BI_NEG","WT_MONO_NEG","MYC_MONO_NEG"}
    # row for label1
    if label1.upper() in SPECIMENS:
        row = {
           "Specimen_Label": label1,
           "K": k1,
           "BIC_1": bic1_1, "BIC_2": bic2_1, "BIC_3": bic3_1,
           "2N%": data1_ploidy[0],  "2N#": data1_ploidy[1],  "2NMean": data1_ploidy[2],
           "4N%": data1_ploidy[3],  "4N#": data1_ploidy[4],  "4NMean": data1_ploidy[5],
           "8N%": data1_ploidy[6],  "8N#": data1_ploidy[7],  "8NMean": data1_ploidy[8],
        }
        ANNOTATION_ROWS.append(row)

    # row for label2
    if label2.upper() in SPECIMENS:
        row = {
           "Specimen_Label": label2,
           "K": k2,
           "BIC_1": bic1_2, "BIC_2": bic2_2, "BIC_3": bic3_2,
           "2N%": data2_ploidy[0],  "2N#": data2_ploidy[1],  "2NMean": data2_ploidy[2],
           "4N%": data2_ploidy[3],  "4N#": data2_ploidy[4],  "4NMean": data2_ploidy[5],
           "8N%": data2_ploidy[6],  "8N#": data2_ploidy[7],  "8NMean": data2_ploidy[8],
        }
        ANNOTATION_ROWS.append(row)


def kde_combined_plot(data1, data2, label1, label2, feature, out_file, group_data):
    """Generate both normal and binned density plots."""
    def get_pos_data(label, feature, group_data):
        pos_label = label.replace("NEG", "POS")
        return group_data.get(pos_label, pd.DataFrame())[feature].dropna().values

    try:
        pos1 = get_pos_data(label1, feature, group_data)
    except Exception:
        pos1 = None
    try:
        pos2 = get_pos_data(label2, feature, group_data)
    except Exception:
        pos2 = None

    normal_out = out_file.replace(".png", "_zscore.png")
    binned_out = out_file.replace(".png", "_zscore_binned.png")

    # 1) Normal Z-score plot
    kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, normal_out)
    # 2) Optionally, generate a binned version with different settings if desired:
    #    (We skip to keep code unchanged.)

##############################################################################
# Group Analysis Workflow
##############################################################################
def group_analysis(df, output_dir):
    """Main analysis workflow with progress tracking"""
    features = ['volume_microns', 'energy', 'integrated_intensity']
    if 'group_label' not in df.columns:
        df['group_label'] = df['cohort'] + '_' + df['nuclei_type'] + '_' + df['brdu_status']

    group_data = {}
    all_labels = sorted(df['group_label'].unique())
    for glab in all_labels:
        group_data[glab] = df[df['group_label'] == glab]

    valid_pairs = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            g1 = all_labels[i]
            g2 = all_labels[j]
            if (('WT' in g1 and 'MYC' in g2) or ('WT' in g2 and 'MYC' in g1)) \
                    and (('BI' in g1 and 'BI' in g2) or ('MONO' in g1 and 'MONO' in g2)):
                if ('NEG' in g1 and 'NEG' in g2):
                    valid_pairs.append((g1, g2))

    with Progress() as progress:
        main_task = progress.add_task("[cyan]Analyzing groups...", total=len(valid_pairs) * len(features))
        for group1, group2 in valid_pairs:
            data1 = group_data.get(group1, pd.DataFrame())
            data2 = group_data.get(group2, pd.DataFrame())
            if data1.empty or data2.empty:
                progress.update(main_task, advance=len(features))
                print(f"{c.WARNING}No data found for{c.ENDC} {group1} vs {group2}.")
                continue

            for feature in features:
                feature_dir = os.path.join(output_dir, feature.replace('_', ''))
                os.makedirs(feature_dir, exist_ok=True)
                print(f"{c.OKBLUE}Processing {feature} for {group1} vs {group2}...{c.ENDC}")
                d1 = data1[feature].dropna().values
                d2 = data2[feature].dropna().values
                if len(d1) < 5 or len(d2) < 5:
                    progress.update(main_task, advance=1)
                    continue

                base_name = f"{group1}_vs_{group2}"
                out_path = os.path.join(feature_dir, f"{base_name}.png")
                try:
                    kde_combined_plot(d1, d2, group1, group2, feature, out_path, group_data)
                except Exception as e:
                    print(f"{c.FAIL}Error generating plots:{c.ENDC} {e}")
                    import traceback
                    traceback.print_exc()
                progress.update(main_task, advance=1)

    # NEW: after finishing, we create annotation_ploidy.xlsx from ANNOTATION_ROWS
    if ANNOTATION_ROWS:
        out_file = os.path.join(output_dir, "annotation_ploidy.xlsx")
        df_annot = pd.DataFrame(ANNOTATION_ROWS)
        # Only keep the four relevant distributions
        keep_labels = {"WT_BI_NEG","MYC_BI_NEG","WT_MONO_NEG","MYC_MONO_NEG"}
        df_annot = df_annot[df_annot["Specimen_Label"].str.upper().isin(keep_labels)]
        df_annot.to_excel(out_file, index=False)
        print(f"Saved annotation data to {out_file}")

##############################################################################
# Main Execution
##############################################################################
def main():
    """
    1) Collect annotated Excel files from both 'ADULT WT' and 'ADULT MYC'.
    2) Merge them into one DataFrame.
    3) Perform group analysis (NEG only, WT vs MYC) and save results to subfolders.
    """
    bas_path = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/Nacho new/'
    base_dirs = [f"{bas_path}ADULT WT", f"{bas_path}ADULT MYC"]
    output_dir = f"{bas_path}analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    excel_files = gather_annotated_excels(base_dirs)
    print(f"Found {len(excel_files)} annotated Excel files.")
    df = load_and_merge_annotated(excel_files)
    df.to_excel('combined_measurements.xlsx')
    if df.empty:
        print(f"{c.WARNING}No annotated data found. Exiting...{c.ENDC}")
        return

    generate_summary_files(df, output_dir)
    group_analysis(df, output_dir)

if __name__ == "__main__":
    main()
