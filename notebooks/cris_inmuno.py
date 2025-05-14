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
    "gmm_components": 'auto',  # Fixed to 3 components here
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
    "SHOW_RUGS": True,  # Toggle to show/hide rug plots
    'plot_hist': False,
    'downsample': True
}

# Colors for groups (WT: gray, MYC: dark blue)
WT_COLOR = "#797979"
SH_COLOR = "#0532ff"


# =============================================================================
# Pre-processing for Density Estimation
# =============================================================================
def preprocess_data(data, label, iqr_factor=2):
    data = np.array(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("No valid data found. (Full NaN array)")
    q1, q3 = np.quantile(data, [.1, .9])

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
        n_components = 1 if (bic1 < bic2 and bic1 < bic3) else 2 if bic2 < bic3 else 3

    try:
        print(f"{c.OKGREEN}[GMM]{c.ENDC} BIC1: {bic1}, BIC2: {bic2}, BIC3: {bic3}")
    except NameError:
        pass
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
        n_components = 1 if (bic1 < bic2 and bic1 < bic3) else 2 if bic2 < bic3 else 3

    try:
        print(f"{c.OKGREEN}[GMM]{c.ENDC} BIC1: {bic1}, BIC2: {bic2}, BIC3: {bic3}")
    except NameError:
        pass
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
        ax.text(xpos, y_val, f"{xpos:.2f}", color=color, fontsize=8,
                ha='center', va='bottom', alpha=0.8)
    return peak_vals


# =============================================================================
# Core analysis / Plotting Function
# =============================================================================
def kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file, gmm_enhancement=True):
    data1 = np.concatenate([data1, pos1]) if pos1 is not None else data1
    data2 = np.concatenate([data2, pos2]) if pos2 is not None else data2

    data1_std, inv1 = preprocess_data(data1, label1)
    data2_std, inv2 = preprocess_data(data2, label2)

    n_comp_1, n_comp_2 = 'auto', 'auto'
    if label1 == 'WT':
        n_comp_1 = 2

    if label2 == 'SH':
        n_comp_2 = 2

    if gmm_enhancement:
        data1_enhanced = gmm_peak_enhancement(data1_std, n_components=n_comp_1)
        data2_enhanced = gmm_peak_enhancement(data2_std, n_components=n_comp_2)
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
        grid1_orig, dens1, gmm1 = compute_gmm_pdf_values(data1_enhanced, inv1, n_components=n_comp_1)
        grid2_orig, dens2, gmm2 = compute_gmm_pdf_values(data2_enhanced, inv2, n_components=n_comp_2)
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
    if params["plot_hist"]:
        ax_dist.hist(inv1(data1_std), bins=100, density=False, alpha=0.3, color='blue', edgecolor='black', linewidth=0.5)

    ax_dist.plot(grid2_orig, dens2, label=label2, color='green')
    ax_dist.fill_between(grid2_orig, dens2, alpha=0.25, color='green')
    if params["plot_hist"]:
        ax_dist.hist(inv2(data2_std), bins=100, density=False, alpha=0.3, color='green', edgecolor='black', linewidth=0.5)

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

    annotate_ploidy_peaks(ax_dist, grid1_orig, dens1, 'blue')
    annotate_ploidy_peaks(ax_dist, grid2_orig, dens2, 'green')

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


def kde_combined_plot(data1, data2, label1, label2, feature, out_file):
    """Wrapper to call kde_multi_subplots."""
    pos1 = None
    pos2 = None

    if not params['downsample']:
        x_lims = kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file,
                       gmm_enhancement=params.get("plot_gmm_pdf", False))

    else:
        for i in range(10):
            # Downsample data1 and data2. Both data1 and data2 should be the same size.
            # This is to ensure that the KDE plots are comparable.
            down_size = len(data1) if len(data1) < len(data2) else len(data2)
            data1_sample = np.random.choice(data1, size=down_size, replace=False)
            data2_sample = np.random.choice(data2, size=down_size, replace=False)
            # Call kde_multi_subplots with the downsampled data
            out_file_aux = out_file.replace(".png", f"_{i}.png")
            kde_multi_subplots(data1_sample, data2_sample, pos1, pos2, label1, label2, feature, out_file_aux,
                               gmm_enhancement=params.get("plot_gmm_pdf", False))

    return x_lims


# =============================================================================
# Group Analysis Workflow
# =============================================================================
def group_analysis(df, output_dir):
    features = ['IntDen', 'Mean'] # RawIntDen
    def pick_color(group_label):
        return "#797979" if "WT" in group_label.upper() else "#0532ff"

    data1 = df[df['cohort'] == 'WT']
    data2 = df[df['cohort'] == 'SH']

    group1, group2 = 'WT', 'SH'

    for feature in features:
        feature_dir = os.path.join(output_dir, feature.replace("_", ""))
        os.makedirs(feature_dir, exist_ok=True)
        print(f"{c.OKBLUE}Processing {feature} for {group1} vs {group2}...{c.ENDC}")

        d1 = data1[feature].dropna().values
        d2 = data2[feature].dropna().values
        if len(d1) < 5 or len(d2) < 5:
            continue

        base_name = f"{group1}_vs_{group2}"
        out_path = os.path.join(feature_dir, f"{base_name}.png")

        # 1) Generate the main PDF/density plot and grab its x-limits:
        try:
            x_limits = kde_combined_plot(d1, d2, group1, group2, feature, out_path)
        except Exception as e:
            print(f"{c.FAIL}Error generating plot for {group1} vs {group2}:{c.ENDC} {e}")
            import traceback
            traceback.print_exc()
            continue

        # d1, inv1 = preprocess_data(d1, group1)
        # d2, inv2 = preprocess_data(d2, group2)
        # d1, d2 = inv1(d1), inv2(d2)
        #
        # # 2) Boxplot for group1 (same x-limits as PDF)
        # color1 = pick_color(group1)
        # fig, ax = plt.subplots(figsize=(8, 1))
        # sns.boxplot(x=d1, color=color1, orient='h', ax=ax)
        #
        # # Remove any labels/titles:
        # ax.set_title("")
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # ax.set_yticks([])
        #
        # # Force same x-range:
        # ax.set_xlim(x_limits)
        #
        # # Tick marks at multiples of 50, but no numeric labels:
        # xmin, xmax = x_limits
        # tick_min = np.floor(xmin / 50) * 50
        # tick_max = np.ceil(xmax / 50) * 50
        # ax.set_xticks(np.arange(tick_min, tick_max + 1, 50))
        # ax.set_xticklabels([])
        #
        # boxplot_path1 = out_path.replace(".png", f"_{group1}_box.png")
        # plt.savefig(boxplot_path1, dpi=300, bbox_inches='tight')
        # plt.close(fig)
        #
        # # 3) Boxplot for group2 (same x-limits)
        # color2 = pick_color(group2)
        # fig, ax = plt.subplots(figsize=(8, 1))
        # sns.boxplot(x=d2, color=color2, orient='h', ax=ax)
        #
        # # Remove any labels/titles:
        # ax.set_title("")
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # ax.set_yticks([])
        #
        # # Force same x-range:
        # ax.set_xlim(x_limits)
        #
        # # Tick marks at multiples of 50, but no numeric labels:
        # xmin, xmax = x_limits
        # tick_min = np.floor(xmin / 50) * 50
        # tick_max = np.ceil(xmax / 50) * 50
        # ax.set_xticks(np.arange(tick_min, tick_max + 1, 50))
        # ax.set_xticklabels([])
        #
        # boxplot_path2 = out_path.replace(".png", f"_{group2}_box.png")
        # plt.savefig(boxplot_path2, dpi=300, bbox_inches='tight')
        # plt.close(fig)


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
    bas_path = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/ExtraInmuno/'

    input_dir = f"{bas_path}P1 immuno.xlsx"
    output_dir = f"{bas_path}results/"
    os.makedirs(output_dir, exist_ok=True)

    excel_wt = pd.read_excel(input_dir, sheet_name=1)
    excel_wt['cohort'] = 'WT'


    excel_sh = pd.read_excel(input_dir, sheet_name=3)
    excel_sh['cohort'] = 'SH'

    df = pd.concat([excel_wt, excel_sh], ignore_index=True)
    df = df.dropna(subset=['IntDen'])

    output_excel_path = os.path.join(output_dir, "merged_data.xlsx")
    df.to_excel(output_excel_path, index=False)

    if df.empty:
        print(f"{c.WARNING}No annotated data found. Exiting...{c.ENDC}")
        return

    group_analysis(df, output_dir)


if __name__ == "__main__":
    main()