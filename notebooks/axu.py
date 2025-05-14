import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from rich.progress import Progress
from scipy.stats import ttest_ind, gaussian_kde, iqr, skew
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.mixture import GaussianMixture

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.misc.colors import bcolors as c

PLOIDY_RATIOS = np.array([2, 4, 8])
PEAK_TOLERANCE = 0.3 # 20% tolerance for peak validation
MIN_PEAK_PROMINENCE = 0.05 # 15% of max prominece
BIOLOGICAL_CONSTRAINTS = {
    'max_bandwidth_factor': 2,
    'min_bandwidth_factor': 1
}


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


##############################################################################
# Pre-processing
##############################################################################

def preprocess_data(data, iqr_factor=2):
    def safe_log(x):
        return np.log10(np.maximum(x, 1e-10))

    data = np.array(data, dtype=np.float64)

    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("No valid data found. (Full NaN array)")

    q1, q3 = np.quantile(data, [.25, .75])
    iqr_val = q3 - q1
    filtered = data[
        (data >= q1 - iqr_factor * iqr_val) &
        (data <= q3 + iqr_factor * iqr_val)
    ]

    # # Log-transform if skewness is too high
    # if np.abs(skew(gaussian_kde(filtered).dataset.T)) > .5:
    #     filtered = safe_log(filtered)
    #     log_transformed = True
    # else:
    #     log_transformed = False
    log_transformed = False

    # MAD standardization
    mad = np.median(np.abs(filtered - np.median(filtered)))
    if mad < 1e-6:
        mad = np.std(filtered)
    standardized = (filtered - np.median(filtered)) / mad

    def inverse_transform(x):
        x = x * mad + np.median(filtered)
        return np.power(10, x) if log_transformed else x

    return standardized, inverse_transform

##############################################################################
# Optimal bandwidth estimation and kde params
##############################################################################

def calculate_kde_score(data, bw, params):
    kde = KernelDensity(bandwidth=bw, **params).fit(data.reshape(-1, 1))
    return kde.score_samples(data.reshape(-1, 1)).mean()

def biological_bandwidth(data):
    n = len(data)
    if n < 10:
        return max(.1, np.ptp(data) / 10)

    iqr_val = iqr(data)
    std = min(np.std(data), iqr_val / 1.34)
    h_silverman = 0.9 * std * (n ** (-0.2))

    # min_bw, max_bw = .8, 2.0
    h_silverman = h_silverman * 1.25
    print(f"{c.OKGREEN}Silverman's bandwidth{c.ENDC}: {h_silverman}")
    # print(f"{c.OKGREEN}Min BW{c.ENDC}: {min_bw}, {c.OKGREEN}Max BW{c.ENDC}: {max_bw}")

    # # GridSearch with cross-validation
    # try:
    #     bw_candidates = np.linspace(min_bw, max_bw, 30)
    #
    #     best_bw = None
    #     best_score = -np.inf
    #
    #     for bw in bw_candidates:
    #         score = calculate_kde_score(data, bw, {})
    #         if score > best_score:
    #             best_score = score
    #             best_bw = bw
    #
    #     print(f"{c.OKGREEN}Optimal bandwidth{c.ENDC}: {best_bw}")
    #     return best_bw
    # except Exception as e:
    #     print(f"{c.WARNING}Error optimizing bandwidth. Using Silverman's rule.{c.ENDC}")
    #     print(f"{c.OKGREEN}Silverman's bandwidth{c.ENDC}: {h_silverman}")
    #     return h_silverman
    return h_silverman


def optimize_kde_params(data):
    params = {
        'algorithm': ['kd_tree', 'ball_tree'],
        'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'],
        'metric': ['euclidean', 'manhattan', 'chebyshev'],
    }

    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data.reshape(-1, 1))
    return grid.best_params_

##############################################################################
# Adaptative grid generation
##############################################################################

def create_ploidy_grid(data, bw, n_points=750):
    # q1, q3 = np.quantile(data, [.01, .99])
    # main_grid = np.linspace(q1, q3, int(n_points * 0.8))
    #
    # # ENhanced res arround expected ploidy peaks
    # ploidy_centers = PLOIDY_RATIOS * np.median(data)
    # for center in ploidy_centers:
    #     if center > q1 and center < q3:
    #         window = np.linspace(center - 2 * bw, center + 2 * bw, 50)
    #         main_grid = np.unique(np.concatenate([main_grid, window]))
    #
    # # Add tails with logspace
    # lower_tails = np.geomspace(max(data.min(), 1e-6), q1, 50)
    # upper_tails = np.geomspace(q3, data.max(), 50)
    #
    # return np.unique(np.concatenate([lower_tails, main_grid, upper_tails]))
    return np.linspace(data.min(), data.max(), n_points)

# def optimal_grid_size(data, bw, min_points=500, max_points=1000):
#     data_range = np.ptp(data)
#     ideal_points = int(data_range / (bw * 2))
#     return np.clip(ideal_points, min_points, max_points)

##############################################################################
# Peak detection and validation
##############################################################################

def validate_peaks(peaks, grid, inverse_transform):
    valid_peaks = []
    original_peaks = inverse_transform(grid[peaks])

    for peak in original_peaks:
        ratios = peak / (PLOIDY_RATIOS * original_peaks[0]/2)
        if np.any((ratios > 1 - PEAK_TOLERANCE) & (ratios < 1 + PEAK_TOLERANCE)):
            valid_peaks.append(peak)

    return valid_peaks

def gmm_peak_enhancement(data, n_components=2, downsample=False, upsample=False):
    down_factor = 0.5
    if len(data) < 200:
        down_factor = 0.95

    gmm = GaussianMixture(n_components=n_components)
    labels = gmm.fit_predict(data.reshape(-1, 1))

    enhanced_data = []
    for i in range(n_components):
        component_data = data[labels == i]
        if len(component_data) > len(data) * .2:
            if i == np.argmax([len(data[labels == j]) for j in range(n_components)]):
                if downsample:
                    component_data = np.random.choice(component_data, int(len(data) * down_factor), replace=False)
                enhanced_data.append(component_data)

            else:
                if upsample:
                    component_data = np.random.choice(component_data, int(len(data) * 1.5), replace=True)
                enhanced_data.append(component_data)

    return np.concatenate(enhanced_data)


# def gmm_peak_validation(data, max_components=3):
#     """Validate peaks using Gaussian Mixture Models"""
#     bic = []
#     n_components = range(1, max_components + 1)
#
#     for n in n_components:
#         gmm = GaussianMixture(n_components=n, covariance_type='spherical')
#         gmm.fit(data.reshape(-1, 1))
#         bic.append(gmm.bic(data.reshape(-1, 1)))
#
#     optimal_n = n_components[np.argmin(bic)]
#
#     # Refit with optimal components
#     gmm = GaussianMixture(n_components=optimal_n, covariance_type='spherical')
#     gmm.fit(data.reshape(-1, 1))
#
#     return {
#         'means': gmm.means_.flatten(),
#         'weights': gmm.weights_,
#         'bic': bic,
#         'n_components': optimal_n
#     }

##############################################################################
# Core analysis / Plotting
##############################################################################

def kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file, gmm_enhancement=True):
    """Enhanced 5-subplot visualization with new KDE implementation"""
    # Merge with positive data
    data1 = np.concatenate([data1, pos1]) if pos1 is not None else data1
    data2 = np.concatenate([data2, pos2]) if pos2 is not None else data2

    data1_std, inv1 = preprocess_data(data1)
    data2_std, inv2 = preprocess_data(data2)

    if gmm_enhancement and feature == 'volume_microns':
        data1_enhanced = gmm_peak_enhancement(data1_std, downsample=True)
        data2_enhanced = gmm_peak_enhancement(data2_std, downsample=True)
    else:
        data1_enhanced = data1_std
        data2_enhanced = data2_std

    # Create figure with original layout
    fig, (ax_box1, ax_box2, ax_dist, ax_rug1, ax_rug2) = plt.subplots(
        5, 1, sharex=True,
        gridspec_kw={'height_ratios': (0.2, 0.2, 1, 0.05, 0.05)},
        figsize=(10, 8)
    )

    # Bandwidth estimation
    bw1 = biological_bandwidth(data1_std if not gmm_enhancement else data1_enhanced)
    bw2 = biological_bandwidth(data2_std if not gmm_enhancement else data2_enhanced)

    # Grid generation
    grid1 = create_ploidy_grid(
        data1_std if not gmm_enhancement else data1_enhanced,
        bw1, n_points=350 if len(data1) < 300 else 550
    )
    grid2 = create_ploidy_grid(
        data2_std if not gmm_enhancement else data2_enhanced,
        bw2, n_points=350 if len(data2) < 300 else 550
    )

    # Fit KDE models
    # kde_params1 = optimize_kde_params(data1_std if not gmm_enhancement else data1_enhanced)
    # kde_params2 = optimize_kde_params(data2_std if not gmm_enhancement else data2_enhanced)

    kde_params1 = {
        'algorithm': 'kd_tree',
        'kernel': 'cosine',
        'metric': 'euclidean'
    }

    kde_params2 = {
        'algorithm': 'kd_tree',
        'kernel': 'cosine',
        'metric': 'euclidean'
    }

    print(f"{c.OKGREEN}KDE Params{c.ENDC}: {kde_params1}")

    kde1 = KernelDensity(
        bandwidth=bw1, **kde_params1
    ).fit(data1_std.reshape(-1, 1) if not gmm_enhancement else data1_enhanced.reshape(-1, 1))
    kde2 = KernelDensity(
        bandwidth=bw2, **kde_params2
    ).fit(data2_std.reshape(-1, 1) if not gmm_enhancement else data2_enhanced.reshape(-1, 1))

    # NaN handling (Removing NaNs from grid)
    idx1 = np.isnan(grid1)
    idx2 = np.isnan(grid2)

    if np.any(idx1):
        grid1 = grid1[~idx1]
    if np.any(idx2):
        grid2 = grid2[~idx2]

    # Grids back to original scale
    grid1_orig = inv1(grid1)
    grid2_orig = inv2(grid2)

    # Evaluate densities
    dens1 = np.exp(kde1.score_samples(grid1.reshape(-1, 1)))
    dens2 = np.exp(kde2.score_samples(grid2.reshape(-1, 1)))

    # Boxplots (preserved from original)
    def plot_box(ax, data, color, inv):
        original_data = inv(data)

        sns.boxplot(x=original_data, ax=ax, color=color, orient='h', width=0.6)
        ax.axvline(np.median(original_data), color=f'dark{color}', linestyle='--')
        ax.set(xticks=[], yticks=[])

    plot_box(ax_box1, data1_std, 'blue', inv1)
    plot_box(ax_box2, data2_std, 'green', inv2)

    # Enhanced KDE plot
    ax_dist.plot(grid1_orig, dens1, label=label1, color='blue')
    ax_dist.plot(grid2_orig, dens2, label=label2, color='green')

    ax_dist.fill_between(grid1_orig, dens1, alpha=0.25, color='blue')
    ax_dist.fill_between(grid2_orig, dens2, alpha=0.25, color='green')

    # Detect and Annotate peaks
    peaks1 = find_peaks(
        dens1, #prominence=MIN_PEAK_PROMINENCE * np.max(dens1),
    )[0]
    peaks2 = find_peaks(
        dens2, #prominence=MIN_PEAK_PROMINENCE * np.max(dens2),
    )[0]

    valid_peaks1 = validate_peaks(peaks1, grid1_orig, lambda x: x)
    valid_peaks2 = validate_peaks(peaks2, grid2_orig, lambda x: x)
    # valid_peaks1 = peaks1
    # valid_peaks2 = peaks2

    for p in valid_peaks1:
        ax_dist.axvline(p, color='blue', linestyle='--', alpha=0.5)
    for p in valid_peaks2:
        ax_dist.axvline(p, color='green', linestyle='--', alpha=0.5)

    # ax_dist.axvline(np.median(data1), color='blue', linestyle='--', alpha=0.5)
    # ax_dist.axvline(np.median(data2), color='green', linestyle='--', alpha=0.5)
    ax_dist.set(ylabel='Density', xlabel='', yticks=[])
    ax_dist.legend()

    # Rug plots (preserved from original)
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


def kde_combined_plot(data1, data2, label1, label2, feature, out_file, group_data):
    """Generate both normal and binned KDE plots"""

    def get_pos_data(label, feature, group_data):
        pos_label = label.replace("NEG", "POS")
        return group_data.get(pos_label, pd.DataFrame())[feature].dropna().values

    try:
        pos1 = get_pos_data(label1, feature, group_data)
    except:
        pos1 = None

    try:
        pos2 = get_pos_data(label2, feature, group_data)
    except:
        pos2 = None

    normal_out = out_file.replace(".png", "_zscore.png")
    binned_out = out_file.replace(".png", "_zscore_binned.png")

    kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, normal_out)
    # kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, binned_out)


##############################################################################
# Group Analysis Workflow
##############################################################################

def group_analysis(df, output_dir):
    """Main analysis workflow with progress tracking"""
    features = ['volume_microns', 'energy']
    if 'group_label' not in df.columns:
        df['group_label'] = df['cohort'] + '_' + df['nuclei_type'] + '_' + df['brdu_status']

    group_data = {}
    all_labels = sorted(df['group_label'].unique())
    for glab in all_labels:
        group_data[glab] = df[df['group_label'] == glab]

    valid_pairs = []
    for i in range(len(all_labels)):
        for j in range(i+1, len(all_labels)):
            g1 = all_labels[i]
            g2 = all_labels[j]
            if (('WT' in g1 and 'MYC' in g2) or ('WT' in g2 and 'MYC' in g1)):
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

                # Data extraction
                d1 = data1[feature].dropna().values
                d2 = data2[feature].dropna().values

                if len(d1) < 5 or len(d2) < 5:
                    progress.update(main_task, advance=1)
                    continue

                # Generate plots
                base_name = f"{group1}_vs_{group2}"
                out_path = os.path.join(feature_dir, f"{base_name}.png")

                try:
                    kde_combined_plot(d1, d2, group1, group2, feature, out_path, group_data)
                except Exception as e:
                    print(f"{c.FAIL}Error generating plots:{c.ENDC} {e}")
                    import traceback
                    traceback.print_exc()

                progress.update(main_task, advance=1)


##############################################################################
# Main Execution
##############################################################################
def main():
    """
    1) Collect annotated Excel files from both 'ADULT WT' and 'ADULT MYC'.
    2) Merge them into one DataFrame.
    3) Perform group analysis (NEG only, WT vs MYC) and save results to subfolders.
    """
    bas_path = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/'
    base_dirs = [
        f"{bas_path}ADULT WT",
        f"{bas_path}ADULT MYC"
    ]
    output_dir = f"{bas_path}analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    excel_files = gather_annotated_excels(base_dirs)
    print(f"Found {len(excel_files)} annotated Excel files.")

    df = load_and_merge_annotated(excel_files)
    if df.empty:
        print(f"{c.WARNING}No annotated data found. Exiting...{c.ENDC}")
        return

    group_analysis(df, output_dir)

if __name__ == "__main__":
    main()