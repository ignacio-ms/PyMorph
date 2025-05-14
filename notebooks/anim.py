from manim import *
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import ttest_ind, iqr, skew, kurtosis
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter1d
import warnings


warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Use OpenGL renderer and high quality
config.renderer = "opengl"
config.quality = "high_quality"
config.background_color = "#1E1E1E"  # dark background

import os
import sys
# Append parent directory if needed
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.misc.colors import bcolors as c

np.random.seed(0)
PLOIDY_RATIOS = np.array([2, 4, 8])
MIN_PEAK_PROMINENCE = 0.05

##############################################################################
# GLOBAL CONFIGURATION
##############################################################################
CONFIG = {
    "colors": {
        "axes": WHITE,
        "kde_curve": "#2ca02c",
        "gmm_components": "#1f77b4",
        "gmm_sum": "#FF5733",
        "peak_marker": YELLOW,
        "data_points": "#FFC300"
    },
    "durations": {
        "axes_fade_in": 1.0,
        "data_intro": 1.0,
        "kde_draw": 2.0,
        "component_intro": 1.5,
        "component_pause": 1.0,
        "sum_draw": 1.5,
        "annotation_fade": 1.0
    },
    "animation_speed": 1.0
}

# Parameters for density estimation and GMM PDF
params = {
    "bandwidth_method": "adaptive",  # Options: "silverman", "scott", "adaptive"
    "adaptive_grid": True,  # Use quantile-based adaptive grid
    "kde_grid_points": 550,
    "auto_skew_adjust": False,
    "skewness_threshold": 1.0,
    "auto_kurt_adjust": False,
    "kurtosis_threshold": 3.0,
    "bw_adjust": True,
    "bw_adjust_factor": 1.25,
    "prominence_adjust": True,
    "prominence_factor": 0.15,
    "gmm_components": "auto",  # "auto" to select between 2 and 3
    "gmm_selection_criterion": "BIC",  # Options: "BIC" or "AIC"
    "tail_trim_quantile": 0.94,
    "tail_trim_fraction": 0.1,
    "smooth_kde": False,  # If True, smooth the KDE curve (ignored when using GMM PDF)
    "smoothing_method": "savitzky_golay",  # "gaussian", "moving_average", or "savitzky_golay"
    "savgol_window": 51,
    "savgol_order": 3,
    "smoothing_sigma": 6.0,
    "smoothing_window": 100,
    "peak_prominence": "auto",
    "peak_ratio_tolerance": 0.95,
    "plot_gmm_pdf": True,  # When True, use GMM PDF as the density estimate (KDE is not drawn)
    "gmm_pdf_points": 500
}


##############################################################################
# Data Loading and Preparation Functions
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
        except Exception as e:
            print(f"{c.WARNING}Error reading {f}: {e}{c.ENDC}")
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined.dropna(subset=['nuclei_type', 'brdu_status', 'cohort'])


def preprocess_data(data, iqr_factor=2):
    data = np.array(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        raise ValueError("No valid data found. (Full NaN array)")
    q1, q3 = np.quantile(data, [.25, .75])
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
        bw *= params.get("bw_adjust_factor", 1.5)
    return bw


def create_ploidy_grid(data, bw, n_points=750, adaptive=False):
    if adaptive:
        quantiles = np.linspace(0.0, 1.0, n_points)
        grid = np.quantile(data, quantiles)
        return grid
    else:
        return np.linspace(data.min(), data.max(), n_points)


def gmm_peak_enhancement(data, n_components=None, tail_trim=True):
    X = data.reshape(-1, 1)
    if n_components is None or n_components == "auto":
        criterion = params.get("gmm_selection_criterion", "BIC").upper()
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
        gmm3 = GaussianMixture(n_components=3, random_state=0).fit(X)
        n_components = 2 if (criterion == "BIC" and gmm2.bic(X) < gmm3.bic(X)) or (
                    criterion == "AIC" and gmm2.aic(X) < gmm3.aic(X)) else 3
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


def compute_gmm_pdf_values(data_std, inv):
    X = data_std.reshape(-1, 1)
    if params.get("gmm_components", "auto") == "auto":
        criterion = params.get("gmm_selection_criterion", "BIC").upper()
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
        gmm3 = GaussianMixture(n_components=3, random_state=0).fit(X)
        n_components = 2 if (criterion == "BIC" and gmm2.bic(X) < gmm3.bic(X)) or (
                    criterion == "AIC" and gmm2.aic(X) < gmm3.aic(X)) else 3
    else:
        n_components = params.get("gmm_components")
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    n_points = params.get("gmm_pdf_points", 500)
    x_grid = np.linspace(data_std.min(), data_std.max(), n_points)
    log_probs = gmm.score_samples(x_grid.reshape(-1, 1))
    pdf = np.exp(log_probs)
    x_grid_orig = inv(x_grid)
    return x_grid_orig, pdf


def annotate_ploidy_peaks(ax, grid_orig, dens, color):
    if params.get("peak_prominence", "auto") == "auto":
        prom_val = 0.1 * (dens.max() - dens.min())
    else:
        prom_val = float(params.get("peak_prominence"))
    if params.get("prominence_adjust", False):
        prom_val *= params.get("prominence_factor", 2.0)
    peaks, _ = find_peaks(dens, prominence=prom_val)
    if len(peaks) == 0:
        return
    peak_vals = grid_orig[peaks]
    sorted_idx = np.argsort(peak_vals)
    peak_vals = peak_vals[sorted_idx]
    ploidy_peaks = {}
    if peak_vals.size > 0:
        base = peak_vals[0]
        ploidy_peaks["2N"] = base
        tol = params.get("peak_ratio_tolerance", 0.10)
        for pv in peak_vals[1:]:
            ratio = pv / base if base > 0 else np.inf
            if abs(ratio - 2.0) <= 2.0 * tol:
                ploidy_peaks["4N"] = pv
            elif abs(ratio - 4.0) <= 4.0 * tol:
                ploidy_peaks["8N"] = pv
    for label, xpos in ploidy_peaks.items():
        y_val = np.interp(xpos, grid_orig, dens)
        ax.plot(xpos, y_val, 'o', color=color)
        annotation = Tex(f"{label} ({xpos:.2f})", color=CONFIG["colors"]["peak_marker"]).scale(0.5)
        annotation.next_to(ax.coords_to_point(xpos, y_val), UP, buff=0.1)
        ax.add(annotation)


##############################################################################
# Density Plotting Functions
##############################################################################
def kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file, gmm_enhancement=True):
    """Creates side-by-side density plots for two datasets using either KDE or GMM PDF."""
    data1 = np.concatenate([data1, pos1]) if pos1 is not None else data1
    data2 = np.concatenate([data2, pos2]) if pos2 is not None else data2
    data1_std, inv1 = preprocess_data(data1)
    data2_std, inv2 = preprocess_data(data2)
    if gmm_enhancement and feature == 'volume_microns':
        data1_enhanced = gmm_peak_enhancement(data1_std, n_components=params.get("gmm_components"))
        data2_enhanced = gmm_peak_enhancement(data2_std, n_components=params.get("gmm_components"))
    else:
        data1_enhanced = data1_std
        data2_enhanced = data2_std

    # Create two Axes objects side-by-side
    axes_left = Axes(
        x_range=[data1_std.min() - 1, data1_std.max() + 1, 1],
        y_range=[0, 0.4, 0.1],
        x_length=5,
        y_length=3,
        axis_config={"stroke_color": CONFIG["colors"]["axes"], "include_numbers": True}
    )
    axes_right = Axes(
        x_range=[data2_std.min() - 1, data2_std.max() + 1, 1],
        y_range=[0, 0.4, 0.1],
        x_length=5,
        y_length=3,
        axis_config={"stroke_color": CONFIG["colors"]["axes"], "include_numbers": True}
    )
    axes_right.next_to(axes_left, RIGHT, buff=1)
    title_left = Tex("Group: " + label1, color=CONFIG["colors"]["axes"]).scale(0.7)
    title_right = Tex("Group: " + label2, color=CONFIG["colors"]["axes"]).scale(0.7)
    title_left.next_to(axes_left, UP, buff=0.2)
    title_right.next_to(axes_right, UP, buff=0.2)

    self.add(axes_left, axes_right, title_left, title_right)

    # Determine density estimates based on option:
    if params.get("plot_gmm_pdf", False):
        grid1_orig, dens1 = compute_gmm_pdf_values(data1_std if not gmm_enhancement else data1_enhanced, inv1)
        grid2_orig, dens2 = compute_gmm_pdf_values(data2_std if not gmm_enhancement else data2_enhanced, inv2)
    else:
        bw1 = select_bandwidth(data1_std if not gmm_enhancement else data1_enhanced,
                               method=params["bandwidth_method"])
        bw2 = select_bandwidth(data2_std if not gmm_enhancement else data2_enhanced,
                               method=params["bandwidth_method"])
        n_points1 = 350 if len(data1) < 300 else params.get("kde_grid_points", 550)
        n_points2 = 350 if len(data2) < 300 else params.get("kde_grid_points", 550)
        grid1 = create_ploidy_grid(data1_std if not gmm_enhancement else data1_enhanced, bw1,
                                   n_points=n_points1, adaptive=params.get("adaptive_grid", False))
        grid2 = create_ploidy_grid(data2_std if not gmm_enhancement else data2_enhanced, bw2,
                                   n_points=n_points2, adaptive=params.get("adaptive_grid", False))
        kde_params = {'algorithm': 'kd_tree', 'kernel': 'cosine', 'metric': 'euclidean'}
        kde1 = KernelDensity(bandwidth=bw1, **kde_params).fit(
            (data1_std if not gmm_enhancement else data1_enhanced).reshape(-1, 1))
        kde2 = KernelDensity(bandwidth=bw2, **kde_params).fit(
            (data2_std if not gmm_enhancement else data2_enhanced).reshape(-1, 1))
        grid1 = grid1[~np.isnan(grid1)]
        grid2 = grid2[~np.isnan(grid2)]
        grid1_orig = inv1(grid1)
        grid2_orig = inv2(grid2)
        dens1 = np.exp(kde1.score_samples(grid1.reshape(-1, 1)))
        dens2 = np.exp(kde2.score_samples(grid2.reshape(-1, 1)))
        if params.get("smooth_kde", False):
            method = params.get("smoothing_method", "gaussian")
            if method == "gaussian":
                dens1 = gaussian_filter1d(dens1, sigma=params.get("smoothing_sigma", 2.0))
                dens2 = gaussian_filter1d(dens2, sigma=params.get("smoothing_sigma", 2.0))
            elif method == "moving_average":
                window = int(params.get("smoothing_window", 5))
                kernel = np.ones(window) / window
                dens1 = np.convolve(dens1, kernel, mode='same')
                dens2 = np.convolve(dens2, kernel, mode='same')
            elif method == "savitzky_golay":
                window = int(params.get("savgol_window", 31))
                order = int(params.get("savgol_order", 3))
                window = min(window, len(dens1) - (1 - len(dens1) % 2))
                dens1 = savgol_filter(dens1, window, order)
                dens2 = savgol_filter(dens2, window, order)
    # Plot the density curves on each axis
    kde_curve_left = axes_left.plot_line_graph(grid1_orig, dens1, add_vertex_dots=False,
                                               line_color=CONFIG["colors"]["gmm_sum"] if params.get("plot_gmm_pdf",
                                                                                                    False) else
                                               CONFIG["colors"]["kde_curve"])
    kde_curve_right = axes_right.plot_line_graph(grid2_orig, dens2, add_vertex_dots=False,
                                                 line_color=CONFIG["colors"]["gmm_sum"] if params.get("plot_gmm_pdf",
                                                                                                      False) else
                                                 CONFIG["colors"]["kde_curve"])
    self.play(Create(kde_curve_left, run_time=CONFIG["durations"]["kde_draw"] * CONFIG["animation_speed"]),
              Create(kde_curve_right, run_time=CONFIG["durations"]["kde_draw"] * CONFIG["animation_speed"]))

    # Annotate peaks on each plot
    annotate_ploidy_peaks(axes_left, grid1_orig, dens1, CONFIG["colors"]["peak_marker"])
    annotate_ploidy_peaks(axes_right, grid2_orig, dens2, CONFIG["colors"]["peak_marker"])

    # Add rug plots for data points
    def add_rug(axes, data_std, inv):
        rug_lines = VGroup()
        for x in data_std:
            point = inv(np.array([x]))
            line = axes.get_vertical_line(axes.coords_to_point(x, 0.05), color=CONFIG["colors"]["data_points"],
                                          stroke_width=2)
            rug_lines.add(line)
        axes.add(rug_lines)

    add_rug(axes_left, data1_std, inv1)
    add_rug(axes_right, data2_std, inv2)

    self.wait(2 * CONFIG["animation_speed"])
    self.play(FadeOut(axes_left), FadeOut(axes_right))
    self.wait(1)


##############################################################################
# Group Analysis Workflow (Data reading and grouping)
##############################################################################
def group_analysis_workflow(bas_path):
    base_dirs = [f"{bas_path}ADULT WT", f"{bas_path}ADULT MYC"]
    excel_files = gather_annotated_excels(base_dirs)
    print(f"Found {len(excel_files)} annotated Excel files.")
    df = load_and_merge_annotated(excel_files)
    # Save a combined file for record keeping (optional)
    df.to_excel('combined_measurements.xlsx')
    # Create a group_label if not already present
    if 'group_label' not in df.columns:
        df['group_label'] = df['cohort'] + '_' + df['nuclei_type'] + '_' + df['brdu_status']
    return df


##############################################################################
# Manim Scene for GMM PDF Animation
##############################################################################
class GMMDensityScene(Scene):
    def construct(self):
        # 1. Data Loading
        # Use the same base path as in your analysis script.
        bas_path = "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/"
        df = group_analysis_workflow(bas_path)
        # For this animation, we select two groups for comparison.
        # For example: "WT_MONO_NEG" and "MYC_MONO_NEG" (adjust if needed).
        group1 = "WT_MONO_NEG"
        group2 = "MYC_MONO_NEG"
        data1 = df[df["group_label"] == group1]["volume_microns"].dropna().values
        data2 = df[df["group_label"] == group2]["volume_microns"].dropna().values

        # 2. Set up the scene title
        title = Tex("GMM PDF Density Estimation", color=WHITE).scale(0.9)
        title.to_edge(UP)
        self.play(Write(title), run_time=CONFIG["durations"]["axes_fade_in"] * CONFIG["animation_speed"])

        # 3. Create the side-by-side density plots (using GMM PDF if enabled)
        kde_combined_plot(data1, data2, group1, group2, "volume_microns", "gmm_density_animation.png", group_data=None)

        self.wait(2 * CONFIG["animation_speed"])

##############################################################################
# Main Execution (Manim will run the Scene when called)
##############################################################################
# To render, run in terminal: manim -pqh <script_name.py> GMMDensityScene