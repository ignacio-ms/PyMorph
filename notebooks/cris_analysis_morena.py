"""
analysis.py

Merges data from a single annotated Excel file (with no header and exactly 4 columns)
and performs group analysis for two specific comparisons:
   - Adult WT_Mono_NEG vs MYC_Mono_NEG
   - Adult WT_Bi_NEG   vs MYC_Bi_NEG

For the numeric feature (volume in microns):
  - Creates a subfolder (VolumeMu) under a parent output folder.
  - Produces violin plots (with group means annotated).
  - Produces multi-subplot KDE plots using the original KDE interpolation logic,
    with x-axis ticks added. Rug plots display NEG points (blue/green) and, if available,
    corresponding POS points in red.
"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rich.progress import Progress
from scipy.stats import ttest_ind, wasserstein_distance, ks_2samp

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Set paths for input and output (modify as needed)
DATA_FILE = "/path/to/your/data.xlsx"       # Excel file with no header (4 columns)
OUTPUT_PARENT = "/path/to/your/output_parent"  # Parent folder for results

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))
from util.misc.colors import bcolors as c

##############################################################################
# 1. Read and Convert Excel Data
##############################################################################

def read_convert_excel(data_file):
    """
    Reads the Excel file (with no header and 4 columns) where:
      - Column 1: Adult WT_Mono_NEG
      - Column 2: Adult WT_Bi_NEG
      - Column 3: MYC_Mono_NEG
      - Column 4: MYC_Bi_NEG

    Returns a DataFrame in long form with columns:
       'volume_microns', 'group_label', 'cohort', 'nuclei_type', 'brdu_status'.
    """
    # Read without header
    df_raw = pd.read_excel(data_file, header=None)
    df_raw.columns = ["Adult WT_Mono_NEG", "Adult WT_Bi_NEG", "MYC_Mono_NEG", "MYC_Bi_NEG"]
    # Melt the DataFrame
    df_long = df_raw.melt(var_name="group_label", value_name="volume_microns")
    # Define mapping for our groups
    mapping = {
        "Adult WT_Mono_NEG": ("Adult WT", "Mono", "NEG"),
        "Adult WT_Bi_NEG": ("Adult WT", "Bi", "NEG"),
        "MYC_Mono_NEG": ("MYC", "Mono", "NEG"),
        "MYC_Bi_NEG": ("MYC", "Bi", "NEG")
    }
    def assign_parts(label):
        return mapping.get(label, ("Unknown", "Unknown", "Unknown"))
    df_long[["cohort", "nuclei_type", "brdu_status"]] = df_long["group_label"].apply(lambda x: pd.Series(assign_parts(x)))
    return df_long

##############################################################################
# 2. Plotting Helpers (unchanged from your provided code)
##############################################################################

def zscore_by_first_group(data1, data2):
    mu = np.mean(data1)
    sigma = np.std(data1)
    if sigma < 1e-12:
        sigma = 1.0
    data1_z = (data1 - mu) / sigma
    data2_z = (data2 - mu) / sigma
    return data1_z, data2_z, mu, sigma

def remove_upper_outliers(data, threshold=2.5):
    z_scores = (data - np.mean(data)) / np.std(data)
    return data[z_scores <= threshold]

def get_kde_data(arr, gridsize=500):
    fig_kde, ax_kde = plt.subplots()
    sns.kdeplot(arr, gridsize=gridsize, bw_adjust=1, cut=0, ax=ax_kde, warn_singular=False)
    lines = ax_kde.get_lines()
    if len(lines) == 0:
        x, y = np.array([0]), np.array([0])
    else:
        x, y = lines[0].get_data()
    plt.close(fig_kde)
    return x, y

def plot_boxplots(ax_box, data, color='blue'):
    mean_val = np.mean(data)
    sns.boxplot(data, ax=ax_box, color=color, orient='h')
    ax_box.axvline(mean_val, color=color, linestyle='--')
    ax_box.set(xlabel='', ylabel='')
    return mean_val

def plot_rug(ax_rug, data, pos_data=None, color='blue'):
    sns.rugplot(data, color=color, ax=ax_rug, alpha=0.3, height=1)
    if pos_data is not None and len(pos_data) > 0:
        sns.rugplot(pos_data, color='red', ax=ax_rug, alpha=0.75, height=1)
    ax_rug.set(xticks=[], yticks=[])
    for spine in ax_rug.spines.values():
        spine.set_visible(False)

def violin_pairwise_plot(data1, data2, label1, label2, feature, out_file):
    combined = {feature: np.concatenate([data1, data2]),
                'Group': [label1]*len(data1) + [label2]*len(data2)}
    df_plot = pd.DataFrame(combined)
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    plt.figure(figsize=(6, 4))
    ax = sns.violinplot(x='Group', y=feature, data=df_plot, inner="quartile")
    sns.boxplot(x='Group', y=feature, data=df_plot, showcaps=True,
                boxprops={'facecolor':'None'}, showfliers=False,
                whiskerprops={'linewidth':2}, ax=ax)
    if len(data1) >= 2 and len(data2) >= 2:
        _, p_val = ttest_ind(data1, data2, equal_var=False)
    else:
        p_val = np.nan
    ax.text(0, mean1, f"mean={mean1:.3g}", color='black', ha='center', va='bottom', fontsize=8)
    ax.text(1, mean2, f"mean={mean2:.3g}", color='black', ha='center', va='bottom', fontsize=8)
    ax.set_title(f"{label1} (n={len(data1)}) vs {label2} (n={len(data2)})\nWelch t-test p={p_val:.3g}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"{c.OKGREEN}Saved violin plot: {out_file}{c.ENDC}")

def kde_with_resampling(data1_z, data2_z, label1, label2, feature, out_file, binnarize=False, num_points=500):
    # 2) Get KDE lines (exactly as provided)
    x1, y1 = get_kde_data(data1_z, gridsize=num_points)
    x2, y2 = get_kde_data(data2_z, gridsize=num_points)
    common_x = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), num_points)
    y1_mod = np.interp(common_x, x1, y1)
    y2_mod = np.interp(common_x, x2, y2)
    if binnarize:
        bins1 = np.histogram_bin_edges(y1_mod, bins=num_points)
        bins2 = np.histogram_bin_edges(y2_mod, bins=num_points)
        y1_mod = np.digitize(y1_mod, bins1)
        y2_mod = np.digitize(y2_mod, bins2)
    plt.figure(figsize=(6,4))
    plt.plot(common_x, y1_mod, color='blue', label=label1)
    plt.plot(common_x, y2_mod, color='green', label=label2)
    plt.fill_between(common_x, y1_mod, color='blue', alpha=0.2)
    plt.fill_between(common_x, y2_mod, color='green', alpha=0.2)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.xticks(np.linspace(common_x.min(), common_x.max(), 10))
    if binnarize:
        plt.title(f"{feature} - Zscore + Binned KDE\n{label1} vs {label2}")
    else:
        plt.title(f"{feature} - Zscore KDE\n{label1} vs {label2}")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"{c.OKGREEN}Saved KDE: {out_file}{c.ENDC}")

def kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature, out_file, binnarize=False, num_points=500):
    """
    Creates a multi-subplot KDE plot with:
      - Top two rows: boxplots for the z-scored NEG data (for each group)
      - Middle row: resampled KDE curves for the z-scored NEG data
      - Bottom two rows: rug plots for the NEG data (blue/green) with corresponding POS data (if available) in red.
    """
    # Merge POS data if available
    data1 = np.concatenate([data1, pos1]) if pos1 is not None else data1
    data2 = np.concatenate([data2, pos2]) if pos2 is not None else data2

    data1 = remove_upper_outliers(data1, threshold=2)
    data2 = remove_upper_outliers(data2, threshold=2)

    med1_raw = np.mean(data1)
    med2_raw = np.mean(data2)

    # 1) Z-score by data1
    def get_mu_sigma(data):
        return np.mean(data), np.std(data)

    def normalize(data, mu, sigma):
        return np.array((data - mu) / sigma)

    mu_1, sigma_1 = get_mu_sigma(data1)
    # data1_z = normalize(data1, mu_1, sigma_1)
    data1_z = data1
    mu_2, sigma_2 = get_mu_sigma(data2)
#     data2_z = normalize(data2, mu_2, sigma_2)
    data2_z = data2
    # Zscore to the pos
    # if pos1 is not None:
    #     pos1 = (pos1 - mu_1) / sigma_1
    # if pos2 is not None:
    #     pos2 = (pos2 - mu_2) / sigma_2

    # 2) Build the KDE lines
    x1, y1 = get_kde_data(data1_z, gridsize=num_points)
    x2, y2 = get_kde_data(data2_z, gridsize=num_points)

    # Common grid
    x_min, x_max = x1.min(), x1.max()
    y_min, y_max = x2.min(), x2.max()

    x_interp = np.linspace(x_min, x_max, num_points)
    y_interp = np.linspace(y_min, y_max, num_points)

    # Interpolate
    y1_mod = np.interp(x_interp, x1, y1)
    y2_mod = np.interp(y_interp, x2, y2)

    if binnarize:
        bins1 = np.histogram_bin_edges(y1_mod, bins=num_points)
        bins2 = np.histogram_bin_edges(y2_mod, bins=num_points)
        y1_mod = np.digitize(y1_mod, bins1)
        y2_mod = np.digitize(y2_mod, bins2)

    # 3) Subplots: 5 rows
    # top row: boxplot for data1
    # next row: boxplot for data2
    # middle row: the two lines
    # next row: rug for data1
    # last row: rug for data2
    fig, (ax_box1, ax_box2, ax_dist, ax_rug1, ax_rug2) = plt.subplots(
        5, 1, sharex=True,
        gridspec_kw={'height_ratios': (0.2, 0.2, 1, 0.05, 0.05)}
    )

    med1 = plot_boxplots(ax_box1, data1_z, color='blue')
    med2 = plot_boxplots(ax_box2, data2_z, color='green')

    plt.subplots_adjust(top=0.87)
    # Middle
    ax_dist.plot(x_interp, y1_mod, color='blue', label=label1)
    ax_dist.plot(y_interp, y2_mod, color='green', label=label2)
    ax_dist.fill_between(x_interp, y1_mod, color='blue', alpha=0.2)
    ax_dist.fill_between(y_interp, y2_mod, color='green', alpha=0.2)
    ax_dist.axvline(med1, color='blue', linestyle='--', alpha=0.5)
    ax_dist.axvline(med2, color='green', linestyle='--', alpha=0.5)

    ax_dist.set(
        xlabel='', ylabel='',
        xticklabels=[], yticklabels=[]
    )

    # Text with the numerical value of the raw means
    midpt_y = 0.5 * (y1.max() + y1.min())
    offset = 0.05 * (y1.max() - y1.min())
    ax_dist.text(med1, midpt_y, f"{med1_raw:.3g}", color='blue',
                    ha='center', va='bottom', fontsize=8)
    ax_dist.text(med2, midpt_y + offset, f"{med2_raw:.3g}", color='green',
                    ha='center', va='bottom', fontsize=8)

    ax_dist.legend()

    # Show number of instances for each group
    plt.suptitle(f"{label1}(n={len(data1)}) vs {label2}(n={len(data2)})")

    # 4) Rug plots for NEG data, with corresponding POS in red if available
    plot_rug(ax_rug1, data1_z, pos1, color='blue')
    plot_rug(ax_rug2, data2_z, pos2, color='green')

    plt.savefig(out_file, format='png')
    plt.close()
    print(f"{c.OKGREEN}Saved multi-subplot KDE: {out_file}{c.ENDC}")

def kde_combined_plot(data1, data2, label1, label2, feature, out_file, group_data):
    def get_pos_data(label, feature, group_data):
        pos_label = label.replace("NEG", "POS")
        if pos_label in group_data:
            return group_data[pos_label][feature].dropna().values
        else:
            return None
    pos1 = get_pos_data(label1, feature, group_data)
    pos2 = get_pos_data(label2, feature, group_data)
    normal_out = out_file.replace(".png", "_zscore.png")
    binned_out = out_file.replace(".png", "_zscore_binned.png")
    kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature,
                        normal_out, binnarize=False, num_points=500)
    kde_multi_subplots(data1, data2, pos1, pos2, label1, label2, feature,
                        binned_out, binnarize=True, num_points=500)

##############################################################################
# 3. Group Analysis Function
##############################################################################

def group_analysis(df, output_dir):
    """
    We compare only:
      - NEG vs NEG groups, where one group is from 'Adult WT' and the other from 'MYC'.
    For each numeric feature, produce violin and multi-subplot KDE plots in subfolders:
      Analysis/Energy, Analysis/IntegratedIntensity, Analysis/VolumeMu, Analysis/VolumeVx.
    """
    # For this analysis, we only have one numeric feature: "volume_microns"
    subfolders = {
        'volume_microns': os.path.join(output_dir, 'VolumeMu')
    }
    for sf in subfolders.values():
        os.makedirs(sf, exist_ok=True)
    numeric_features = ["volume_microns"]
    if 'group_label' not in df.columns:
        df['group_label'] = df['cohort'] + "_" + df['nuclei_type'] + "_" + df['brdu_status']
    summary = df.groupby(['group_label']).size().reset_index(name='count')
    summary.to_csv(os.path.join(output_dir, 'group_counts.csv'), index=False)
    group_data = {}
    for glab in sorted(df['group_label'].unique()):
        group_data[glab] = df[df['group_label'] == glab]
    # Valid pairs: Only compare matching nuclei types between WT and MYC.
    valid_pairs = []
    groups = sorted(df['group_label'].unique())
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            # Check different cohorts and same nuclei type, both NEG
            if (("Adult WT" in g1 and "MYC" in g2) or ("MYC" in g1 and "Adult WT" in g2)):
                if (("Mono" in g1 and "Mono" in g2) or ("Bi" in g1 and "Bi" in g2)):
                    if ("NEG" in g1 and "NEG" in g2):
                        valid_pairs.append((g1, g2))
    ttest_rows = []
    with Progress() as progress:
        total_tests = len(numeric_features) * len(valid_pairs)
        task = progress.add_task("[cyan]Performing pairwise t-tests...", total=total_tests)
        for feature in numeric_features:
            print(f"{c.OKBLUE}Analyzing feature:{c.ENDC} {feature}")
            for (g1, g2) in valid_pairs:
                data1 = group_data[g1][feature].dropna().values
                data2 = group_data[g2][feature].dropna().values
                if len(data1) < 2 or len(data2) < 2:
                    p_val = np.nan
                else:
                    _, p_val = ttest_ind(data1, data2, equal_var=False)
                ttest_rows.append({
                    'feature': feature,
                    'group1': g1,
                    'group2': g2,
                    'p_value': p_val
                })
                out_dir = subfolders[feature]
                base_file = f"{feature}_{g1}_vs_{g2}.png"
                violin_file = os.path.join(out_dir, "violin_" + base_file)
                kde_file = os.path.join(out_dir, "kde_" + base_file)
                violin_pairwise_plot(data1, data2, g1, g2, feature, violin_file)
                kde_combined_plot(data1, data2, g1, g2, feature, kde_file, group_data)
                progress.update(task, advance=1)
        ttest_df = pd.DataFrame(ttest_rows)
        ttest_df.to_csv(os.path.join(output_dir, 'pairwise_ttests.csv'), index=False)
        print(f"{c.OKGREEN}Group analysis complete. Results saved in {output_dir}{c.ENDC}")

##############################################################################
# 4. Main Orchestrator
##############################################################################

def main():
    """
    1) Set the path to the Excel file containing the data.
    2) Set the parent output folder.
    3) Read the Excel file (which has no header and 4 columns):
         Col1: Adult WT_Mono_NEG
         Col2: Adult WT_Bi_NEG
         Col3: MYC_Mono_NEG
         Col4: MYC_Bi_NEG
    4) Convert the data to long form and create columns for cohort, nuclei_type, and brdu_status.
    5) Perform group analysis comparing:
         Adult WT_Mono_NEG vs MYC_Mono_NEG and Adult WT_Bi_NEG vs MYC_Bi_NEG.
    6) Save the results in the appropriate subfolder under the parent output folder.
    """
    DATA_FILE = "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/sum.xlsx"   # <-- modify this path as needed
    OUTPUT_PARENT = "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/U_Bioinformatica/Morena/CrisReviewer/copy/Nacho/analysis_results/PastData"  # <-- modify this path as needed
    df_raw = pd.read_excel(DATA_FILE, header=None)
    df_raw.columns = ["Adult WT_Mono_NEG", "Adult WT_Bi_NEG", "MYC_Mono_NEG", "MYC_Bi_NEG"]
    df_long = df_raw.melt(var_name="group_label", value_name="volume_microns")
    mapping = {
        "Adult WT_Mono_NEG": ("Adult WT", "Mono", "NEG"),
        "Adult WT_Bi_NEG": ("Adult WT", "Bi", "NEG"),
        "MYC_Mono_NEG": ("MYC", "Mono", "NEG"),
        "MYC_Bi_NEG": ("MYC", "Bi", "NEG")
    }
    def assign_parts(label):
        return mapping.get(label, ("Unknown", "Unknown", "Unknown"))
    df_long[["cohort", "nuclei_type", "brdu_status"]] = df_long["group_label"].apply(lambda x: pd.Series(assign_parts(x)))
    # For this analysis, we only have NEG data (no POS), so no filtering needed.
    OUTPUT_DIR = os.path.join(OUTPUT_PARENT, "analysis_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    group_analysis(df_long, OUTPUT_DIR)

if __name__ == "__main__":
    main()