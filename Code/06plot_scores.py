# =============================================================================
# Scores
# source: SosciSurvey
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import random


def percentiles(lst_vals, alpha, func='mean'):
    lower = np.percentile(np.array(lst_vals), ((1.0 - alpha) / 2.0) * 100, axis=0)
    upper = np.percentile(lst_vals, (alpha + ((1.0 - alpha) / 2.0)) * 100, axis=0)
    if func == 'mean':
        mean = np.mean(lst_vals, axis=0)
    elif func == 'median':
        mean = np.median(lst_vals, axis=0)
    return lower, mean, upper


def bootstrapping(input_sample,
                  sample_size=None,
                  numb_iterations=1000,
                  alpha=0.95,
                  plot_hist=False,
                  as_dict=True,
                  func='mean'):  # mean, median

    if sample_size == None:
        sample_size = len(input_sample)

    lst_means = []

    # ---------- Bootstrapping ------------------------------------------------

    print('\nBootstrapping with {} iterations and alpha: {}'.format(numb_iterations, alpha))
    for i in range(numb_iterations):
        try:
            re_sampled = random.choices(input_sample.values, k=sample_size)
        except:
            re_sampled = random.choices(input_sample, k=sample_size)

        if func == 'mean':
            lst_means.append(np.nanmean(np.array(re_sampled), axis=0))
        elif func == 'median':
            lst_means.append(np.median(np.array(re_sampled), axis=0))
        # lst_means.append(np.median(np.array(re_sampled), axis=0))

    # ---------- Konfidenzintervall -------------------------------------------

    lower, mean, upper = percentiles(lst_means, alpha)

    dict_return = {'lower': lower, 'mean': mean, 'upper': upper}

    # ---------- Visulisierung ------------------------------------------------

    if plot_hist:
        plt.hist(lst_means)

    # ---------- RETURN -------------------------------------------------------

    if as_dict:
        return dict_return
    else:
        return mean, np.array([np.abs(lower - mean), (upper - mean)])

wave = 2
dir_path = os.getcwd()
file_path = os.path.join(dir_path, f'Data-Wave{wave}')

save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Scores')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

# ToDo: Adapt problematic subject list
if wave == 1:
    problematic_subjects = [1, 3, 12, 19, 33, 45, 46]
elif wave == 2:
    problematic_subjects = [1, 2, 3, 4]


df = pd.read_csv(os.path.join(file_path, 'scores_summary.csv'), decimal=',', sep=';')
df = df.loc[~(df["ID"].isin(problematic_subjects))]
print(f"N = {len(df)}")
print(f"Mean Age = {df['age'].mean()}, SD = {df['age'].std()}, Range = {df['age'].min()}-{df['age'].max()}")
print(df['gender'].value_counts(normalize=True))

df = pd.read_csv(os.path.join(file_path, 'scores_summary.csv'), decimal=',', sep=';')
cutoff_ssq = df["SSQ-diff"].mean() + df["SSQ-diff"].std()  # .quantile(0.75)

scales = ["SSQ-post", "SSQ-diff", "IPQ", "MPS", "ASI", "SPAI", "SIAS", "AQ", "ISK"]
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']

for idx_scale, scale in enumerate(scales):
    # idx_scale = 5
    # scale = scales[idx_scale]
    df_scale = df.filter(like=scale)
    cutoff = None
    if scale == "SPAI":
        min = 0
        max = 6
        cutoff = 2.79
    elif scale == "IPQ":
        min = 0
        max = 6
    elif scale == "MPS":
        min = 1
        max = 5
    elif scale == "ASI":
        min = 0
        max = 4 * 18
    elif scale == "SIAS":
        cutoff = 30
        min = np.min(df_scale.min())
        max = np.max(df_scale.max())
    elif scale == "AQ":
        min = 0
        max = 33
        cutoff = 17
    else:
        min = np.min(df_scale.min())
        max = np.max(df_scale.max())
    min = min - 0.02 * max
    max = max + 0.02 * max
    n_subscales = len(df_scale.columns)
    if n_subscales > 1:
        fig, axes = plt.subplots(nrows=1, ncols=n_subscales, figsize=(n_subscales * 2, 6))
        boxWidth = 1
        pos = [1]

        for idx_subscale, subscale in enumerate(df_scale.columns):
            # idx_subscale = 0
            # subscale = df_scale.columns[idx_subscale]

            # Plot raw data points
            for i in range(len(df)):
                # i = 0
                x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
                y = df_scale.reset_index().loc[i, subscale].item()
                if df.reset_index().loc[i, "ID"].item() in problematic_subjects:
                    axes[idx_subscale].plot(x, y, marker='o', ms=5, mfc="grey", mec="grey", alpha=0.3)
                else:
                    axes[idx_subscale].plot(x, y, marker='o', ms=5, mfc=colors[idx_subscale], mec=colors[idx_subscale], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
            fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            boxprops = dict(color=colors[idx_subscale])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = bootstrapping(df_scale.loc[:, subscale].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            axes[idx_subscale].boxplot([df_scale.loc[:, subscale].values],
                       notch=True,  # bootstrap=5000,
                       medianprops=medianlineprops,
                       meanline=True,
                       showmeans=True,
                       meanprops=meanlineprops,
                       showfliers=False, flierprops=fliermarkerprops,
                       whiskerprops=whiskerprops,
                       capprops=capprops,
                       boxprops=boxprops,
                       conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
                       whis=[2.5, 97.5],
                       positions=[pos[0]],
                       widths=0.8 * boxWidth)

            axes[idx_subscale].set_xticklabels([subscale])
            axes[idx_subscale].set_ylim(min, max)
            axes[idx_subscale].grid(color='lightgrey', linestyle='-', linewidth=0.3)
            if subscale == "SSQ-diff":
                axes[idx_subscale].axhline(cutoff_ssq, color="lightgrey", linewidth=0.8, linestyle="dashed")
            elif subscale == "AQ-K":
                axes[idx_subscale].axhline(cutoff, color="tomato", linewidth=0.8, linestyle="dashed")

        fig.suptitle(scale)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{scale}_vr.png"), dpi=300)
        plt.close()
    elif n_subscales == 1:
        fig, ax = plt.subplots(nrows=1, ncols=n_subscales, figsize=(n_subscales * 2, 6))
        boxWidth = 1
        pos = [1]
        for idx_subscale, subscale in enumerate(df_scale.columns):
            # idx_subscale = 0
            # subscale = df_scale.columns[idx_subscale]

            # Plot raw data points
            for i in range(len(df)):
                # i = 0
                x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
                y = df_scale.reset_index().loc[i, subscale].item()
                if df.reset_index().loc[i, "ID"].item() in problematic_subjects:
                    ax.plot(x, y, marker='o', ms=5, mfc="grey", mec="grey", alpha=0.3)
                else:
                    ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_subscale], mec=colors[idx_subscale], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
            fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            boxprops = dict(color=colors[idx_subscale])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = bootstrapping(df_scale.loc[:, subscale].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            ax.boxplot([df_scale.loc[:, subscale].values],
                                       notch=True,  # bootstrap=5000,
                                       medianprops=medianlineprops,
                                       meanline=True,
                                       showmeans=True,
                                       meanprops=meanlineprops,
                                       showfliers=False, flierprops=fliermarkerprops,
                                       whiskerprops=whiskerprops,
                                       capprops=capprops,
                                       boxprops=boxprops,
                                       conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
                                       whis=[2.5, 97.5],
                                       positions=[pos[0]],
                                       widths=0.8 * boxWidth)

            ax.set_xticklabels([subscale])
            ax.set_ylim(min, max)
            ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
            if cutoff:
                ax.axhline(cutoff, color="tomato", linewidth=0.8, linestyle="dashed")

        fig.suptitle(scale)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{scale}_vr.png"), dpi=300)
        plt.close()

    df_plot = df.copy()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 1.6))
colors = ['#FF5733', '#FFC300', '#183DB2']
df_plot = df_plot.loc[~(df_plot["ID"].isin(problematic_subjects))]
sns.histplot(df_plot["SPAI"], color="#1B4C87", ax=ax, binwidth=0.2, binrange=(0, 5),
             kde=True, line_kws={"linewidth": 1, "color": "#173d6a"}, edgecolor='#f8f8f8',)
ax.set_xlabel("SPAI (Social Anxiety)")
ax.set_xlim([0, 5])
# ax.set_ylim([0, 11])
ax.set_yticks(range(0, 5, 2))
ax.axvline(x=df_plot["SPAI"].median(), color="#FFC300")
ax.axvline(x=2.79, color="#FF5733")
ax.legend(
        [Line2D([0], [0], color='#FFC300'), Line2D([0], [0], color='#FF5733')],
        ['Median', 'Remission Cut-Off'], fontsize='xx-small', loc="best", frameon=False)
ax.set_facecolor('#f8f8f8')
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"Distribution_SPAI.png"), dpi=300)
plt.close()

"""
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
# Execute one block after another
%load_ext rpy2.ipython

df_corr = df[['ASI3', 'SPAI', 'SIAS', 'AQ-K', 'VAS_start_anxiety', 'SSQ-diff', 'IPQ', 'MPS-SocP']]

%%R -i df_corr
library(apaTables)
library(tidyverse)
library(readr)
apa.cor.table(df_corr, filename = "test.doc", show.sig.stars=TRUE)
"""