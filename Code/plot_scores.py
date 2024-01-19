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
from Code.toolbox import utils

from Code import preproc_scores, preproc_ratings



def plot_scale(df, scale, colors, problematic_subjects):
    cutoff_ssq = df["SSQ-diff"].mean() + df["SSQ-diff"].std()  # .quantile(0.75)
    df_scale = df.filter(like=scale).dropna()
    cutoff = None
    titles_subscale = []
    if "SSQ" in scale:
        min = np.min(df_scale.min())
        max = np.max(df_scale.max())
        if "diff" in scale:
            print(f"SSQ threshold: {round(cutoff_ssq, 1)}")
        titles_subscale = ["SSQ", "Nausea", "Oculomotor\nDisturbance", "Disorientation"]
    elif scale == "IPQ":
        min = 0
        max = 6
        print(f"IPQ median: {round(df_scale['IPQ'].median(), 1)}")
        titles_subscale = ["IPQ", "Spatial\nPresence", "Experienced\nRealism", "Involvement"]
    elif scale == "MPS":
        min = 1
        max = 5
        titles_subscale = ["Physical\nPresence", "Social\nPresence", "Self-Presence"]
    elif scale == "ASI":
        min = 0
        max = 4 * 18
        titles_subscale = ["ASI3", "Physical\nConcerns", "Cognitive\nConcerns", "Social\nConcerns"]
    elif scale == "SPAI":
        min = 0
        max = 6
        cutoff = 2.79
    elif scale == "SIAS":
        cutoff = 30
        min = np.min(df_scale.min())
        max = np.max(df_scale.max())
    elif scale == "AQ":
        min = 0
        max = 33
        cutoff = 17
        titles_subscale = ["AQ-k", "Social\nInteraction", "Communication and\nReciprocity", "Imagination and\nCreativity"]
    elif scale == "ISK":
        min = np.min(df_scale.min())
        max = np.max(df_scale.max())
        titles_subscale = ["Social\nOrientation", "Offensiveness", "Self-Control", "Reflexibility"]
    else:
        min = np.min(df_scale.min())
        max = np.max(df_scale.max())
    min = min - 0.02 * max
    max = max + 0.02 * max
    n_subscales = len(df_scale.columns)
    if not titles_subscale:
        titles_subscale = df_scale.columns
    if n_subscales > 1:
        fig, axes = plt.subplots(nrows=1, ncols=n_subscales, figsize=(n_subscales * 2, 4))
        boxWidth = 1
        pos = [1]

        for idx_subscale, (subscale, title_subscale) in enumerate(zip(df_scale.columns, titles_subscale)):
            # idx_subscale = 0
            # subscale = df_scale.columns[idx_subscale]

            # Plot raw data points
            for i in range(len(df_scale)):
                # i = 0
                x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
                y = df_scale.reset_index().loc[i, subscale].item()
                if df.reset_index().loc[i, "ID"].item() in problematic_subjects:
                    axes[idx_subscale].plot(x, y, marker='o', ms=5, mfc="grey", mec="grey", alpha=0.3)
                else:
                    axes[idx_subscale].plot(x, y, marker='o', ms=5, mfc=colors[idx_subscale], mec=colors[idx_subscale], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_subscale])
            fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_subscale])
            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            boxprops = dict(color=colors[idx_subscale])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = utils.bootstrapping(df_scale.loc[:, subscale].values,
                                                     numb_iterations=5000,
                                                     alpha=alpha,
                                                     as_dict=True,
                                                     func='mean')

            axes[idx_subscale].boxplot([df_scale.loc[:, subscale].values],
                       whiskerprops=whiskerprops,
                       capprops=capprops,
                       boxprops=boxprops,
                       medianprops=medianlineprops,
                       showfliers=False, flierprops=fliermarkerprops,
                       # meanline=True,
                       # showmeans=True,
                       # meanprops=meanprops,
                       # notch=True,  # bootstrap=5000,
                       # conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
                       whis=[2.5, 97.5],
                       positions=[pos[0]],
                       widths=0.8 * boxWidth)

            axes[idx_subscale].errorbar(x=pos[0], y=bootstrapping_dict['mean'],
                        yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                        elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

            axes[idx_subscale].set_xticklabels([title_subscale])
            axes[idx_subscale].set_ylim(min, max)
            axes[idx_subscale].grid(color='lightgrey', linestyle='-', linewidth=0.3)
            if subscale == "SSQ-diff":
                axes[idx_subscale].axhline(cutoff_ssq, color="lightgrey", linewidth=0.8, linestyle="dashed")
            elif subscale == "AQ-K":
                axes[idx_subscale].axhline(cutoff, color="tomato", linewidth=0.8, linestyle="dashed")

    elif n_subscales == 1:
        fig, ax = plt.subplots(nrows=1, ncols=n_subscales, figsize=(n_subscales * 2, 4))
        boxWidth = 1
        pos = [1]
        for idx_subscale, subscale in enumerate(df_scale.columns):
            # idx_subscale = 0
            # subscale = df_scale.columns[idx_subscale]

            # Plot raw data points
            for i in range(len(df_scale)):
                # i = 0
                x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
                y = df_scale.reset_index().loc[i, subscale].item()
                if df.reset_index().loc[i, "ID"].item() in problematic_subjects:
                    ax.plot(x, y, marker='o', ms=5, mfc="grey", mec="grey", alpha=0.3)
                else:
                    ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_subscale], mec=colors[idx_subscale], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_subscale])
            fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_subscale])
            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            boxprops = dict(color=colors[idx_subscale])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = utils.bootstrapping(df_scale.loc[:, subscale].values,
                                                     numb_iterations=5000,
                                                     alpha=alpha,
                                                     as_dict=True,
                                                     func='mean')

            ax.boxplot([df_scale.loc[:, subscale].values],
                       whiskerprops=whiskerprops,
                       capprops=capprops,
                       boxprops=boxprops,
                       medianprops=medianlineprops,
                       showfliers=False, flierprops=fliermarkerprops,
                       # meanline=True,
                       # showmeans=True,
                       # meanprops=meanprops,
                       # notch=True,  # bootstrap=5000,
                       # conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
                       whis=[2.5, 97.5],
                       positions=[pos[0]],
                       widths=0.8 * boxWidth)

            ax.errorbar(x=pos[0], y=bootstrapping_dict['mean'],
                        yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                        elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

            ax.set_xticklabels([subscale])
            ax.set_ylim(min, max)
            ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
            ax.set_title(scale)
            if cutoff:
                ax.axhline(cutoff, color="tomato", linewidth=0.8, linestyle="dashed")
    # fig.suptitle(scale)
    plt.tight_layout()


def plot_sad(df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 1.6))
    sns.histplot(df["SPAI"], color="#1B4C87", ax=ax, binwidth=0.2, binrange=(0, 5),
                 kde=True, line_kws={"linewidth": 1, "color": "#173d6a"}, edgecolor='#f8f8f8',)
    ax.set_xlabel("SPAI (Social Anxiety)")
    ax.set_xlim([0, 6])
    # ax.set_ylim([0, 11])
    ax.set_yticks(range(0, 6, 2))
    ax.axvline(x=df["SPAI"].median(), color="#FFC300")
    ax.axvline(x=2.79, color="#FF5733")
    ax.legend(
            [Line2D([0], [0], color='#FFC300'), Line2D([0], [0], color='#FF5733')],
            ['Median', 'Remission Cut-Off'], fontsize='xx-small', loc="best", frameon=False)
    ax.set_facecolor('#f8f8f8')
    plt.tight_layout()


if __name__ == '__main__':
    wave = 1
    dir_path = os.getcwd()
    filepath = os.path.join(dir_path, f'Data-Wave{wave}')

    save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Scores')
    if not os.path.exists(save_path):
        print('creating path for saving')
        os.makedirs(save_path)

    if wave == 1:
        problematic_subjects = [1, 3, 12, 19, 33, 45, 46]
    elif wave == 2:
        problematic_subjects = [1, 2, 3, 4, 20, 29, 64]

    file_name = [item for item in os.listdir(filepath) if (item.endswith(".xlsx") and "raw" in item)][0]
    df_scores_raw = pd.read_excel(os.path.join(filepath, file_name))
    df_scores_raw = df_scores_raw.loc[df_scores_raw["FINISHED"] == 1]
    df_scores, problematic_subjects = preproc_scores.create_scores(df_scores_raw, problematic_subjects)

    start = 1
    vp_folder = [int(item.split("_")[1]) for item in os.listdir(filepath) if ("VP" in item)]
    end = np.max(vp_folder)
    vps = np.arange(start, end + 1)
    vps = [vp for vp in vps if not vp in problematic_subjects]

    df_ratings, problematic_subjects = preproc_ratings.create_ratings(vps, filepath, problematic_subjects, df_scores)
