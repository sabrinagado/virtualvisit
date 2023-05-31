# =============================================================================
# Eye_tracking and Gaze: Proportion of Gaze on Social vs. Non-Social Stimuli
# sensor: HMD
# study: Virtual Visit
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
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


dir_path = os.getcwd()
save_path = os.path.join(dir_path, 'Plots', 'Gaze')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

dvs = ["Gaze Proportion", "Number"]
y_labels = ["% Fixations on Person", "Number of Fixations"]

for idx_dv, dv in enumerate(dvs):
    # idx_dv = 0
    # dv = dvs[idx_dv]
    df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';')

    max = round(df_gaze.loc[df_gaze["ROI"] != "other", dv].max(), 2) + 0.1

    meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
    medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
    fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

    phases = ["FriendlyInteraction", "UnfriendlyInteraction", "Test_FriendlyWasClicked", "Test_NeutralWasClicked", "Test_UnfriendlyWasClicked"]
    fig, axes = plt.subplots(nrows=1, ncols=len(phases), figsize=(3*len(phases), 6))
    titles = ["Friendly Interaction", "Unfriendly Interaction", "Clicked Friendly", "Clicked Neutral", "Clicked Unfriendly"]
    for idx_phase, phase in enumerate(phases):
        # idx_phase = 1
        # phase = "UnfriendlyInteraction"
        if "Friendly" in phase:
            condition = "friendly"
        elif "Unfriendly" in phase:
            condition = "unfriendly"
        elif "Neutral" in phase:
            condition = "neutral"
        rois = ["body", "head"]
        labels = ["Body", "Head"]
        y_label = y_labels[idx_dv]
        df_phase = df_gaze.loc[df_gaze['Phase'] == phase]
        df_phase = df_phase.loc[df_phase['Condition'] == condition]
        df_phase = df_phase.loc[df_phase['ROI'] != "other"].reset_index(drop=True)
        data_phase = df_phase[dv].to_list()

        boxWidth = 1 / (len(rois) + 1)  # mean_train + mean_test + train_f1 + f_test + 1
        pos = [0 + x * boxWidth for x in np.arange(1, len(rois) + 1)]

        colors = ['#183DB2', '#7FCEBC']

        for idx_roi, roi in enumerate(rois):
            # idx_roi = 0
            # roi = rois[idx_roi]

            # Plot raw data points
            df_roi = df_phase.loc[df_phase['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)
            for i in range(len(df_roi)):
                # i = 0
                x = random.uniform(pos[idx_roi] - (0.25 * boxWidth), pos[idx_roi] + (0.25 * boxWidth))
                y = df_roi.loc[i, dv].item()
                axes[idx_phase].plot(x, y, marker='o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.3)

            # Plot boxplots
            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
            boxprops = dict(color=colors[idx_roi])

            fwr_correction = False
            alpha = (1 - (0.05 / 2)) if fwr_correction else (1 - (0.05))
            bootstrapping_dict = bootstrapping(df_roi.loc[:, dv].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            axes[idx_phase].boxplot([df_roi.loc[:, dv].values],
                                    # notch=True,  # bootstrap=5000,
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
                                    positions=[pos[idx_roi]],
                                    widths=0.8 * boxWidth)

        axes[idx_phase].set_xticklabels(labels)
        axes[idx_phase].set_title(f"{titles[idx_phase]}", fontweight='bold')
        axes[idx_phase].set_ylim([0, max])
        axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
    axes[0].set_ylabel(y_label)

    plt.tight_layout()
    for end in (['.png']):  # '.pdf',
        plt.savefig(os.path.join(save_path, f"gaze_interactions_{dv}{end}"), dpi=300)
    plt.close()

    # Gaze Test-Phase Rooms
    df_test = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
    conditions = ["friendly", "neutral", "unfriendly"]
    fig, axes = plt.subplots(nrows=1, ncols=len(conditions), figsize=(3*len(conditions), 6))
    titles = ["Friendly Person", "Neutral Person", "Unfriendly Person"]
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        rois = ["body", "head"]
        labels = ["Body", "Head"]
        y_label = y_labels[idx_dv]
        df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.loc[df_cond['ROI'] != "other"].reset_index(drop=True)
        data_phase = df_cond[dv].to_list()

        boxWidth = 1 / (len(rois) + 1)  # mean_train + mean_test + train_f1 + f_test + 1
        pos = [0 + x * boxWidth for x in np.arange(1, len(rois) + 1)]

        colors = ['#183DB2', '#7FCEBC']

        for idx_roi, roi in enumerate(rois):
            # idx_roi = 0
            # roi = rois[idx_roi]

            # Plot raw data points
            df_roi = df_cond.loc[df_cond['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)
            for i in range(len(df_roi)):
                # i = 0
                x = random.uniform(pos[idx_roi] - (0.25 * boxWidth), pos[idx_roi] + (0.25 * boxWidth))
                y = df_roi.loc[i, dv].item()
                axes[idx_condition].plot(x, y, marker='o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.3)

            # Plot boxplots
            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
            boxprops = dict(color=colors[idx_roi])

            fwr_correction = False
            alpha = (1 - (0.05 / 2)) if fwr_correction else (1 - (0.05))
            bootstrapping_dict = bootstrapping(df_roi.loc[:, dv].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            axes[idx_condition].boxplot([df_roi.loc[:, dv].values],
                                    # notch=True,  # bootstrap=5000,
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
                                    positions=[pos[idx_roi]],
                                    widths=0.8 * boxWidth)

        axes[idx_condition].set_xticklabels(labels)
        axes[idx_condition].set_title(f"{titles[idx_condition]}", fontweight='bold')
        axes[idx_condition].set_ylim([0, max])
        axes[idx_condition].grid(color='lightgrey', linestyle='-', linewidth=0.3)
    axes[0].set_ylabel(y_label)

    plt.tight_layout()
    for end in (['.png']):  # '.pdf',
        plt.savefig(os.path.join(save_path, f"gaze_rooms_{dv}{end}"), dpi=300)
    plt.close()

# Relationship with Social Anxiety
df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';')
df = df_gaze.loc[df_gaze["Phase"].str.contains("Interaction")]
rois = ["body", "head"]
for idx_roi, roi in enumerate(rois):
    # idx_roi = 0
    # roi = rois[idx_roi]
    df_roi = df.loc[df['ROI'] == roi].dropna(subset="Gaze Proportion").reset_index(drop=True)
    x = df_roi["SPAI"].to_numpy()
    y = df_roi["Gaze Proportion"].to_numpy()
    linreg = linregress(x, y)
    print(f"{roi.capitalize()}: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")
