# =============================================================================
# Behavior
# sensor: HMD & Unreal Engine (Log Writer)
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
save_path = os.path.join(dir_path, 'Plots', 'Behavior')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

red = '#E2001A'
green = '#B1C800'
colors = [green, red]

df = pd.read_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';')

# Gaze Test-Phase Rooms
df_phase = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
conditions = ["friendly", "unfriendly"]

fig, axes = plt.subplots(nrows=1, ncols=len(conditions), figsize=(2.5 * len(conditions), 6))
boxWidth = 1
pos = [1]

titles = ["Friendly Person", "Unfriendly Person"]
for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = conditions[idx_condition]
    df_cond = df_phase.loc[df_phase['Condition'] == condition].reset_index(drop=True)
    data_phase = df_cond["duration"].to_list()
    df_cond = df_cond.dropna(subset="duration")
    df_cond = df_cond.groupby(["VP", "event"]).mean().reset_index()
    df_hab = df_cond.loc[df_cond['event'].str.contains("Habituation")]
    df_hab = df_hab[["VP", "duration"]]
    df_hab = df_hab.rename(columns={"duration": "Habituation"})
    df_test = df_cond.loc[df_cond['event'].str.contains("Test")]
    df_test = df_test.merge(df_hab, on="VP")
    df_test["duration"] = df_test["duration"] - df_test["Habituation"]

    # Plot raw data points
    for i in range(len(df_test)):
        # i = 0
        x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
        y = df_test.reset_index().loc[i, "duration"].item()
        axes[idx_condition].plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

    # Plot boxplots
    meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
    medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
    fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

    whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
    capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
    boxprops = dict(color=colors[idx_condition])

    fwr_correction = True
    alpha = (1 - (0.05))
    bootstrapping_dict = bootstrapping(df_test.loc[:, "duration"].values,
                                       numb_iterations=5000,
                                       alpha=alpha,
                                       as_dict=True,
                                       func='mean')

    axes[idx_condition].boxplot([df_test.loc[:, "duration"].values],
                            # notch=True,  # bootstrap=5000,
                            medianprops=medianlineprops,
                            meanline=True,
                            showmeans=True,
                            meanprops=meanlineprops,
                            showfliers=False, flierprops=fliermarkerprops,
                            whiskerprops=whiskerprops,
                            capprops=capprops,
                            boxprops=boxprops,
                            # conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
                            whis=[2.5, 97.5],
                            positions=[pos[0]],
                            widths=0.8 * boxWidth)

    axes[idx_condition].set_xticklabels([titles[idx_condition]])
    axes[idx_condition].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel(f"Duration in the room [s] in comparison to habituation phase")
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"timeInRoom_test{end}"), dpi=300)
plt.close()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
boxWidth = 1
pos = [1]

titles = ["Friendly Person", "Unfriendly Person"]
for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = conditions[idx_condition]
    df_cond = df_phase.loc[df_phase['Condition'] == condition].reset_index(drop=True)
    data_phase = df_cond["duration"].to_list()
    df_cond = df_cond.dropna(subset="duration")
    df_cond = df_cond.groupby(["VP", "event"]).mean().reset_index()
    df_hab = df_cond.loc[df_cond["event"].str.contains("Habituation")]
    df_hab = df_hab[["VP", "duration"]]
    df_hab = df_hab.rename(columns={"duration": "Habituation"})
    df_test = df_cond.loc[df_cond["event"].str.contains("Test")]
    df_test = df_test.merge(df_hab, on="VP")
    df_test["duration"] = df_test["duration"] - df_test["Habituation"]
    df_test = df_test.sort_values(by=["SPAI"])

    x = df_test["SPAI"].to_numpy()
    y = df_test["duration"].to_numpy()
    linreg = linregress(x, y)
    all_x = df_phase.sort_values(by=["SPAI"])["SPAI"].to_numpy()
    all_y = df_test["duration"].to_numpy()
    all_y_est = linreg.slope * all_x + linreg.intercept
    all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
        1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

    # Plot regression line
    ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
    ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

    # Plot raw data points
    ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

ax.set_xlabel("SPAI")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Duration [s] in comparison to habituation phase")
ax.legend()
# ax.set_xlim([0, 5])
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"timeInRoom_test_corr{end}"), dpi=300)
plt.close()
