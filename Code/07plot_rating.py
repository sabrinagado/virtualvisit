import os
import pandas as pd
import numpy as np
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
save_path = os.path.join(dir_path, 'Plots', 'Ratings')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

df_rating = pd.read_csv(os.path.join(dir_path, 'Data', 'ratings.csv'), decimal='.', sep=';')
colors = ['#B1C800', '#1F82C0', '#E2001A', '#179C7D', '#F29400']

# Ratings Virtual Humans
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
labels = ["Sympathy", "Fear", "Anger", "Attractiveness", "Behavior"]
conditions = ['Friendly', 'Neutral', 'Unfriendly', 'Unknown']

for idx_label, label in enumerate(labels):
    # idx_label = 2
    # label = labels[idx_label]

    boxWidth = 1 / (len(conditions) + 2)
    pos = [idx_label + x * boxWidth for x in np.arange(1, len(conditions) + 1)]
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 1
        # condition = conditions[idx_condition]
        df = df_rating.loc[(df_rating["Criterion"] == label) & (df_rating["Condition"] == condition.lower())]
        df = df.dropna(subset="Value")

        # Plot raw data points
        for i in range(len(df)):
            # i = 0
            x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
            y = df.reset_index().loc[i, "Value"].item()
            ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3, label=label)

        # Plot boxplots
        meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
        medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
        fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

        whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        boxprops = dict(color=colors[idx_condition])

        fwr_correction = True
        alpha = (1 - (0.05))
        bootstrapping_dict = bootstrapping(df.loc[:, "Value"].values,
                                           numb_iterations=5000,
                                           alpha=alpha,
                                           as_dict=True,
                                           func='mean')

        ax.boxplot([df.loc[:, "Value"].values],
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
                                         positions=[pos[idx_condition]],
                                         widths=0.8 * boxWidth)

ax.set_xticks([x + 1 / 2 for x in range(len(labels))])
ax.set_xticklabels(labels)
ax.set_ylim(-1, 101)
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.legend(
    [Line2D([0], [0], color=colors[0], marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7),
     Line2D([0], [0], color=colors[1], marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7),
     Line2D([0], [0], color=colors[2], marker='o', markeredgecolor=colors[2], markeredgewidth=1, markerfacecolor=colors[2], alpha=.7),
     Line2D([0], [0], color=colors[3], marker='o', markeredgecolor=colors[3], markeredgewidth=1, markerfacecolor=colors[3], alpha=.7)],
    conditions, loc="best")
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"ratings_humans{end}"), dpi=300)
plt.close()


# Relationship with Social Anxiety
df = df_rating.loc[df_rating["Criterion"].str.contains("Fear")]
x = df["SPAI"].to_numpy()
y = df["Value"].to_numpy()
linreg = linregress(x, y)
print(f"r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

# Ratings Rooms
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
rooms = ["Living", "Dining", "Office", "Terrace"]
phases = ['Habituation', 'Test']
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']
red = '#E2001A'
green = '#B1C800'

for idx_room, room in enumerate(rooms):
    # idx_room = 0
    # room = rooms[idx_room]

    boxWidth = 1 / (len(phases) + 1)
    pos = [idx_room + x * boxWidth for x in np.arange(1, len(phases) + 1)]

    for idx_phase, phase in enumerate(phases):
        # idx_phase = 1
        # phase = phases[idx_phase]
        df = df_rating.loc[(df_rating["Phase"] == phase) & (df_rating["Object"] == room)]
        df = df.dropna(subset="Value")

        if (phase == "Test") & ((room == "Living") | (room == "Dining")):
            conditions = ['friendly', 'unfriendly']
            boxWidth_room = boxWidth/2
            pos_room = [pos[idx_phase] - boxWidth_room*0.5, pos[idx_phase] + boxWidth_room]

            for idx_condition, condition in enumerate(conditions):
                # idx_condition = 0
                # condition = conditions[idx_condition]
                df_cond = df.loc[df["Condition"] == condition]
                if condition == "friendly":
                    color = green
                if condition == "unfriendly":
                    color = red

                # Plot raw data points
                for i in range(len(df_cond)):
                    # i = 0
                    x = random.uniform(pos_room[idx_condition] - (0.2 * boxWidth_room), pos_room[idx_condition] + (0.2 * boxWidth_room))
                    y = df_cond.reset_index().loc[i, "Value"].item()
                    ax.plot(x, y, marker='o', ms=5, mfc=color, mec=color, alpha=0.3)

                # Plot boxplots
                meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
                medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
                fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

                whiskerprops = dict(linestyle='solid', linewidth=1, color=color)
                capprops = dict(linestyle='solid', linewidth=1, color=color)
                boxprops = dict(color=color)

                fwr_correction = True
                alpha = (1 - (0.05))
                bootstrapping_dict = bootstrapping(df_cond.loc[:, "Value"].values,
                                                   numb_iterations=5000,
                                                   alpha=alpha,
                                                   as_dict=True,
                                                   func='mean')

                ax.boxplot([df_cond.loc[:, "Value"].values],
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
                           positions=[pos_room[idx_condition]],
                           widths=0.8 * boxWidth_room)
        else:
            # Plot raw data points
            for i in range(len(df)):
                # i = 0
                x = random.uniform(pos[idx_phase] - (0.2 * boxWidth), pos[idx_phase] + (0.2 * boxWidth))
                y = df.reset_index().loc[i, "Value"].item()
                ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_phase], mec=colors[idx_phase], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
            fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_phase])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_phase])
            boxprops = dict(color=colors[idx_phase])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = bootstrapping(df.loc[:, "Value"].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            ax.boxplot([df.loc[:, "Value"].values],
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
                       positions=[pos[idx_phase]],
                       widths=0.8 * boxWidth)

ax.set_xticks([x + 1 / 2 for x in range(len(rooms))])
ax.set_xticklabels(rooms)
ax.set_ylim(-1, 101)
ax.set_ylabel("Subjective Wellbeing")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
    ["Habituation", "Test", "Test (friendly)", "Test (unfriendly)"], loc="best")
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"ratings_rooms{end}"), dpi=300)
plt.close()


# Ratings VR
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))
phases = ['Orientation', 'Habituation', 'Test']
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']
boxWidth = 1
pos = [1]

for idx_phase, phase in enumerate(phases):
    # idx_phase = 0
    # phase = phases[idx_phase]
    df = df_rating.loc[(df_rating["Phase"] == phase) & (df_rating["Object"] == "VR")]
    df = df.dropna(subset="Value")

    # Plot raw data points
    for i in range(len(df)):
        # i = 0
        x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
        y = df.reset_index().loc[i, "Value"].item()
        axes[idx_phase].plot(x, y, marker='o', ms=5, mfc=colors[idx_phase], mec=colors[idx_phase], alpha=0.3)

    # Plot boxplots
    meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
    medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
    fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

    whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_phase])
    capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_phase])
    boxprops = dict(color=colors[idx_phase])

    fwr_correction = True
    alpha = (1 - (0.05))
    bootstrapping_dict = bootstrapping(df.loc[:, "Value"].values,
                                       numb_iterations=5000,
                                       alpha=alpha,
                                       as_dict=True,
                                       func='mean')

    axes[idx_phase].boxplot([df.loc[:, "Value"].values],
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

    axes[idx_phase].set_xticklabels([phase])
    axes[idx_phase].set_ylim(-1, 101)
    axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel("Subjective Wellbeing")
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"ratings_vr{end}"), dpi=300)
plt.close()

df = df_rating.groupby(["VP"]).mean().reset_index()
scales = ["SSQ", "IPQ", "MPS", "ASI", "SPAI", "SIAS", "AQ", "ISK"]
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']

for idx_scale, scale in enumerate(scales):
    # idx_scale = 0
    # scale = scales[idx_scale]
    df_scale = df.filter(like=scale)
    min = np.min(df_scale.min())
    max = np.max(df_scale.max())
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

            axes[idx_subscale].set_xticklabels([subscale])
            # axes[idx_subscale].set_ylim(min, max)
            axes[idx_subscale].grid(color='lightgrey', linestyle='-', linewidth=0.3)
        fig.suptitle(scale)
        plt.tight_layout()
        for end in (['.png']):  # '.pdf',
            plt.savefig(os.path.join(save_path, f"{scale}_vr{end}"), dpi=300)
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
                ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_subscale], mec=colors[idx_subscale],
                                        alpha=0.3)

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

            ax.set_xticklabels([subscale])
            # ax.set_ylim(min, max)
            ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
        fig.suptitle(scale)
        plt.tight_layout()
        for end in (['.png']):  # '.pdf',
            plt.savefig(os.path.join(save_path, f"{scale}_vr{end}"), dpi=300)
        plt.close()