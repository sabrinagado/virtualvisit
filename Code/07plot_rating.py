# =============================================================================
# Ratings
# sensor: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
import pymer4

from Code.toolbox import utils

dir_path = os.getcwd()
save_path = os.path.join(dir_path, 'Plots', 'Ratings')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

df_rating = pd.read_csv(os.path.join(dir_path, 'Data', 'ratings.csv'), decimal='.', sep=';')
colors = ['#B1C800', '#1F82C0', '#E2001A', '#179C7D', '#F29400']
SA_score = "SPAI"

# Ratings Virtual Humans
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
labels = ["Sympathy", "Fear", "Anger", "Attractiveness", "Behavior"]
conditions = ['Friendly', 'Neutral', 'Unfriendly', 'Unknown']

for idx_label, label in enumerate(labels):
    # idx_label = 1
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
        bootstrapping_dict = utils.bootstrapping(df.loc[:, "Value"].values,
                                           numb_iterations=5000,
                                           alpha=alpha,
                                           as_dict=True,
                                           func='mean')

        ax.boxplot([df.loc[:, "Value"].values],
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
                                         positions=[pos[idx_condition]],
                                         widths=0.8 * boxWidth)

    df_crit = df_rating.loc[df_rating["Criterion"] == label]
    df_crit = df_crit.loc[~(df_crit["Condition"] == "unknown")]
    formula = f"Value ~ Condition + (1 | VP)"
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "neutral", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i*i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")
    p = anova.loc["Condition", "P-val"].item()

    # ancova = pg.rm_anova(dv='Value', within='Condition', subject='VP', data=df_crit, effsize='np2')
    # p = ancova["p-unc"].item()
    # contrasts = pg.pairwise_tests(dv='Value', within='Condition', subject='VP', data=df_crit).round(3)

    max = df_crit["Value"].max()
    if p < 0.05:
        ax.hlines(y=max*1.10, xmin=pos[0] - boxWidth/2, xmax=pos[2] + boxWidth/2, linewidth=0.7, color='k')
        ax.vlines(x=pos[0] - boxWidth/2, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        ax.vlines(x=pos[2] + boxWidth/2, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(np.mean([pos[0], pos[2]]), max*1.105, p_sign, color='k', horizontalalignment='center',)

        p_con_f_n = contrasts.loc[contrasts["Contrast"] == "friendly - neutral", "P-val"].item()
        # p_con_f_n = contrasts.loc[(contrasts["A"] == "friendly") & (contrasts["B"] == "neutral"), "p-unc"].item()
        if p_con_f_n < 0.05:
            ax.hlines(y=max*1.05, xmin=pos[0] + boxWidth/80, xmax=pos[1] - boxWidth/80, linewidth=0.7, color='k')
            ax.vlines(x=pos[0] + boxWidth/80, ymin=max*1.04, ymax=max*1.05, linewidth=0.7, color='k')
            ax.vlines(x=pos[1] - boxWidth/80, ymin=max*1.04, ymax=max*1.05, linewidth=0.7, color='k')
            p_sign = "***" if p_con_f_n < 0.001 else "**" if p_con_f_n < 0.01 else "*" if p_con_f_n < 0.05 else ""
            ax.text(np.mean([pos[0], pos[1]]), max*1.055, p_sign, color='k', horizontalalignment='center',)
        p_con_n_u = contrasts.loc[contrasts["Contrast"] == "neutral - unfriendly", "P-val"].item()
        # p_con_n_u = contrasts.loc[(contrasts["A"] == "neutral") & (contrasts["B"] == "unfriendly"), "p-unc"].item()
        if p_con_n_u < 0.05:
            ax.hlines(y=max*1.05, xmin=pos[1] + boxWidth/80, xmax=pos[2] - boxWidth/80, linewidth=0.7, color='k')
            ax.vlines(x=pos[1] + boxWidth/80, ymin=max*1.04, ymax=max*1.05, linewidth=0.7, color='k')
            ax.vlines(x=pos[2] - boxWidth/80, ymin=max*1.04, ymax=max*1.05, linewidth=0.7, color='k')
            p_sign = "***" if p_con_n_u < 0.001 else "**" if p_con_n_u < 0.01 else "*" if p_con_n_u < 0.05 else ""
            ax.text(np.mean([pos[1], pos[2]]), max*1.055, p_sign, color='k', horizontalalignment='center',)

ax.set_xticks([x + 1 / 2 for x in range(len(labels))])
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(0, 101, 10))
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
fig.legend(
    [Line2D([0], [0], marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7, lw=0),
     Line2D([0], [0], marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7, lw=0),
     Line2D([0], [0], marker='o', markeredgecolor=colors[2], markeredgewidth=1, markerfacecolor=colors[2], alpha=.7, lw=0),
     Line2D([0], [0], marker='o', markeredgecolor=colors[3], markeredgewidth=1, markerfacecolor=colors[3], alpha=.7, lw=0)],
    conditions, loc="center right")
fig.subplots_adjust(right=0.89)
# plt.tight_layout()
plt.savefig(os.path.join(save_path, f"ratings_humans.png"), dpi=300, bbox_inches="tight")
plt.close()


# Ratings Virtual Humans, Relationship with Social Anxiety
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
labels = ["Sympathy", "Fear", "Anger"]
conditions = ['Unknown', 'Neutral', 'Friendly', 'Unfriendly']
colors = ['lightgrey', '#1F82C0', '#B1C800', '#E2001A']

for idx_label, label in enumerate(labels):
    # idx_label = 0
    # label = labels[idx_label]
    print(label)
    df_crit = df_rating.loc[df_rating["Criterion"] == label]
    df_crit = df_crit.sort_values(by=SA_score)
    # df_crit = df_crit.loc[~(df_crit["Condition"] == "unknown")]

    df_ancova = df_crit.copy()
    df_ancova = df_ancova.loc[df_ancova["Condition"].isin(["friendly", "unfriendly"])]
    df_ancova[SA_score] = (df_ancova[SA_score] - df_ancova[SA_score].mean()) / df_ancova[SA_score].std()
    formula = f"Value ~  {SA_score} + Condition + Condition:{SA_score} + (1 | VP)"
    model = pymer4.models.Lmer(formula, data=df_ancova)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i*i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

    if anova.loc[SA_score, "P-val"].item() < 0.05:
        print(f"{SA_score} Main Effect ({label}), p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={anova.loc[SA_score, 'p_eta_2'].item()}")
    if anova.loc["Condition", "P-val"].item() < 0.05:
        print(f"Condition Main Effect ({label}), p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={anova.loc['Condition', 'p_eta_2'].item()}")
    if anova.loc[f'{SA_score}:Condition', 'P-val'].item() < 0.05:
        print(f"Significant Interaction, p={round(anova.loc[f'{SA_score}:Condition', 'P-val'].item(), 3)}, p_eta_2={anova.loc[f'{SA_score}:Condition', 'p_eta_2'].item()}")

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_spai = df_crit.groupby(["VP"])[SA_score].mean(numeric_only=True).reset_index()
        df_spai = df_spai.sort_values(by=SA_score)

        df_cond = df_crit.loc[df_crit['Condition'] == condition.lower()].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="Value")
        df_cond = df_cond.groupby(["VP"]).mean(numeric_only=True).reset_index()
        df_cond = df_cond.sort_values(by=SA_score)

        x = df_cond[SA_score].to_numpy()
        y = df_cond["Value"].to_numpy()
        linreg = linregress(x, y)
        all_x = df_crit[SA_score].to_numpy()
        all_y = df_crit["Value"].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        axes[idx_label].plot(all_x, all_y_est, '-', color=colors[idx_condition])
        axes[idx_label].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 2:
            axes[idx_label].text(df_crit[SA_score].min() + 0.01 * np.max(x), 0.95 * df_crit["Value"].max(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        elif idx_condition == 3:
            axes[idx_label].text(df_crit[SA_score].min() + 0.01 * np.max(x), 0.91 * df_crit["Value"].max(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        elif idx_condition == 1:
            axes[idx_label].text(df_crit[SA_score].min() + 0.01 * np.max(x), 0.87 * df_crit["Value"].max(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        elif idx_condition == 0:
            axes[idx_label].text(df_crit[SA_score].min() + 0.01 * np.max(x), 0.83 * df_crit["Value"].max(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])

        # Plot raw data points
        axes[idx_label].plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6,
                label=conditions[idx_condition])

    # Style Plot
    axes[idx_label].set_ylim([-2, df_crit["Value"].max()+2])
    axes[idx_label].set_title(f"{label} (N = {len(df_cond['VP'].unique())})", fontweight='bold')
    axes[idx_label].set_ylabel(label)
    if "SPAI" in SA_score:
        axes[idx_label].set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        axes[idx_label].set_xticks(range(5, 65, 5))
    axes[idx_label].set_xlabel(SA_score)
    axes[idx_label].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[2].legend(loc="upper right")
axes[2].legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor='#B1C800', markeredgewidth=1, markerfacecolor='#B1C800', alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor='#E2001A', markeredgewidth=1, markerfacecolor='#E2001A', alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor='lightgrey', markeredgewidth=1, markerfacecolor='lightgrey', alpha=.7)],
    ["Friendly", "Unfriendly", "Neutral", "Unknown"], loc="upper right")

# fig.legend(
#     [Line2D([0], [0], marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7, lw=0),
#      Line2D([0], [0], marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7, lw=0),
#      Line2D([0], [0], marker='o', markeredgecolor=colors[2], markeredgewidth=1, markerfacecolor=colors[2], alpha=.7, lw=0),],
#     conditions, loc="center right")
# fig.subplots_adjust(right=0.89)
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"ratings_humans_{SA_score}.png"), dpi=300, bbox_inches="tight")
plt.close()


# Ratings Rooms
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
rooms = ["Living", "Dining", "Office", "Terrace"]
phases = ['Habituation', 'Test']
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']
red = '#E2001A'
green = '#B1C800'

for idx_room, room in enumerate(rooms):
    # idx_room = 1
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
                bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, "Value"].values,
                                                   numb_iterations=5000,
                                                   alpha=alpha,
                                                   as_dict=True,
                                                   func='mean')

                ax.boxplot([df_cond.loc[:, "Value"].values],
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
            bootstrapping_dict = utils.bootstrapping(df.loc[:, "Value"].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            ax.boxplot([df.loc[:, "Value"].values],
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
                       positions=[pos[idx_phase]],
                       widths=0.8 * boxWidth)
    if not room == "Terrace":
        df_crit = df_rating.loc[df_rating["Object"] == room]
        if room == "Office":
            formula = f"Value ~ Phase + (1 | VP)"
            model = pymer4.models.Lmer(formula, data=df_crit)
            model.fit(factors={"Phase": ["Habituation", "Test"]}, summarize=False)
            estimates, contrasts = model.post_hoc(marginal_vars="Phase", p_adjust="holm")
        else:
            formula = f"Value ~ Phase + Condition + Phase:Condition + (1 | VP)"
            model = pymer4.models.Lmer(formula, data=df_crit)
            model.fit(factors={"Phase": ["Habituation", "Test"], "Condition": ["friendly", "unfriendly"]}, summarize=False)
            estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="Phase", p_adjust="holm")

        anova = model.anova(force_orthogonal=True)
        sum_sq_error = (sum(i * i for i in model.residuals))
        anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)

        max = df_crit["Value"].max()
        if not room == "Office":
            p_cond = contrasts.loc[contrasts["Phase"] == "Test", "P-val"].item()
            if p_cond < 0.05:
                ax.hlines(y=max * 1.05, xmin=pos_room[0], xmax=pos_room[1], linewidth=0.7, color='k')
                ax.vlines(x=pos_room[0], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
                ax.vlines(x=pos_room[1], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
                p_sign = "***" if p_cond < 0.001 else "**" if p_cond < 0.01 else "*" if p_cond < 0.05 else ""
                ax.text(np.mean([pos_room[0], pos_room[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

        p_phase = anova.loc["Phase", "P-val"].item()
        if p_phase < 0.05:
            ax.hlines(y=max * 1.10, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
            ax.vlines(x=pos[0], ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
            ax.vlines(x=pos[1], ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
            p_sign = "***" if p_phase < 0.001 else "**" if p_phase < 0.01 else "*" if p_phase < 0.05 else ""
            ax.text(np.mean([pos[0], pos[1]]), max * 1.105, p_sign, color='k', horizontalalignment='center')

ax.set_xticks([x + 1 / 2 for x in range(len(rooms))])
ax.set_xticklabels(rooms)
ax.set_yticks(np.arange(0, 101, 10))
ax.set_ylabel("Subjective Wellbeing")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
fig.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
    ["Habituation", "Test", "Test (friendly)", "Test (unfriendly)"], loc="center right")
fig.subplots_adjust(right=0.82)
plt.savefig(os.path.join(save_path, f"ratings_rooms.png"), dpi=300, bbox_inches="tight")
plt.close()


# Ratings VR
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))
phases = ['Orientation', 'Habituation', 'Test']
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']
boxWidth = 1
pos = [1]

for idx_phase, phase in enumerate(phases):
    # idx_phase = 2
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
    bootstrapping_dict = utils.bootstrapping(df.loc[:, "Value"].values,
                                       numb_iterations=5000,
                                       alpha=alpha,
                                       as_dict=True,
                                       func='mean')

    axes[idx_phase].boxplot([df.loc[:, "Value"].values],
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

    axes[idx_phase].set_xticklabels([phase])
    axes[idx_phase].set_ylim(-1, 101)
    axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)

    if phase == "Test":
        linreg = linregress(df.loc[:, "Value"], df.loc[:, "SSQ-diff"])
        print(f"r = {linreg.rvalue}, p = {linreg.pvalue}")

axes[0].set_ylabel("Subjective Wellbeing")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"ratings_vr.png"), dpi=300)
plt.close()
