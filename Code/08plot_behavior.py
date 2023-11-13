# =============================================================================
# Behavior
# sensor: HMD & Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import patches
from scipy.stats import linregress
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
import pymer4

from Code.toolbox import utils


wave = 1
if wave == 1:
    problematic_subjects = [1, 3, 12, 15, 19, 20, 23, 24, 31, 33, 41, 45, 46, 47]
elif wave == 2:
    problematic_subjects = []

dir_path = os.getcwd()
save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Behavior')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

red = '#E2001A'
green = '#B1C800'
colors = [green, red]
SA_score = "SPAI"

df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';')

# Time spent in Rooms
df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test")]
df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
df_subset.loc[(df_subset['event'].str.contains("Office")) & ~(df_subset['event'].str.contains("With")), "room"] = "Office"
df_subset.loc[(df_subset['event'].str.contains("Living")) & ~(df_subset['event'].str.contains("With")), "room"] = "Living"
df_subset.loc[(df_subset['event'].str.contains("Dining")) & ~(df_subset['event'].str.contains("With")), "room"] = "Dining"
df_subset = df_subset.dropna(subset="duration")
df_subset = df_subset.groupby(["VP", "phase", "room"]).sum(numeric_only=True).reset_index()
df_subset = df_subset.drop(columns=SA_score)
df_subset = df_subset.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
rooms = ["Living", "Dining", "Office"]
phases = ['Habituation', 'Test']
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']

for idx_room, room in enumerate(rooms):
    # idx_room = 2
    # room = rooms[idx_room]
    df_room = df_subset.loc[df_subset['room'] == room].reset_index(drop=True)

    boxWidth = 1 / (len(phases) + 1)
    pos = [idx_room + x * boxWidth for x in np.arange(1, len(phases) + 1)]

    for idx_phase, phase in enumerate(phases):
        # idx_phase = 0
        # phase = phases[idx_phase]
        df_phase = df_room.loc[df_room['phase'] == phase].reset_index(drop=True)
        df_phase = df_phase.dropna(subset="duration")

        # Plot raw data points
        for i in range(len(df_phase)):
            # i = 0
            x = random.uniform(pos[idx_phase] - (0.2 * boxWidth), pos[idx_phase] + (0.2 * boxWidth))
            y = df_phase.reset_index().loc[i, "duration"].item()
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
        bootstrapping_dict = utils.bootstrapping(df_phase.loc[:, "duration"].values,
                                           numb_iterations=5000,
                                           alpha=alpha,
                                           as_dict=True,
                                           func='mean')

        ax.boxplot([df_phase.loc[:, "duration"].values],
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
        if (room == "Office") & (phase == "Test"):
            x = df_phase[SA_score].to_numpy()
            y = df_phase["duration"].to_numpy()
            linreg = linregress(x, y)
            print(f"r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

    formula = f"duration ~ phase + (1 | VP)"
    model = pymer4.models.Lmer(formula, data=df_room)
    model.fit(factors={"phase": ["Habituation", "Test"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="phase", p_adjust="holm")
    p = anova.loc["phase", "P-val"].item()

    max = df_subset["duration"].max()
    if p < 0.05:
        ax.hlines(y=max * 1.05, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
        ax.vlines(x=pos[0], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
        ax.vlines(x=pos[1], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
        ax.text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

df_crit = df_subset.copy()
df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

formula = f"duration ~ phase + room + {SA_score} + " \
          f"phase:room + phase:{SA_score} + room:{SA_score} +" \
          f"phase:room:{SA_score} + (1 | VP)"

max = df_subset["duration"].max()
model = pymer4.models.Lmer(formula, data=df_crit)
model.fit(factors={"phase": ["Habituation", "Test"], "room": ["Office", "Living", "Dining"]}, summarize=False)
anova = model.anova(force_orthogonal=True)
sum_sq_error = (sum(i * i for i in model.residuals))
anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
estimates, contrasts = model.post_hoc(marginal_vars="room", p_adjust="holm")
p_con = contrasts.loc[contrasts["Contrast"] == "Dining - Living", "P-val"].item()
if p_con < 0.05:
    ax.hlines(y=max*1.10, xmin=0.51, xmax=1.49, linewidth=0.7, color='k')
    ax.vlines(x=0.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=1.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else "." if p_con < 0.1 else ""
    ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')

p_con = contrasts.loc[contrasts["Contrast"] == "Dining - Office", "P-val"].item()
if p_con < 0.05:
    ax.hlines(y=max*1.10, xmin=1.51, xmax=2.49, linewidth=0.7, color='k')
    ax.vlines(x=1.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=2.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else "." if p_con < 0.1 else ""
    ax.text(2, max*1.105, p_sign, color='k', horizontalalignment='center')

p_con = contrasts.loc[contrasts["Contrast"] == "Living - Office", "P-val"].item()
if p_con < 0.05:
    ax.hlines(y=max*1.15, xmin=0.51, xmax=2.49, linewidth=0.7, color='k')
    ax.vlines(x=0.51, ymin=max*1.14, ymax=max*1.15, linewidth=0.7, color='k')
    ax.vlines(x=2.49, ymin=max*1.14, ymax=max*1.15, linewidth=0.7, color='k')
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else "." if p_con < 0.1 else ""
    ax.text(1.5, max*1.155, p_sign, color='k', horizontalalignment='center')

ax.set_xticks([x + 1 / 2 for x in range(len(rooms))])
ax.set_xticklabels(rooms)
ax.set_ylabel("Duration [s]")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
fig.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
    ["Habituation", "Test"], loc="center right")
fig.subplots_adjust(right=0.85)
plt.savefig(os.path.join(save_path, f"duration_rooms.png"), dpi=300, bbox_inches="tight")
plt.close()


# Time spent in the different rooms of the virtual agents
if wave == 1:
    df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test")]
    df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
    df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
    df_subset = df_subset.dropna(subset="duration")
    df_subset = df_subset.groupby(["VP", "phase", "Condition"]).sum(numeric_only=True).reset_index()
    df_subset = df_subset.drop(columns=SA_score)
    df_subset = df_subset.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

    conditions = ["friendly", "unfriendly"]
    df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]
    phases = ['Habituation', 'Test']
    titles = ["Room with Friendly Agent", "Room with Unfriendly Agent"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    boxWidth = 1 / (len(conditions) + 1)
    pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_subset.loc[df_subset['Condition'] == condition].reset_index(drop=True)

        boxWidth = 1 / (len(phases) + 1)
        pos = [idx_condition + x * boxWidth for x in np.arange(1, len(phases) + 1)]

        for idx_phase, phase in enumerate(phases):
            # idx_phase = 0
            # phase = phases[idx_phase]
            df_phase = df_cond.loc[df_cond['phase'] == phase].reset_index(drop=True)
            df_phase = df_phase.dropna(subset="duration")

            if phase == "Habituation":
                colors = ['#1F82C0', '#1F82C0']
            else:
                colors = [green, red]

            # Plot raw data points
            for i in range(len(df_phase)):
                # i = 0
                x = random.uniform(pos[idx_phase] - (0.25 * boxWidth), pos[idx_phase] + (0.25 * boxWidth))
                y = df_phase.reset_index().loc[i, "duration"].item()
                ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
            fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
            boxprops = dict(color=colors[idx_condition])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = utils.bootstrapping(df_phase.loc[:, "duration"].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            ax.boxplot([df_phase.loc[:, "duration"].values],
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

        formula = f"duration ~ phase + (1 | VP)"
        model = pymer4.models.Lmer(formula, data=df_cond)
        model.fit(factors={"phase": ["Habituation", "Test"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        sum_sq_error = (sum(i * i for i in model.residuals))
        anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
        estimates, contrasts = model.post_hoc(marginal_vars="phase", p_adjust="holm")
        p = anova.loc["phase", "P-val"].item()

        max = df_subset["duration"].max()
        if p < 0.05:
            ax.hlines(y=max * 1.05, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
            ax.vlines(x=pos[0], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
            ax.vlines(x=pos[1], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
            p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p_con < 0.05 else "." if p_con < 0.1 else ""
            ax.text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

    df_crit = df_subset.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"duration ~ phase + Condition + {SA_score} + " \
              f"phase:Condition + phase:{SA_score} + Condition:{SA_score} +" \
              f"phase:Condition:{SA_score} + (1 | VP)"

    max = df_subset["duration"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"phase": ["Habituation", "Test"], "Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="phase", p_adjust="holm")

    p = anova.loc["Condition", "P-val"].item()
    if p < 0.05:
        ax.hlines(y=max*1.10, xmin=0.51, xmax=1.49, linewidth=0.7, color='k')
        ax.vlines(x=0.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        ax.vlines(x=1.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
        ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')

    p_test = contrasts.loc[contrasts["phase"] == "Test", "P-val"].item()
    max = df_subset["duration"].max()
    if p_test < 0.05:
        ax.hlines(y=max*1.10, xmin=2*boxWidth, xmax=1+2*boxWidth, linewidth=0.7, color='k')
        ax.vlines(x=2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        ax.vlines(x=1+2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        p_sign = "***" if p_test < 0.001 else "**" if p_test < 0.01 else "*" if p_test < 0.05 else "." if p_test < 0.1 else ""
        ax.text(np.mean([2*boxWidth, 1+2*boxWidth]), max*1.105, p_sign, color='k', horizontalalignment='center')

    ax.set_xticks([x + 1 / 2 for x in range(len(conditions))])
    ax.set_xticklabels([title.replace("with", "with\n") for title in titles])
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"Total Duration in the Rooms [s]")
    # ax.set_title("Time Spent Close to Virtual Agents", fontweight='bold')
    # ax.legend(
    #     [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
    #      Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
    #      Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
    #     ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='upper left')

    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.7)
    plt.savefig(os.path.join(save_path, f"duration_hab-test.png"), dpi=300, bbox_inches="tight")
    plt.close()


    # Time spent in the different rooms: Correlation with SPAI
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1
    pos = [1]

    titles = ["Room with Friendly Agent", "Room with Unfriendly Agent"]
    df_test = df_subset.loc[df_subset['phase'] == "Test"]
    df_test = df_test.sort_values(by=SA_score)
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)

        x = df_cond[SA_score].to_numpy()
        y = df_cond["duration"].to_numpy()
        linreg = linregress(x, y)
        all_x = df_test[SA_score].to_numpy()
        all_y = df_cond["duration"].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.95 * df_test["duration"].max(),
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_condition])
        else:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.91 * df_test["duration"].max(),
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_condition])

        # Plot raw data points
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3, label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"Total Duration [s] in Test Phase")
    # ax.set_title(f"Time Spent Close to Virtual Agents", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"duration_test_{SA_score}.png"), dpi=300)
    plt.close()

    df_crit = df_test.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"duration ~ Condition + {SA_score} + " \
              f"Condition:{SA_score} + (1 | VP)"

    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)

    # Time spent in the different rooms: Correlation with SPAI (Test-Habituation)
    df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test")]
    df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
    df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
    df_subset = df_subset.dropna(subset="duration")
    df_subset = df_subset.groupby(["VP", "phase", "Condition"]).sum(numeric_only=True).reset_index()
    df_subset = df_subset.drop(columns=SA_score)
    df_subset = df_subset.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

    df_spai = df[["VP", SA_score]].drop_duplicates(subset="VP")
    df_diff = df_subset.pivot(index='VP', columns=['phase', "Condition"], values='duration').reset_index()
    df_diff = df_diff.fillna(0)
    df_diff["friendly"] = df_diff[("Test"), ("friendly")] - df_diff[("Habituation"), ("friendly")]
    df_diff["unfriendly"] = df_diff[("Test"), ("unfriendly")] - df_diff[("Habituation"), ("unfriendly")]
    df_diff = df_diff.iloc[:, [0, 7, 8]]
    df_diff.columns = df_diff.columns.droplevel(level=1)
    df_diff = df_diff.merge(df_spai, on="VP")
    df_diff = pd.melt(df_diff, id_vars=['VP', 'SPAI'], value_vars=['friendly', 'unfriendly'], var_name="Condition",
                      value_name="difference")
    df_diff = df_diff.sort_values(by=SA_score)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1
    pos = [1]
    conditions = ["friendly", "unfriendly"]
    titles = ["Friendly Agent", "Unfriendly Agent"]

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_diff.loc[df_diff['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="difference")
        df_cond = df_cond.sort_values(by=SA_score)

        x = df_cond[SA_score].to_numpy()
        y = df_cond["difference"].to_numpy()
        linreg = linregress(x, y)
        all_x = df_diff[SA_score].to_numpy()
        all_y = df_cond["difference"].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_diff[SA_score].min() + 0.01 * np.max(x),
                    0.95 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        else:
            ax.text(df_diff[SA_score].min() + 0.01 * np.max(x),
                    0.91 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])

        # Plot raw data points
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3,
                label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(
        f"Difference (Test - Habituation) Between\nTime Spent Close to the Position of the Virtual Agents [s]")
    # ax.set_title(f"Time Spent Close to the \nPosition of the Virtual Agents", fontweight='bold')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"duration_diff_{SA_score}.png"), dpi=300)
    plt.close()

if wave == 2:
    df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("With")]
    df_test = df_test.dropna(subset="duration")
    df_test = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
    df_test = df_test.drop(columns=SA_score)
    df_test = df_test.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_test = df_test.sort_values(by=SA_score)

    # Time spent in the different rooms
    conditions = ["friendly", "unfriendly"]
    titles = ["Room with Friendly Agent", "Room with Unfriendly Agent"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1 / (len(conditions) + 1)
    pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 1
        # condition = conditions[idx_condition]
        df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="duration")

        # Plot raw data points
        for i in range(len(df_cond)):
            # i = 0
            x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
            y = df_cond.reset_index().loc[i, "duration"].item()
            y_jittered = random.uniform(y - 0.1, y + 0.1)
            ax.plot(x, y_jittered, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

        # Plot boxplots
        meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
        medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
        fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

        whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        boxprops = dict(color=colors[idx_condition])

        fwr_correction = True
        alpha = (1 - (0.05))
        bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, "duration"].values,
                                           numb_iterations=5000,
                                           alpha=alpha,
                                           as_dict=True,
                                           func='mean')

        ax.boxplot([df_cond.loc[:, "duration"].values],
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

    df_crit = df_test.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"duration ~ Condition + {SA_score} + Condition:{SA_score} + (1 | VP)"

    max = df_subset["duration"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

    p = anova.loc["Condition", "P-val"].item()
    if p < 0.05:
        ax.hlines(y=max*1.10, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
        ax.vlines(x=pos[0], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        ax.vlines(x=pos[1], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"." if p < 0.1 else ""
        ax.text(np.mean([pos[0], pos[1]]), max*1.105, p_sign, color='k', horizontalalignment='center')

    ax.set_xticklabels([title.replace(" ", "\n") for title in titles])
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"Total Duration [s] in Test Phase")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"duration_test.png"), dpi=300)
    plt.close()

    # Time spent in the different rooms: Correlation with SPAI
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1
    pos = [1]
    conditions = ["friendly", "unfriendly"]
    titles = ["Room with Friendly Agent", "Room with Unfriendly Agent"]
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)

        x = df_cond[SA_score].to_numpy()
        y = df_cond["duration"].to_numpy()
        linreg = linregress(x, y)
        all_x = df_test[SA_score].to_numpy()
        all_y = df_cond["duration"].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.95 * df_test["duration"].max(),
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_condition])
        else:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.91 * df_test["duration"].max(),
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_condition])

        # Plot raw data points
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3, label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"Total Duration [s] in Test Phase")
    # ax.set_title(f"Time Spent Close to Virtual Agents", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"duration_test_{SA_score}.png"), dpi=300)
    plt.close()

    df_crit = df_test.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"duration ~ Condition + {SA_score} + " \
              f"Condition:{SA_score} + (1 | VP)"

    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)


# Difference
df_test = df.loc[df["event"].str.contains("Test")]
df_test = df_test.dropna(subset="duration")
df_spai = df_test[["VP", SA_score]].drop_duplicates(subset="VP")
df_test = df_test.loc[df_test["Condition"].isin(["friendly", "unfriendly"])]
df_diff = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
df_diff = df_diff.pivot(index='VP', columns='Condition', values='duration').reset_index()
df_diff = df_diff.fillna(0)
df_diff["difference"] = df_diff["unfriendly"] - df_diff["friendly"]

df_diff = df_diff[["VP", "difference"]].merge(df_spai, on="VP")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
df_diff = df_diff.sort_values(by=SA_score)
colors = ['teal']
x = df_diff[SA_score].to_numpy()
y = df_diff["difference"].to_numpy()
linreg = linregress(x, y)
y_est = linreg.slope * x + linreg.intercept
y_err = np.sqrt(np.sum((y - np.mean(y)) ** 2) / (len(y) - 2)) * np.sqrt(
    1 / len(x) + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

# Plot regression line
ax.plot(x, y_est, '-', color="lightgrey")
ax.fill_between(x, y_est + y_err, y_est - y_err, alpha=0.2, color="lightgrey")

# Plot raw data points
c = np.where(y < 0, 'teal', 'gold')
ax.scatter(x, y, s=30, c=c, alpha=0.6)

p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
ax.text(df_diff[SA_score].min() + 0.01 * np.max(x), 0.95 * (df_diff["difference"].max()-df_diff["difference"].min()) + df_diff["difference"].min(),
        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color="grey")

ax.set_title(f"Approach vs. Avoidance (N = {len(df_diff['VP'].unique())})", fontweight='bold')
if "SPAI" in SA_score:
    ax.set_xticks(range(0, 6))
elif "SIAS" in SA_score:
    ax.set_xticks(range(5, 65, 5))
ax.set_xlabel(SA_score)
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.axhline(0, linewidth=0.8, color="k", linestyle="dashed")
ax.set_ylabel("Difference Duration in Proximity: Unfriendly-Friendly")
ax.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7)],
    ["Approach", "Avoidance"], loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"duration_test-diff_{SA_score}.png"), dpi=300)
plt.close()

df_diff = df_diff[["VP", "difference"]]
df_diff = df_diff.rename(columns={"difference": "time_diff"})
df_diff = df_diff.sort_values(by="VP").reset_index(drop=True)
try:
    df_aa = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'aa_tendency.csv'), decimal='.', sep=';')
    if "time_diff" in df_aa.columns:
        df_aa.update(df_diff)
    else:
        df_aa = df_aa.merge(df_diff, on="VP")
    df_aa.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'aa_tendency.csv'), decimal='.', sep=';', index=False)
except:
    df_diff.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'aa_tendency.csv'), decimal='.', sep=';', index=False)


# Interpersonal Distance
df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance_vh.csv'), decimal='.', sep=';')
for dist, title in zip(["avg", "min"], ["Average", "Minimum"]):
    # dist, title = "avg", "Average"
    # dist, title = "min", "Minimum"
    conditions = ["friendly", "unfriendly"]
    df_test = df.loc[df["phase"].str.contains("Test")]
    df_test = df_test.loc[df_test["Condition"].isin(conditions)]
    titles = ["Friendly Agent", "Unfriendly Agent"]
    if dist == "avg":
        df_grouped = df_test.groupby(["VP", "Condition"]).mean(numeric_only=True).reset_index()
    elif dist == "min":
        df_grouped = df_test.groupby(["VP", "Condition"]).min(numeric_only=True).reset_index()
    df_grouped = df_grouped.drop(columns=SA_score)
    df_grouped = df_grouped.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_grouped = df_grouped.sort_values(by=SA_score)

    colors = [green, red]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1 / (len(conditions) + 1)
    pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 1
        # condition = conditions[idx_condition]
        df_cond = df_grouped.loc[df_grouped['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="distance")

        # Plot raw data points
        for i in range(len(df_cond)):
            # i = 0
            x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
            y = df_cond.reset_index().loc[i, "distance"].item()
            ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

        # Plot boxplots
        meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
        medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
        fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

        whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        boxprops = dict(color=colors[idx_condition])

        fwr_correction = True
        alpha = (1 - (0.05))
        bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, "distance"].values,
                                           numb_iterations=5000,
                                           alpha=alpha,
                                           as_dict=True,
                                           func='mean')

        ax.boxplot([df_cond.loc[:, "distance"].values],
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

    df_crit = df_grouped.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"distance ~ Condition + {SA_score} + Condition:{SA_score} + (1 | VP)"

    max = df_grouped["distance"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

    p = anova.loc["Condition", "P-val"].item()
    if p < 0.1:
        ax.hlines(y=max*1.10, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
        ax.vlines(x=pos[0], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        ax.vlines(x=pos[1], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"." if p < 0.1 else ""
        ax.text(np.mean([pos[0], pos[1]]), max*1.105, p_sign, color='k', horizontalalignment='center')

    ax.set_xticklabels([title.replace(" ", "\n") for title in titles])
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"{title} Distance to the Virtual Agents [m]")
    # ax.set_title(f"{title} Interpersonal Distance", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"distance_{dist}_test.png"), dpi=300)
    plt.close()

    # Interpersonal Distance: Correlation with SPAI
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1
    pos = [1]
    conditions = ["friendly", "unfriendly"]
    titles = ["Friendly Agent", "Unfriendly Agent"]

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_grouped.loc[df_grouped['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="distance")
        df_cond = df_cond.sort_values(by=SA_score)

        x = df_cond[SA_score].to_numpy()
        y = df_cond["distance"].to_numpy()
        linreg = linregress(x, y)
        all_x = df_grouped[SA_score].to_numpy()
        all_y = df_cond["distance"].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_grouped[SA_score].min() + 0.01 * np.max(x), 0.95 * (df_grouped["distance"].max() - df_grouped["distance"].min()) + df_grouped["distance"].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        else:
            ax.text(df_grouped[SA_score].min() + 0.01 * np.max(x), 0.91 * (df_grouped["distance"].max() - df_grouped["distance"].min()) + df_grouped["distance"].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])

        # Plot raw data points
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3,
                label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"{title} Distance to the Virtual Agents [m]")
    # ax.set_title(f"{title} Interpersonal Distance", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"distance_{dist}_test_{SA_score}.png"), dpi=300)
    plt.close()

    if wave == 1:
        # Distance to virtual agents (Comparison to Habituation)
        if dist == "avg":
            df_subset = df.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
        elif dist == "min":
            df_subset = df.groupby(["VP", "phase", "Condition"]).min(numeric_only=True).reset_index()
        df_subset = df_subset.drop(columns=SA_score)
        df_subset = df_subset.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

        conditions = ["friendly", "unfriendly"]
        phases = ['Habituation', 'Test']
        titles = ["Position of\nFriendly Agent", "Position of\nUnfriendly Agent"]
        df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        boxWidth = 1 / (len(conditions) + 1)
        pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

        for idx_condition, condition in enumerate(conditions):
            # idx_condition = 0
            # condition = conditions[idx_condition]
            df_cond = df_subset.loc[df_subset['Condition'] == condition].reset_index(drop=True)

            boxWidth = 1 / (len(phases) + 1)
            pos = [idx_condition + x * boxWidth for x in np.arange(1, len(phases) + 1)]

            for idx_phase, phase in enumerate(phases):
                # idx_phase = 0
                # phase = phases[idx_phase]
                df_phase = df_cond.loc[df_cond['phase'] == phase].reset_index(drop=True)
                df_phase = df_phase.dropna(subset="distance")

                if phase == "Habituation":
                    colors = ['#1F82C0', '#1F82C0']
                else:
                    colors = [green, red]

                # Plot raw data points
                for i in range(len(df_phase)):
                    # i = 0
                    x = random.uniform(pos[idx_phase] - (0.25 * boxWidth), pos[idx_phase] + (0.25 * boxWidth))
                    y = df_phase.reset_index().loc[i, "distance"].item()
                    ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

                # Plot boxplots
                meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
                medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
                fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

                whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
                capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
                boxprops = dict(color=colors[idx_condition])

                fwr_correction = True
                alpha = (1 - (0.05))
                bootstrapping_dict = utils.bootstrapping(df_phase.loc[:, "distance"].values,
                                                   numb_iterations=5000,
                                                   alpha=alpha,
                                                   as_dict=True,
                                                   func='mean')

                ax.boxplot([df_phase.loc[:, "distance"].values],
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

            formula = f"distance ~ phase + (1 | VP)"
            model = pymer4.models.Lmer(formula, data=df_cond)
            model.fit(factors={"phase": ["Habituation", "Test"]}, summarize=False)
            anova = model.anova(force_orthogonal=True)
            sum_sq_error = (sum(i * i for i in model.residuals))
            anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
            estimates, contrasts = model.post_hoc(marginal_vars="phase", p_adjust="holm")
            p = anova.loc["phase", "P-val"].item()

            max = df_subset["distance"].max()
            if p < 0.05:
                ax.hlines(y=max * 1.05, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
                ax.vlines(x=pos[0], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
                ax.vlines(x=pos[1], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
                p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"." if p < 0.1 else ""
                ax.text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

        df_crit = df_subset.copy()
        df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

        formula = f"distance ~ phase + Condition + {SA_score} + " \
                  f"phase:Condition + phase:{SA_score} + Condition:{SA_score} +" \
                  f"phase:Condition:{SA_score} + (1 | VP)"

        max = df_subset["distance"].max()
        model = pymer4.models.Lmer(formula, data=df_crit)
        model.fit(factors={"phase": ["Habituation", "Test"], "Condition": ["friendly", "unfriendly"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        sum_sq_error = (sum(i * i for i in model.residuals))
        anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
        estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="phase", p_adjust="holm")

        p = anova.loc["Condition", "P-val"].item()
        if p < 0.05:
            ax.hlines(y=max*1.10, xmin=0.51, xmax=1.49, linewidth=0.7, color='k')
            ax.vlines(x=0.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
            ax.vlines(x=1.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
            p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"." if p < 0.1 else ""
            ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')

        p_test = contrasts.loc[contrasts["phase"] == "Test", "P-val"].item()
        max = df_subset["distance"].max()
        if p_test < 0.05:
            ax.hlines(y=max*1.10, xmin=2*boxWidth, xmax=1+2*boxWidth, linewidth=0.7, color='k')
            ax.vlines(x=2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
            ax.vlines(x=1+2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
            p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"." if p < 0.1 else ""
            ax.text(np.mean([2*boxWidth, 1+2*boxWidth]), max*1.105, p_sign, color='k', horizontalalignment='center')

        ax.set_xticks([x + 1 / 2 for x in range(len(conditions))])
        ax.set_xticklabels([title.replace("with", "with\n") for title in titles])
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
        ax.set_ylabel(f"{title} Distance to Position of Virtual Agents [m]")
        # ax.set_title("Interpersonal Distance to Virtual Agents", fontweight='bold')
        # ax.legend(
        #     [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
        #      Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
        #      Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        #     ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='best')

        fig.legend(
            [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
             Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
             Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
            ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(right=0.7)
        plt.savefig(os.path.join(save_path, f"distance_{dist}_hab-test.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Interpersonal Distance: Correlation with SPAI
        df_spai = df_subset[["VP", SA_score]].drop_duplicates(subset="VP")
        df_diff = df_subset.pivot(index='VP', columns=['phase', "Condition"], values='distance').reset_index()
        df_diff = df_diff.fillna(0)
        df_diff["friendly"] = df_diff[("Test"), ("friendly")] - df_diff[("Habituation"), ("friendly")]
        df_diff["unfriendly"] = df_diff[("Test"), ("unfriendly")] - df_diff[("Habituation"), ("unfriendly")]
        df_diff = df_diff.iloc[:, [0, 5, 6]]
        df_diff.columns = df_diff.columns.droplevel(level=1)
        df_diff = df_diff.merge(df_spai, on="VP")
        df_diff = pd.melt(df_diff, id_vars=['VP', 'SPAI'], value_vars=['friendly', 'unfriendly'], var_name="Condition", value_name="difference")
        df_diff = df_diff.sort_values(by=SA_score)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
        boxWidth = 1
        pos = [1]
        conditions = ["friendly", "unfriendly"]
        titles = ["Friendly Agent", "Unfriendly Agent"]

        for idx_condition, condition in enumerate(conditions):
            # idx_condition = 0
            # condition = conditions[idx_condition]
            df_cond = df_diff.loc[df_diff['Condition'] == condition].reset_index(drop=True)
            df_cond = df_cond.dropna(subset="difference")
            df_cond = df_cond.sort_values(by=SA_score)

            x = df_cond[SA_score].to_numpy()
            y = df_cond["difference"].to_numpy()
            linreg = linregress(x, y)
            all_x = df_diff[SA_score].to_numpy()
            all_y = df_cond["difference"].to_numpy()
            all_y_est = linreg.slope * all_x + linreg.intercept
            all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
                1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

            # Plot regression line
            ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
            ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

            p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
            if idx_condition == 0:
                ax.text(df_diff[SA_score].min() + 0.01 * np.max(x),
                        0.95 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])
            else:
                ax.text(df_diff[SA_score].min() + 0.01 * np.max(x),
                        0.91 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])

            # Plot raw data points
            ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3,
                    label=titles[idx_condition])

        ax.set_xlabel(SA_score)
        if "SPAI" in SA_score:
            ax.set_xticks(range(0, 6))
        elif "SIAS" in SA_score:
            ax.set_xticks(range(5, 65, 5))
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
        ax.set_ylabel(f"Difference (Test - Habituation) Between\n{title} Distance to the Position of Virtual Agents [m]")
        # ax.set_title(f"Interpersonal Distance", fontweight='bold')
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"distance_{dist}_diff_{SA_score}.png"), dpi=300)
        plt.close()

        if dist != "avg":
            continue

        # Difference
        df_test = df_grouped.copy()
        df_spai = df_test[["VP", SA_score]].drop_duplicates(subset="VP")
        df_diff = df_test.groupby(["VP", "Condition"]).sum().reset_index()
        df_diff = df_diff.pivot(index='VP', columns='Condition', values='distance').reset_index()
        df_diff = df_diff.fillna(0)
        df_diff["difference"] = df_diff["unfriendly"] - df_diff["friendly"]
        df_diff = df_diff[["VP", "difference"]].merge(df_spai, on="VP")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
        df_diff = df_diff.sort_values(by=SA_score)
        colors = ['teal']
        x = df_diff[SA_score].to_numpy()
        y = df_diff["difference"].to_numpy()
        linreg = linregress(x, y)
        y_est = linreg.slope * x + linreg.intercept
        y_err = np.sqrt(np.sum((y - np.mean(y)) ** 2) / (len(y) - 2)) * np.sqrt(
            1 / len(x) + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

        # Plot regression line
        ax.plot(x, y_est, '-', color="lightgrey")
        ax.fill_between(x, y_est + y_err, y_est - y_err, alpha=0.2, color="lightgrey")

        # Plot raw data points
        c = np.where(y > 0, 'teal', 'gold')
        ax.scatter(x, y, s=30, c=c, alpha=0.6)

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        ax.text(df_diff[SA_score].min() + 0.01 * np.max(x), 0.95 * (df_diff["difference"].max()-df_diff["difference"].min()) + df_diff["difference"].min(),
                r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color="grey")

        ax.set_title(f"Approach vs. Avoidance (N = {len(df_diff['VP'].unique())})", fontweight='bold')
        # ax.set_ylim([0, max])
        ax.set_xlabel(SA_score)
        if "SPAI" in SA_score:
            ax.set_xticks(range(0, 6))
        elif "SIAS" in SA_score:
            ax.set_xticks(range(5, 65, 5))
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
        ax.axhline(0, linewidth=0.8, color="k", linestyle="dashed")
        ax.set_ylabel("Difference Average Interpersonal Distance: Unfriendly-Friendly")
        ax.legend(
            [Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7),
             Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7)],
            ["Approach", "Avoidance"], loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"distance_test-diff_{SA_score}.png"), dpi=300)
        plt.close()

        df_diff = df_diff[["VP", "difference"]]
        df_diff = df_diff.rename(columns={"difference": "distance_diff"})
        df_diff = df_diff.sort_values(by="VP").reset_index(drop=True)
        try:
            df_aa = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'aa_tendency.csv'), decimal='.', sep=';')
            if "distance_diff" in df_aa.columns:
                df_aa.update(df_diff)
            else:
                df_aa = df_aa.merge(df_diff, on="VP")
            df_aa.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'aa_tendency.csv'), decimal='.', sep=';', index=False)
        except:
            df_diff.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'aa_tendency.csv'), decimal='.', sep=';', index=False)

# Clicks
if wave == 1:
    df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';')
    df_subset = df.loc[df["event"].str.contains("Clicked")]
    df_subset = df_subset.groupby(["VP", "Condition"])["event"].count().reset_index()
    df_subset = df_subset.rename(columns={"event": "click_count"})

    df_vp1 = df[["VP", SA_score]].drop_duplicates(subset="VP")
    df_vp1["Condition"] = "friendly"
    df_vp2 = df_vp1.copy()
    df_vp2["Condition"] = "unfriendly"
    df_vp = pd.concat([df_vp1, df_vp2])

    df_subset = df_vp.merge(df_subset, on=["VP", "Condition"], how="outer")
    df_subset = df_subset.fillna(0)

    conditions = ["friendly", "unfriendly"]
    df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]
    titles = ["Friendly Agent", "Unfriendly Agent"]
    colors = [green, red]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1 / (len(conditions) + 1)
    pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 1
        # condition = conditions[idx_condition]
        df_cond = df_subset.loc[df_subset['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="click_count")

        # Plot raw data points
        for i in range(len(df_cond)):
            # i = 0
            x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
            y = df_cond.reset_index().loc[i, "click_count"].item()
            y_jittered = random.uniform(y - 0.1, y + 0.1)
            ax.plot(x, y_jittered, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

        # Plot boxplots
        meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
        medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
        fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

        whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
        boxprops = dict(color=colors[idx_condition])

        fwr_correction = True
        alpha = (1 - (0.05))
        bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, "click_count"].values,
                                           numb_iterations=5000,
                                           alpha=alpha,
                                           as_dict=True,
                                           func='mean')

        ax.boxplot([df_cond.loc[:, "click_count"].values],
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

    df_crit = df_subset.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"click_count ~ Condition + {SA_score} + Condition:{SA_score} + (1 | VP)"

    max = df_subset["click_count"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

    p = anova.loc["Condition", "P-val"].item()
    if p < 0.05:
        ax.hlines(y=max*1.10, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
        ax.vlines(x=pos[0], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        ax.vlines(x=pos[1], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"." if p < 0.1 else ""
        ax.text(np.mean([pos[0], pos[1]]), max*1.105, p_sign, color='k', horizontalalignment='center')

    ax.set_xticklabels([title.replace(" ", "\n") for title in titles])
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"Number of Clicks on the Virtual Agents")
    # ax.set_title("Additional Interaction Attempts", fontweight='bold')
    ax.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor='#B1C800', markeredgewidth=1, markerfacecolor='#B1C800', alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor='#E2001A', markeredgewidth=1, markerfacecolor='#E2001A', alpha=.7)],
        ["Friendly\nAgent", "Unfriendly\nAgent"], loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"clicks_test.png"), dpi=300)
    plt.close()

    # Clicks: Correlation with SPAI
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1
    pos = [1]
    conditions = ["friendly", "unfriendly"]
    titles = ["Friendly Agent", "Unfriendly Agent"]
    df_subset = df_subset.sort_values(by=SA_score)

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_subset.loc[df_subset['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="click_count")
        df_cond = df_cond.sort_values(by=SA_score)

        x = df_cond[SA_score].to_numpy()
        y = df_cond["click_count"].to_numpy()
        linreg = linregress(x, y)
        all_x = df_subset[SA_score].to_numpy()
        all_y = df_cond["click_count"].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_subset[SA_score].min() + 0.01 * np.max(x), 0.95 * df_subset["click_count"].max(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        else:
            ax.text(df_subset[SA_score].min() + 0.01 * np.max(x), 0.91 * df_subset["click_count"].max(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])

        # Plot raw data points
        y_jittered = [random.uniform(value - 0.1, value + 0.1) for value in y]
        ax.plot(x, y_jittered, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3, label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"Number of Clicks on the Virtual Agents (Test-Phase)")
    # ax.set_title(f"Additional Interaction Attempts", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"clicks_test_{SA_score}.png"), dpi=300)
    plt.close()


# Movement
df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';')
if wave == 2:
    df["x"] = df["x_player"]
    df["y"] = df["y_player"]
    df["distance_to_previous_scaled"] = df["distance_to_previous_player_scaled"]
# df_dist = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'walking_distance.csv'), decimal='.', sep=';')

df_spai = df[SA_score].unique()
df_spai.sort()
df_spai = df_spai[~np.isnan(df_spai)]
cNorm = matplotlib.colors.Normalize(vmin=np.min(df_spai) - 0.1 * np.max(df_spai), vmax=np.max(df_spai) + 0.1 * np.max(df_spai))
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('viridis_r'))
scalarMap.set_array([])

vps = df["VP"].unique()
vps.sort()
vps = np.reshape(vps, (-1, 8))

for vp_block in vps:
    # vp_block = vps[0]
    fig, axes = plt.subplots(nrows=len(vp_block), ncols=2, figsize=(9, 1.5 * len(vp_block)))

    for idx_vp, vp in enumerate(vp_block):
        # idx_vp, vp = 0, vp_block[0]
        df_vp = df.loc[df["VP"] == vp]
        # df_vp_dist = df_dist.loc[df_dist["VP"] == vp]
        df_vp = df_vp.dropna(subset="phase")
        index = df_vp.first_valid_index()
        spai = df_vp.loc[index, SA_score]

        if idx_vp == 0:
            spai = 2

        axes[idx_vp, 0].text(400, -870, f"VP {vp}", color="lightgrey", fontweight="bold", horizontalalignment='left')

        for idx_phase, phase in enumerate(["Habituation", "Test"]):
            # idx_phase, phase = 1, "Test"
            df_phase = df_vp.loc[df_vp["phase"].str.contains(phase)]
            df_phase = df_phase.sort_values(by="time")
            # df_phase_dist = df_vp_dist.loc[df_vp_dist["phase"].str.contains(phase)]
            # walking_distance = df_phase_dist["walking_distance"].item()

            axes[idx_vp, idx_phase].hlines(y=-954, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].hlines(y=-409, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=430, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-101, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-661, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-1280, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-661, ymin=-739, ymax=-614, linewidth=5, color='white')
            axes[idx_vp, idx_phase].vlines(x=-101, ymin=-554, ymax=-434, linewidth=5, color='white')

            axes[idx_vp, idx_phase].text(np.mean((-1291, 438)), -870, phase, color="k", horizontalalignment='center', fontsize="small")

            # axes[idx_vp, idx_phase].text(-1251, -870, f"{round(walking_distance, 2)} m", color="lightgrey", horizontalalignment='right', fontsize="small", fontstyle="italic")

            if phase == "Test":
                if wave == 1:
                    try:
                        df_cond = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Conditions3")
                        df_cond = df_cond[["VP", "Roles", "Rooms"]]
                        df_cond = df_cond.loc[df_cond["VP"] == int(vp)]
                        df_rooms = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Rooms3")
                        df_rooms = df_rooms[["Role", int(df_cond["Rooms"].item())]]
                        df_rooms = df_rooms.rename(columns={int(df_cond["Rooms"].item()): "Rooms"})
                    except:
                        print("no conditions file")

                    for room in ["Dining", "Living"]:
                        # room = "Dining"
                        role = df_rooms.loc[df_rooms["Rooms"] == room, "Role"].item()
                        color = green if role == "friendly" else red
                        if room == "Dining":
                            position_x = -490
                            position_y = -1034
                        else:
                            position_x = -870
                            position_y = 262
                        circle = patches.Circle((position_y, position_x), radius=30, color=color, alpha=0.5)
                        axes[idx_vp, idx_phase].add_patch(circle)
                elif wave == 2:
                    for condition in ["friendly", "unfriendly"]:
                        color = green if condition == "friendly" else red
                        x = df_phase[f"y_{condition}"].to_list()
                        y = df_phase[f"x_{condition}"].to_list()
                        lw = [item * 5 for item in df_phase[f"distance_to_previous_{condition}_scaled"].to_list()]
                        for i in np.arange(len(df_phase) - 1):
                            axes[idx_vp, idx_phase].plot([x[i], x[i + 1]], [y[i], y[i + 1]], lw=lw[i], label=phase, c=color, alpha=0.7)

            x = df_phase["y"].to_list()
            y = df_phase["x"].to_list()
            lw = [item * 5 for item in df_phase["distance_to_previous_scaled"].to_list()]
            for i in np.arange(len(df_phase) - 1):
                axes[idx_vp, idx_phase].plot([x[i], x[i + 1]], [y[i], y[i + 1]], lw=lw[i], label=phase, c=scalarMap.to_rgba(spai), alpha=0.8)

            axes[idx_vp, idx_phase].axis('scaled')
            axes[idx_vp, idx_phase].invert_xaxis()
            axes[idx_vp, idx_phase].invert_yaxis()
            axes[idx_vp, idx_phase].axis('off')

        axes[idx_vp, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"movement_{vp_block[0]}-{vp_block[-1]}_{SA_score}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Get gifs:
vps = df["VP"].unique()
vps.sort()
import matplotlib.animation as animation


def update(num, data_hab, hab, data_test, test, test_dot, data_friendly, friendly, friendly_dot, data_unfriendly, unfriendly, unfriendly_dot):
    hab.set_data(data_hab[..., :num])
    test.set_data(data_test[..., :num])
    friendly.set_data(data_friendly[..., :num])
    unfriendly.set_data(data_unfriendly[..., :num])
    test_dot.set_data(data_test[..., :num])
    friendly_dot.set_data(data_friendly[..., :num])
    unfriendly_dot.set_data(data_unfriendly[..., :num])
    return hab, test, friendly, unfriendly, test_dot, friendly_dot, unfriendly_dot


def update(num, data_hab, data_test, data_friendly, data_unfriendly, ax1, ax2):
    ax1.clear()
    ax1.text(400, -870, f"VP {vp}", color="lightgrey", fontweight="bold", horizontalalignment='left')
    ax1.hlines(y=-954, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
    ax1.hlines(y=-409, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
    ax1.vlines(x=430, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax1.vlines(x=-101, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax1.vlines(x=-661, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax1.vlines(x=-1280, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax1.vlines(x=-661, ymin=-739, ymax=-614, linewidth=5, color='white')
    ax1.vlines(x=-101, ymin=-554, ymax=-434, linewidth=5, color='white')
    ax1.text(np.mean((-1291, 438)), -870, "Habituation", color="k", horizontalalignment='center', fontsize="small")

    ax2.clear()
    ax2.hlines(y=-954, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
    ax2.hlines(y=-409, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
    ax2.vlines(x=430, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax2.vlines(x=-101, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax2.vlines(x=-661, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax2.vlines(x=-1280, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
    ax2.vlines(x=-661, ymin=-739, ymax=-614, linewidth=5, color='white')
    ax2.vlines(x=-101, ymin=-554, ymax=-434, linewidth=5, color='white')
    ax2.text(np.mean((-1291, 438)), -870, "Test", color="k", horizontalalignment='center', fontsize="small")

    ax1.scatter(data_hab[0][num], data_hab[1][num], label="Habituation", c=scalarMap.to_rgba(spai), alpha=0.8, marker="o", s=20, zorder=5)
    ax1.axis('scaled')
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.axis('off')

    ax2.scatter(data_test[0][num], data_test[1][num], label="Test", c=scalarMap.to_rgba(spai), alpha=0.8, marker="o", s=20, zorder=5)
    ax2.scatter(data_friendly[0][num], data_friendly[1][num], label="Friendly", c=green, alpha=0.8, marker="o", s=20, zorder=5)
    ax2.scatter(data_unfriendly[0][num], data_unfriendly[1][num], label="Unfriendly", c=red, alpha=0.8, marker="o", s=20, zorder=5)
    ax2.axis('scaled')
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.axis('off')

    ax2.axis('off')


if wave == 2:
    for idx_vp, vp in enumerate(vps):
        # idx_vp, vp = 0, vps[0]
        df_vp = df.loc[df["VP"] == vp]
        # df_vp_dist = df_dist.loc[df_dist["VP"] == vp]
        df_vp = df_vp.dropna(subset="phase")
        index = df_vp.first_valid_index()
        spai = df_vp.loc[index, SA_score]

        Writer = animation.writers['ffmpeg']
        Writer = Writer(fps=30, metadata=dict(title=f"Visualization of VP {vp}"), bitrate=-1)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 1.5))
        # hab, = ax1.plot([], [], label="Habituation", c=scalarMap.to_rgba(spai), alpha=0.5)
        # test, = ax2.plot([], [], label="Test", c=scalarMap.to_rgba(spai), alpha=0.5)
        # friendly, = ax2.plot([], [], label="Friendly", c=green, alpha=0.5)
        # unfriendly, = ax2.plot([], [], label="Unfriendly", c=red, alpha=0.5)

        # ax1.text(400, -870, f"VP {vp}", color="lightgrey", fontweight="bold", horizontalalignment='left')
        # ax1.hlines(y=-954, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
        # ax1.hlines(y=-409, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
        # ax1.vlines(x=430, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax1.vlines(x=-101, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax1.vlines(x=-661, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax1.vlines(x=-1280, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax1.vlines(x=-661, ymin=-739, ymax=-614, linewidth=5, color='white')
        # ax1.vlines(x=-101, ymin=-554, ymax=-434, linewidth=5, color='white')
        # ax1.text(np.mean((-1291, 438)), -870, "Habituation", color="k", horizontalalignment='center', fontsize="small")
        #
        # ax2.hlines(y=-954, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
        # ax2.hlines(y=-409, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
        # ax2.vlines(x=430, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax2.vlines(x=-101, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax2.vlines(x=-661, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax2.vlines(x=-1280, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
        # ax2.vlines(x=-661, ymin=-739, ymax=-614, linewidth=5, color='white')
        # ax2.vlines(x=-101, ymin=-554, ymax=-434, linewidth=5, color='white')
        # ax2.text(np.mean((-1291, 438)), -870, "Test", color="k", horizontalalignment='center', fontsize="small")
        
        # Habituation:
        df_hab = df_vp.loc[df_vp["phase"].str.contains("Habituation")]
        df_hab = df_hab.sort_values(by="time")

        x = df_hab["y"].to_list()
        y = df_hab["x"].to_list()
        data_hab = np.array([x, y])

        # ax1.axis('scaled')
        # ax1.invert_xaxis()
        # ax1.invert_yaxis()
        # ax1.axis('off')
        
        # Test
        df_test = df_vp.loc[df_vp["phase"].str.contains("Test")]
        df_test = df_test.sort_values(by="time")

        x = df_test["y"].to_list()
        y = df_test["x"].to_list()
        data_test = np.array([x, y])

        x = df_test["y_friendly"].to_list()
        y = df_test["x_friendly"].to_list()
        data_friendly = np.array([x, y])

        x = df_test["y_unfriendly"].to_list()
        y = df_test["x_unfriendly"].to_list()
        data_unfriendly = np.array([x, y])

        # ax2.axis('scaled')
        # ax2.invert_xaxis()
        # ax2.invert_yaxis()
        # ax2.axis('off')
        #
        # ax2.axis('off')

        plt.tight_layout()

        # line_animation = animation.FuncAnimation(fig, update, frames=900, fargs=(data_hab, hab, data_test, test, data_friendly, friendly, data_unfriendly, unfriendly))
        line_animation = animation.FuncAnimation(fig, update, frames=900, fargs=(data_hab, data_test, data_friendly, data_unfriendly, ax1, ax2))
        line_animation.save(os.path.join(save_path, f"movement_{vp}.mp4"), writer=Writer, dpi=300)

        plt.close()

# Movement per SA-Group
df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';')
df_spai = list(df.drop_duplicates(subset="VP")[SA_score])
df_spai.sort()
# cNorm = matplotlib.colors.Normalize(vmin=np.min(df_spai) - 0.4 * np.max(df_spai), vmax=np.max(df_spai) + 0.2 * np.max(df_spai))
cNorm = matplotlib.colors.Normalize(vmin=np.min(df_spai) - 0.1 * np.max(df_spai), vmax=np.max(df_spai) + 0.1 * np.max(df_spai))
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('viridis_r'))
vps = df["VP"].unique()
vps.sort()
cutoff_sa = 2.79 if SA_score == "SPAI" else 30

for cutoff, text, title in zip([cutoff_sa, np.median(df_spai)], [f"Cutoff ({round(cutoff_sa, 2)})", f"Median ({round(np.median(df_spai), 2)})"], ["cutoff", "median"]):
    # cutoff = np.median(df_spai)
    lsa, hsa = 0, 0
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
    for idx_row in [0, 1]:
        for idx_col in [0, 1, 2]:
            axes[idx_row, idx_col].hlines(y=-954, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
            axes[idx_row, idx_col].hlines(y=-409, xmin=-1285, xmax=435, linewidth=2, color='lightgrey')
            axes[idx_row, idx_col].vlines(x=430, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_row, idx_col].vlines(x=-101, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_row, idx_col].vlines(x=-661, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_row, idx_col].vlines(x=-1280, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_row, idx_col].vlines(x=-661, ymin=-739, ymax=-614, linewidth=5, color='white')
            axes[idx_row, idx_col].vlines(x=-101, ymin=-554, ymax=-434, linewidth=5, color='white')
            axes[idx_row, idx_col].axis('scaled')
            axes[idx_row, idx_col].axis('off')
            axes[idx_row, idx_col].invert_xaxis()
            axes[idx_row, idx_col].invert_yaxis()

    idx_row = 0
    idx_col = 0

    for idx_vp, vp in enumerate(vps):
        # idx_vp = 0
        # vp = vps[idx_vp]
        df_vp = df.loc[df["VP"] == vp]
        df_vp = df_vp.dropna(subset="phase")
        index = df_vp.first_valid_index()
        spai = df_vp.loc[index, SA_score]
        idx_row = 1 if spai < cutoff else 0
        if spai < cutoff:
            lsa += 1
        else:
            hsa += 1

        for idx_col, phase in enumerate(["Habituation", "Test"]):
            # idx_phase, phase = 0, "Habituation"
            df_phase = df_vp.loc[df_vp["phase"].str.contains(phase)]
            df_phase = df_phase.sort_values(by="time")

            if phase == "Test":
                try:
                    df_cond = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Conditions3")
                    df_cond = df_cond[["VP", "Roles", "Rooms"]]
                    df_cond = df_cond.loc[df_cond["VP"] == int(vp)]
                    df_rooms = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Rooms3")
                    df_rooms = df_rooms[["Role", int(df_cond["Rooms"].item())]]
                    df_rooms = df_rooms.rename(columns={int(df_cond["Rooms"].item()): "Rooms"})
                except:
                    print("no conditions file")

                if (df_rooms.loc[df_rooms["Role"] == "friendly", "Rooms"] == "Dining").item():
                    idx_col += 1

                for room in ["Dining", "Living"]:
                    # room = "Dining"
                    role = df_rooms.loc[df_rooms["Rooms"] == room, "Role"].item()
                    color = green if role == "friendly" else red
                    if room == "Dining":
                        position_x = -490
                        position_y = -1034
                    else:
                        position_x = -870
                        position_y = 262
                    circle = patches.Circle((position_y, position_x), radius=30, color=color, alpha=0.5)
                    axes[idx_row, idx_col].add_patch(circle)
            axes[idx_row, idx_col].plot(df_phase["y"], df_phase["x"], lw=0.8, label=phase, c=scalarMap.to_rgba(spai))

    axes[0, 0].text(510, np.mean([-954, -409]), f"≥ {SA_score}-{text}", color="k", fontstyle="italic", verticalalignment='center', rotation=90)
    axes[0, 0].text(580, np.mean([-954, -409]), f"HSA (N = {hsa})", color="k", verticalalignment='center', rotation=90)
    axes[1, 0].text(510, np.mean([-954, -409]), f"< {SA_score}-{text}", color="k", fontstyle="italic", verticalalignment='center', rotation=90)
    axes[1, 0].text(580, np.mean([-954, -409]), f"LSA (N = {lsa})", color="k", verticalalignment='center', rotation=90)

    axes[0, 0].set_title("Habituation", fontweight="bold")
    axes[0, 1].set_title("Test (Option 1)", fontweight="bold")
    axes[0, 2].set_title("Test (Option 2)", fontweight="bold")
    plt.subplots_adjust(right=0.98)
    cax = plt.axes([0.98, 0.25, 0.01, 0.5])

    if SA_score == "SPAI":
        cb = plt.colorbar(scalarMap, cax=cax, ticks=[0, 1, 2, 3, 4, 5], label="SPAI")
        cax.set_ylim([-0.2, 5.2])
    elif SA_score == "SIAS":
        cb = plt.colorbar(scalarMap, cax=cax, ticks=range(10, 65, 10), label="SIAS")
        cax.set_ylim([5, 65])
    cb.outline.set_visible(False)

    fig.subplots_adjust(wspace=-0.05, hspace=-0.4)
    plt.savefig(os.path.join(save_path, f"movement_{title}_{SA_score}.png"), dpi=300, bbox_inches='tight')
    plt.close()


df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';')
df_spai = list(df.drop_duplicates(subset="VP")[SA_score])
df_spai.sort()
cutoff_sa = 2.79 if SA_score == "SPAI" else 30

min_x = -1300
max_x = 450
step_x = 20
res_x = int(abs(min_x-max_x)/step_x)
X = np.arange(min_x, max_x, step_x)
min_y = -1000
max_y = -350
step_y = 20
res_y = int(abs(min_y-max_y)/step_y)
Y = np.arange(min_y, max_y, step_y)

for cutoff, text, title in zip([cutoff_sa, np.median(df_spai)], [f"Cutoff ({round(cutoff_sa, 2)})", f"Median ({round(np.median(df_spai), 2)})"], ["cutoff", "median"]):
    # cutoff, text, title = np.median(df_spai), f"Median ({round(np.median(df_spai), 2)})", "median"
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))

    idx_row = 0
    for idx_row in [0, 1]:
        if idx_row == 0:
            df_group = df.loc[df[SA_score] >= cutoff]
            hsa = len(df_group["VP"].unique())
        elif idx_row == 1:
            df_group = df.loc[df[SA_score] < cutoff]
            lsa = len(df_group["VP"].unique())

        for idx_phase, phase in enumerate(["Habituation", "Test"]):
            # idx_phase, phase = 0, "Habituation"
            # idx_phase, phase = 1, "Test"
            df_phase = df_group.loc[df_group["phase"].str.contains(phase)]

            if phase == "Habituation":
                idx_col = 0
                counts = np.zeros((res_y, res_x))

                for a in range(len(df_phase)):
                    # a = 0
                    for b1 in range(res_y):
                        # b1 = 0
                        if (Y[b1] - step_y / 2) <= df_phase["x"].values[a] < (Y[b1] + step_y / 2):
                            for b2 in range(res_x):
                                # b2 = 0
                                if (X[b2] - step_x / 2) <= df_phase["y"].values[a] < (X[b2] + step_x / 2):
                                    counts[b1, b2] += 1
                cols = (counts != 0).argmax(axis=0)
                first_non_zero_col = (cols != 0).argmax(axis=0)
                last_non_zero_col = (cols[first_non_zero_col:] != 0).argmin(axis=0) + first_non_zero_col - 1
                rows = (counts != 0).argmax(axis=1)
                first_non_zero_row = (rows != 0).argmax(axis=0)
                last_non_zero_row = (rows[first_non_zero_row:] != 0).argmin(axis=0) + first_non_zero_row - 1

                axes[idx_row, idx_col].hlines(y=first_non_zero_row - 2, xmin=first_non_zero_col - 1, xmax=last_non_zero_col + 1, linewidth=1, color='lightgrey')
                axes[idx_row, idx_col].hlines(y=last_non_zero_row + 3, xmin=first_non_zero_col - 1, xmax=last_non_zero_col + 1, linewidth=1, color='lightgrey')
                axes[idx_row, idx_col].vlines(x=first_non_zero_col - 1, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')
                axes[idx_row, idx_col].vlines(x=last_non_zero_col + 1, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')
                room_width = int((last_non_zero_col - first_non_zero_col) / 3)
                axes[idx_row, idx_col].vlines(x=first_non_zero_col + room_width + 2, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')
                axes[idx_row, idx_col].vlines(x=first_non_zero_col + 2 * room_width + 4, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')

                axes[idx_row, idx_col].imshow(counts, cmap="magma_r", norm="log")
                axes[idx_row, idx_col].axis('scaled')
                axes[idx_row, idx_col].axis('off')
                axes[idx_row, idx_col].invert_xaxis()
                # axes[idx_row, idx_col].invert_yaxis()

            elif phase == "Test":
                try:
                    df_cond = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Conditions3")
                    df_cond = df_cond[["VP", "Roles", "Rooms"]]
                    df_rooms = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Rooms3")
                    df_phase = df_phase.merge(df_cond, on="VP")
                    df_phase["Option"] = 1
                    df_phase.loc[df_phase["Rooms"] == 1, "Option"] = 2
                except:
                    print("no conditions file")

                for option in [1, 2]:
                    # option = 1
                    df_opt = df_phase.loc[df_phase["Option"] == option]
                    idx_col = 1 if option == 1 else 2

                    counts = np.zeros((res_y, res_x))
                    for a in range(len(df_opt)):
                        # a = 0
                        for b1 in range(res_y):
                            # b1 = 0
                            if (Y[b1] - step_y / 2) <= df_opt["x"].values[a] < (Y[b1] + step_y / 2):
                                for b2 in range(res_x):
                                    # b2 = 0
                                    if (X[b2] - step_x / 2) <= df_opt["y"].values[a] < (X[b2] + step_x / 2):
                                        counts[b1, b2] += 1

                    for room in ["Dining", "Living"]:
                        # room = "Dining"
                        if room == "Dining":
                            position_x = first_non_zero_col + room_width / 2 - 2
                            position_y = last_non_zero_row - 3
                            color = green if option == 2 else red
                        else:
                            position_x = last_non_zero_col - 4.5
                            position_y = first_non_zero_row + 1
                            color = green if option == 1 else red
                        circle = patches.Circle((position_x, position_y), radius=2, color=color, alpha=0.5)
                        axes[idx_row, idx_col].add_patch(circle)
                    axes[idx_row, idx_col].hlines(y=first_non_zero_row - 2, xmin=first_non_zero_col - 1, xmax=last_non_zero_col + 1, linewidth=1, color='lightgrey')
                    axes[idx_row, idx_col].hlines(y=last_non_zero_row + 3, xmin=first_non_zero_col - 1, xmax=last_non_zero_col + 1, linewidth=1, color='lightgrey')
                    axes[idx_row, idx_col].vlines(x=first_non_zero_col - 1, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')
                    axes[idx_row, idx_col].vlines(x=last_non_zero_col + 1, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')
                    room_width = int((last_non_zero_col - first_non_zero_col) / 3)
                    axes[idx_row, idx_col].vlines(x=first_non_zero_col + room_width + 2, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')
                    axes[idx_row, idx_col].vlines(x=first_non_zero_col + 2 * room_width + 4, ymin=first_non_zero_row - 2, ymax=last_non_zero_row + 3, linewidth=1, color='lightgrey')

                    axes[idx_row, idx_col].imshow(counts, cmap="magma_r", norm="log")
                    axes[idx_row, idx_col].axis('scaled')
                    axes[idx_row, idx_col].axis('off')
                    axes[idx_row, idx_col].invert_xaxis()
    axes[0, 0].text(last_non_zero_col + 5.5, np.mean([first_non_zero_row, last_non_zero_row]), f"≥ {SA_score}-{text}", color="k", fontstyle="italic", verticalalignment='center', rotation=90)
    axes[0, 0].text(last_non_zero_col + 9, np.mean([first_non_zero_row, last_non_zero_row]), f"HSA (N = {hsa})", color="k", verticalalignment='center', rotation=90)
    axes[1, 0].text(last_non_zero_col + 5.5, np.mean([first_non_zero_row, last_non_zero_row]), f"< {SA_score}-{text}", color="k", fontstyle="italic", verticalalignment='center', rotation=90)
    axes[1, 0].text(last_non_zero_col + 9, np.mean([first_non_zero_row, last_non_zero_row]), f"LSA (N = {lsa})", color="k", verticalalignment='center', rotation=90)

    axes[0, 0].set_title("Habituation", fontweight="bold")
    axes[0, 1].set_title("Test (Option 1)", fontweight="bold")
    axes[0, 2].set_title("Test (Option 2)", fontweight="bold")

    fig.subplots_adjust(wspace=0.02, hspace=-0.4)
    plt.savefig(os.path.join(save_path, f"movement_heatmap_{title}_{SA_score}.png"), dpi=300, bbox_inches='tight')
    plt.close()


# Heatmap per SA-Group
def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = matplotlib.image.imread(imagefile)

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img[:, :, 0:3]
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig, ax


def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(-1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M


def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y)

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = gwh / 2
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # i = 0
        # get x and y coordinates
        x = int(strt + gazepoints[i][0] - int(gwh / 2))
        y = int(strt + gazepoints[i][1] - int(gwh / 2))
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    #heatmap = np.flip(heatmap, axis=0)
    heatmap = np.flip(heatmap, axis=1)
    # resize heatmap
    strt = int(strt)
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = np.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    # if savefilename != None:
    #    fig.savefig(savefilename)

    return fig


df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';')
df["y_hm"] = df["x"] + 950  # abs(df["x"].min())
df["x_hm"] = df["y"] + 1300  # abs(df["y"].min())

df_spai = list(df.drop_duplicates(subset="VP")[SA_score])
df_spai.sort()
vps = df["VP"].unique()
vps.sort()
cutoff_sa = 2.79 if SA_score == "SPAI" else 30

for cutoff, text, title in zip([cutoff_sa, np.median(df_spai)], [f"Cutoff ({round(cutoff_sa, 2)})", f"Median ({round(np.median(df_spai), 2)})"], ["cutoff", "median"]):
    # cutoff = np.median(df_spai)
    for group in ["lsa", "hsa"]:
        # group = "hsa"
        if group == "hsa":
            df_group = df.loc[df[SA_score] >= cutoff]
        elif group == "lsa":
            df_group = df.loc[df[SA_score] < cutoff]
        for phase in ["Habituation", "Test"]:
            # phase = "Habituation"
            df_phase = df_group.loc[df_group["phase"] == phase]
            background_image = os.path.join(save_path, 'FloorPlan_Movement_sw.png')
            alpha = 0.7
            gaussianwh = 200
            gaussiansd = 30

            if phase == "Habituation":
                data = list(zip(df_phase["x_hm"].astype("int"), df_phase["y_hm"].astype("int"), len(df_phase) * [1]))
                fig = draw_heatmap(data, dispsize=(1732, 580), alpha=alpha,
                                   imagefile=background_image, gaussianwh=gaussianwh, gaussiansd=gaussiansd)

                plt.savefig(os.path.join(save_path, f"movement_heatmap_{title}_{SA_score}_{group}_habituation.png"), dpi=300, bbox_inches='tight')
                plt.close()
            elif phase == "Test":
                try:
                    df_cond = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Conditions3")
                    df_cond = df_cond[["VP", "Roles", "Rooms"]]
                    df_rooms = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Rooms3")
                    df_phase = df_phase.merge(df_cond, on="VP")
                    df_phase["Option"] = 1
                    df_phase.loc[df_phase["Rooms"] == 1, "Option"] = 2
                except:
                    print("no conditions file")

                for option in [1, 2]:
                    # option = 1
                    df_opt = df_phase.loc[df_phase["Option"] == option]

                    data = list(zip(df_opt["x_hm"].astype("int"), df_opt["y_hm"].astype("int"), len(df_opt) * [1]))
                    fig = draw_heatmap(data, dispsize=(1732, 580), alpha=alpha,
                                 imagefile=background_image, gaussianwh=gaussianwh, gaussiansd=gaussiansd)

                    plt.savefig(os.path.join(save_path, f"movement_heatmap_{title}_{SA_score}_{group}_test_opt{option}.png"), dpi=300, bbox_inches='tight')
                    plt.close()


# Walking Distance
df_dist = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'walking_distance.csv'), decimal='.', sep=';')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
boxWidth = 1
pos = [1]
phases = ["Habituation", "Test"]
titles = ["Habituation", "Test"]
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']

df_dist = df_dist.sort_values(by=SA_score)
for idx_dv, dv in enumerate(['walking_distance', 'average_distance_to_start', 'maximum_distance_to_start']):
    # dv = 'maximum_distance_to_start'
    formula = f"{dv} ~ phase + {SA_score} + phase:{SA_score} + (1 | VP)"

    model = pymer4.models.Lmer(formula, data=df_dist)
    model.fit(factors={"phase": ["Habituation", "Test"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="phase", p_adjust="holm")

    for idx_phase, phase in enumerate(phases):
        # idx_phase = 0
        # phase = phases[idx_phase]
        df_phase = df_dist.loc[df_dist['phase'] == phase].reset_index(drop=True)
        df_phase = df_phase.dropna(subset=dv)

        x = df_phase[SA_score].to_numpy()
        y = df_phase[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_dist[SA_score].to_numpy()
        all_y = df_phase[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        axes[idx_dv].plot(all_x, all_y_est, '-', color=colors[idx_phase])
        axes[idx_dv].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_phase])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_phase == 0:
            axes[idx_dv].text(df_dist[SA_score].min() + 0.01 * np.max(x), 0.95 * (df_dist[dv].max() - df_dist[dv].min()) + df_dist[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_phase])
        else:
            axes[idx_dv].text(df_dist[SA_score].min() + 0.01 * np.max(x), 0.91 * (df_dist[dv].max() - df_dist[dv].min()) + df_dist[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_phase])

        # Plot raw data points
        axes[idx_dv].plot(x, y, 'o', ms=5, mfc=colors[idx_phase], mec=colors[idx_phase], alpha=0.3, label=titles[idx_phase])

    axes[idx_dv].set_xlabel(SA_score)
    if "SPAI" in SA_score:
        axes[idx_dv].set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        axes[idx_dv].set_xticks(range(5, 65, 5))
    axes[idx_dv].grid(color='lightgrey', linestyle='-', linewidth=0.3)
    axes[idx_dv].set_ylabel(f"{dv.replace('_', ' ').title()} [m]")
    axes[idx_dv].set_title(f"{dv.replace('_', ' ').title()}", fontweight="bold")
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"walking_distance_grouped_{SA_score}.png"), dpi=300)
plt.close()

df_hr = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', f'hr.csv'), decimal='.', sep=';')
df_hr = df_hr.loc[(df_hr["Phase"].str.contains("Habituation")) | (df_hr["Phase"].str.contains("Test"))]
df_hr.loc[df_hr["Phase"].str.contains("Habituation"), "phase"] = "Habituation"
df_hr.loc[df_hr["Phase"].str.contains("Test"), "phase"] = "Test"
df_hr = df_hr.merge(df_dist[["VP", "phase", "walking_distance"]], on=["VP", "phase"])
df_hr = df_hr.groupby(["VP", "phase"]).mean().reset_index()
linreg = linregress(df_hr["HR (Mean)"], df_hr["walking_distance"])
print(f"r = {linreg.rvalue}, p = {round(linreg.pvalue, 3)}")
