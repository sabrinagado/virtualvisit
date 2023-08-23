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
from matplotlib import patches
from scipy.stats import linregress
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
import pymer4

from Code.toolbox import utils

dir_path = os.getcwd()
save_path = os.path.join(dir_path, 'Plots', 'Behavior')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

red = '#E2001A'
green = '#B1C800'
colors = [green, red]

df = pd.read_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';')

# Time spent in Rooms
df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
df_subset.loc[df_subset['event'].str.contains("Office"), "room"] = "Office"
df_subset.loc[df_subset['event'].str.contains("Living"), "room"] = "Living"
df_subset.loc[df_subset['event'].str.contains("Dining"), "room"] = "Dining"
df_subset = df_subset.dropna(subset="duration")
df_subset = df_subset.groupby(["VP", "phase", "room"]).sum(numeric_only=True).reset_index()
df_subset = df_subset.drop(columns="SPAI")
df_subset = df_subset.merge(df[["VP", "SPAI"]].drop_duplicates(subset="VP"), on="VP")

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
            x = df_phase["SPAI"].to_numpy()
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
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

df_crit = df_subset.copy()
df_crit["SPAI"] = (df_crit["SPAI"] - df_crit["SPAI"].mean()) / df_crit["SPAI"].std()

formula = f"duration ~ phase + room + SPAI + " \
          f"phase:room + phase:SPAI + room:SPAI +" \
          f"phase:room:SPAI + (1 | VP)"

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
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else ""
    ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')

p_con = contrasts.loc[contrasts["Contrast"] == "Dining - Office", "P-val"].item()
if p_con < 0.05:
    ax.hlines(y=max*1.10, xmin=1.51, xmax=2.49, linewidth=0.7, color='k')
    ax.vlines(x=1.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=2.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else ""
    ax.text(2, max*1.105, p_sign, color='k', horizontalalignment='center')

p_con = contrasts.loc[contrasts["Contrast"] == "Living - Office", "P-val"].item()
if p_con < 0.05:
    ax.hlines(y=max*1.15, xmin=0.51, xmax=2.49, linewidth=0.7, color='k')
    ax.vlines(x=0.51, ymin=max*1.14, ymax=max*1.15, linewidth=0.7, color='k')
    ax.vlines(x=2.49, ymin=max*1.14, ymax=max*1.15, linewidth=0.7, color='k')
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else ""
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


# Time spent in the different rooms of the virtual humans
df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
df_subset.loc[df_subset['event'].str.contains("Office"), "room"] = "Office"
df_subset.loc[df_subset['event'].str.contains("Living"), "room"] = "Living"
df_subset.loc[df_subset['event'].str.contains("Dining"), "room"] = "Dining"
df_subset = df_subset.dropna(subset="duration")
df_subset = df_subset.groupby(["VP", "phase", "room", "Condition"]).sum(numeric_only=True).reset_index()
df_subset = df_subset.drop(columns="SPAI")
df_subset = df_subset.merge(df[["VP", "SPAI"]].drop_duplicates(subset="VP"), on="VP")

conditions = ["friendly", "unfriendly"]
phases = ['Habituation', 'Test']
titles = ["Room with Friendly Person", "Room with Unfriendly Person"]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
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
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

df_crit = df_subset.copy()
df_crit["SPAI"] = (df_crit["SPAI"] - df_crit["SPAI"].mean()) / df_crit["SPAI"].std()

formula = f"duration ~ phase + Condition + SPAI + " \
          f"phase:Condition + phase:SPAI + Condition:SPAI +" \
          f"phase:Condition:SPAI + (1 | VP)"

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
    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')

p_test = contrasts.loc[contrasts["phase"] == "Test", "P-val"].item()
max = df_subset["duration"].max()
if p_test < 0.05:
    ax.hlines(y=max*1.10, xmin=2*boxWidth, xmax=1+2*boxWidth, linewidth=0.7, color='k')
    ax.vlines(x=2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=1+2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p_test < 0.001 else "**" if p_test < 0.01 else "*" if p_test < 0.05 else ""
    ax.text(np.mean([2*boxWidth, 1+2*boxWidth]), max*1.105, p_sign, color='k', horizontalalignment='center')

ax.set_xticks([x + 1 / 2 for x in range(len(conditions))])
ax.set_xticklabels([title.replace("with", "with\n") for title in titles])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Total Duration in the Rooms [s]")
ax.set_title("Time Spent Close to Virtual Humans", fontweight='bold')
ax.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
    ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='upper left')

# fig.legend(
#     [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
#      Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
#      Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
#     ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
# fig.subplots_adjust(right=0.76)
plt.savefig(os.path.join(save_path, f"duration_test.png"), dpi=300, bbox_inches="tight")
plt.close()


# Time spent in the different rooms: Correlation with SPAI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
boxWidth = 1
pos = [1]

titles = ["Room with Friendly Person", "Room with Unfriendly Person"]
df_test = df_subset.loc[df_subset['phase'] == "Test"]
df_test = df_test.sort_values(by="SPAI")
for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = conditions[idx_condition]
    df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)

    x = df_cond["SPAI"].to_numpy()
    y = df_cond["duration"].to_numpy()
    linreg = linregress(x, y)
    all_x = df_test["SPAI"].to_numpy()
    all_y = df_cond["duration"].to_numpy()
    all_y_est = linreg.slope * all_x + linreg.intercept
    all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
        1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

    # Plot regression line
    ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
    ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

    p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
    if idx_condition == 0:
        ax.text(df_test["SPAI"].min() + 0.01 * np.max(x), 0.95 * df_test["duration"].max(),
                             r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                             color=colors[idx_condition])
    else:
        ax.text(df_test["SPAI"].min() + 0.01 * np.max(x), 0.91 * df_test["duration"].max(),
                             r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                             color=colors[idx_condition])

    # Plot raw data points
    ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

ax.set_xlabel("SPAI")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Total Duration [s] in Test Phase")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"duration_test_SA.png"), dpi=300)
plt.close()


# Difference
df_test = df_subset.loc[df_subset['phase'] == "Test"]
df_spai = df_test[["VP", "SPAI"]].drop_duplicates(subset="VP")
df_diff = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
df_diff = df_diff.pivot(index='VP', columns='Condition', values='duration').reset_index()
df_diff = df_diff.fillna(0)
df_diff["difference"] = df_diff["unfriendly"] - df_diff["friendly"]

df_diff = df_diff[["VP", "difference"]].merge(df_spai, on="VP")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
df_diff = df_diff.sort_values(by="SPAI")
colors = ['teal']
x = df_diff["SPAI"].to_numpy()
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
ax.text(df_diff["SPAI"].min() + 0.01 * np.max(x), 0.95 * (df_diff["difference"].max()-df_diff["difference"].min()) + df_diff["difference"].min(),
        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color="grey")

ax.set_title(f"Avoidance vs. Hypervigilance (N = {len(df_diff['VP'].unique())})", fontweight='bold')
# ax.set_ylim([0, max])
ax.set_xlabel("SPAI")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.axhline(0, linewidth=0.8, color="k", linestyle="dashed")
ax.set_ylabel("Difference Duration in Proximity: Unfriendly-Friendly")
ax.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7)],
    ["Hypervigilance", "Avoidance"], loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"duration_test-diff_SPAI.png"), dpi=300)
plt.close()


# Interpersonal Distance
df = pd.read_csv(os.path.join(dir_path, 'Data', 'distance.csv'), decimal='.', sep=';')
df = df.loc[df["distance"] <= 500]
df_spai = df.groupby(["VP"])["SPAI"].mean().reset_index()
df_spai = df_spai.sort_values(by="SPAI")
df_phase = df.loc[df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
df_grouped = df_phase.groupby(["VP", "Condition"]).mean().reset_index()
df_grouped = df_grouped.loc[~(df_grouped["Condition"].str.contains("unknown"))]
df_grouped = df_grouped.drop(columns="SPAI")
df_grouped = df_grouped.merge(df_spai, on="VP")
conditions = ["friendly", "unfriendly"]
df_grouped = df_grouped.loc[df_grouped["Condition"].isin(conditions)].reset_index(drop=True)
titles = ["Friendly Person", "Unfriendly Person"]
colors = [green, red]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
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
df_crit["SPAI"] = (df_crit["SPAI"] - df_crit["SPAI"].mean()) / df_crit["SPAI"].std()

formula = f"distance ~ Condition + SPAI + Condition:SPAI + (1 | VP)"

max = df_grouped["distance"].max()
model = pymer4.models.Lmer(formula, data=df_crit)
model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
anova = model.anova(force_orthogonal=True)
sum_sq_error = (sum(i * i for i in model.residuals))
anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

p = anova.loc["Condition", "P-val"].item()
if p < 0.05:
    ax.hlines(y=max*1.10, xmin=0.51, xmax=1.49, linewidth=0.7, color='k')
    ax.vlines(x=0.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=1.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')

ax.set_xticklabels([title.replace(" ", "\n") for title in titles])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Average Distance to the Virtual Humans [cm]")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"distance_test.png"), dpi=300)
plt.close()

# Interpersonal Distance: Correlation with SPAI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
boxWidth = 1
pos = [1]
conditions = ["friendly", "unfriendly"]
titles = ["Friendly Person", "Unfriendly Person"]

for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = conditions[idx_condition]
    df_cond = df_grouped.loc[df_grouped['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.dropna(subset="distance")
    df_cond = df_cond.sort_values(by="SPAI")

    x = df_cond["SPAI"].to_numpy()
    y = df_cond["distance"].to_numpy()
    linreg = linregress(x, y)
    all_x = df_spai["SPAI"].to_numpy()
    all_y = df_cond["distance"].to_numpy()
    all_y_est = linreg.slope * all_x + linreg.intercept
    all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
        1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

    # Plot regression line
    ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
    ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

    p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
    if idx_condition == 0:
        ax.text(df_grouped["SPAI"].min() + 0.01 * np.max(x), 0.95 * df_grouped["distance"].max(),
                r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                color=colors[idx_condition])
    else:
        ax.text(df_grouped["SPAI"].min() + 0.01 * np.max(x), 0.91 * df_grouped["distance"].max(),
                r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                color=colors[idx_condition])

    # Plot raw data points
    ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

ax.set_xlabel("SPAI")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Average Distance to the Virtual Humans [cm]")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"distance_test_SA.png"), dpi=300)
plt.close()

# Difference
df_test = df_grouped.copy()
df_spai = df_test[["VP", "SPAI"]].drop_duplicates(subset="VP")
df_diff = df_test.groupby(["VP", "Condition"]).sum().reset_index()
df_diff = df_diff.pivot(index='VP', columns='Condition', values='distance').reset_index()
df_diff = df_diff.fillna(0)
df_diff["difference"] = df_diff["unfriendly"] - df_diff["friendly"]

df_diff = df_diff[["VP", "difference"]].merge(df_spai, on="VP")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
df_diff = df_diff.sort_values(by="SPAI")
colors = ['teal']
x = df_diff["SPAI"].to_numpy()
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
ax.text(df_diff["SPAI"].min() + 0.01 * np.max(x), 0.95 * (df_diff["difference"].max()-df_diff["difference"].min()) + df_diff["difference"].min(),
        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color="grey")

ax.set_title(f"Avoidance vs. Hypervigilance (N = {len(df_diff['VP'].unique())})", fontweight='bold')
# ax.set_ylim([0, max])
ax.set_xlabel("SPAI")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.axhline(0, linewidth=0.8, color="k", linestyle="dashed")
ax.set_ylabel("Difference Average Interpersonal Distance: Unfriendly-Friendly")
ax.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7)],
    ["Hypervigilance", "Avoidance"], loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"distance_test-diff_SPAI.png"), dpi=300)
plt.close()


# Interpersonal Distance (Minimum)
df = pd.read_csv(os.path.join(dir_path, 'Data', 'distance.csv'), decimal='.', sep=';')
df_spai = df.groupby(["VP"])["SPAI"].mean().reset_index()
df_spai = df_spai.sort_values(by="SPAI")
df_phase = df.loc[df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
df_grouped = df_phase.groupby(["VP", "actor"]).min().reset_index()
df_grouped = df_grouped.drop(columns="SPAI")
df_grouped = df_grouped.merge(df_spai, on="VP")
conditions = ["friendly", "unfriendly"]
df_grouped = df_grouped.loc[df_grouped["Condition"].isin(conditions)]
titles = ["Friendly Person", "Unfriendly Person"]
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
df_crit["SPAI"] = (df_crit["SPAI"] - df_crit["SPAI"].mean()) / df_crit["SPAI"].std()

formula = f"distance ~ Condition + SPAI + Condition:SPAI + (1 | VP)"

max = df_grouped["distance"].max()
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
    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(np.mean([pos[0], pos[1]]), max*1.105, p_sign, color='k', horizontalalignment='center')

ax.set_xticklabels([title.replace(" ", "\n") for title in titles])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Minimal Distance to the Virtual Humans [cm]")
ax.set_title("Interpersonal Distance to Virtual Humans (Minimum)", fontweight='bold')
ax.legend(
    [Line2D([0], [0], color=green, marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
     Line2D([0], [0], color=red, marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
    ["Friendly", "Unfriendly"], loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"min_distance_test.png"), dpi=300)
plt.close()

# Interpersonal Distance: Correlation with SPAI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
boxWidth = 1
pos = [1]
conditions = ["friendly", "unfriendly"]
titles = ["Friendly Person", "Unfriendly Person"]

for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = conditions[idx_condition]
    df_cond = df_grouped.loc[df_grouped['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.dropna(subset="distance")
    df_cond = df_cond.sort_values(by="SPAI")

    x = df_cond["SPAI"].to_numpy()
    y = df_cond["distance"].to_numpy()
    linreg = linregress(x, y)
    all_x = df_spai["SPAI"].to_numpy()
    all_y = df_cond["distance"].to_numpy()
    all_y_est = linreg.slope * all_x + linreg.intercept
    all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
        1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

    # Plot regression line
    ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
    ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

    p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
    if idx_condition == 0:
        ax.text(df_grouped["SPAI"].min() + 0.01 * np.max(x), 0.95 * df_grouped["distance"].max(),
                r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                color=colors[idx_condition])
    else:
        ax.text(df_grouped["SPAI"].min() + 0.01 * np.max(x), 0.91 * df_grouped["distance"].max(),
                r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                color=colors[idx_condition])

    # Plot raw data points
    ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

ax.set_xlabel("SPAI")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Minimal Distance to the Virtual Humans [cm]")
ax.set_title("Minimal Interpersonal Distance", fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"min_distance_test_SA.png"), dpi=300)
plt.close()


# Clicks
df = pd.read_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';')
df_subset = df.loc[df["event"].str.contains("Clicked")]
df_subset = df_subset.groupby(["VP", "Condition"])["event"].count().reset_index()
df_subset = df_subset.rename(columns={"event": "click_count"})

df_vp1 = df[["VP", "SPAI"]].drop_duplicates(subset="VP")
df_vp1["Condition"] = "friendly"
df_vp2 = df_vp1.copy()
df_vp2["Condition"] = "unfriendly"
df_vp = pd.concat([df_vp1, df_vp2])

df_subset = df_vp.merge(df_subset, on=["VP", "Condition"], how="outer")
df_subset = df_subset.fillna(0)

conditions = ["friendly", "unfriendly"]
df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]
titles = ["Friendly Person", "Unfriendly Person"]
colors = [green, red]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
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
df_crit["SPAI"] = (df_crit["SPAI"] - df_crit["SPAI"].mean()) / df_crit["SPAI"].std()

formula = f"click_count ~ Condition + SPAI + Condition:SPAI + (1 | VP)"

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
    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(np.mean([pos[0], pos[1]]), max*1.105, p_sign, color='k', horizontalalignment='center')

ax.set_xticklabels([title.replace(" ", "\n") for title in titles])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Number of Clicks on the Virtual Humans")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"clicks_test.png"), dpi=300)
plt.close()

# Clicks: Correlation with SPAI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
boxWidth = 1
pos = [1]
conditions = ["friendly", "unfriendly"]
titles = ["Friendly Person", "Unfriendly Person"]
df_subset = df_subset.sort_values(by="SPAI")

for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = conditions[idx_condition]
    df_cond = df_subset.loc[df_subset['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.dropna(subset="click_count")
    df_cond = df_cond.sort_values(by="SPAI")

    x = df_cond["SPAI"].to_numpy()
    y = df_cond["click_count"].to_numpy()
    linreg = linregress(x, y)
    all_x = df_subset["SPAI"].to_numpy()
    all_y = df_cond["click_count"].to_numpy()
    all_y_est = linreg.slope * all_x + linreg.intercept
    all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
        1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

    # Plot regression line
    ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
    ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

    p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
    if idx_condition == 0:
        ax.text(df_subset["SPAI"].min() + 0.01 * np.max(x), 0.95 * df_subset["click_count"].max(),
                r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                color=colors[idx_condition])
    else:
        ax.text(df_subset["SPAI"].min() + 0.01 * np.max(x), 0.91 * df_subset["click_count"].max(),
                r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                color=colors[idx_condition])

    # Plot raw data points
    y_jittered = [random.uniform(value - 0.1, value + 0.1) for value in y]
    ax.plot(x, y_jittered, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

ax.set_xlabel("SPAI")
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Number of Clicks on the Virtual Humans (Test-Phase)")
ax.set_title(f"Additional Interaction Attempts", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"clicks_test_SA.png"), dpi=300)
plt.close()


# Movement
df = pd.read_csv(os.path.join(dir_path, 'Data', 'movement.csv'), decimal='.', sep=';')
df_dist = pd.read_csv(os.path.join(dir_path, 'Data', 'walking_distance.csv'), decimal='.', sep=';')

vps = df["VP"].unique()
vps.sort()
vps = np.reshape(vps, (-1, 6))

df_spai = df["SPAI"].unique()
df_spai.sort()
cNorm = matplotlib.colors.Normalize(vmin=df_spai.min()-1, vmax=df_spai.max())
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('Blues'))
scalarMap.set_array([])

for vp_block in vps:
    # vp_block = vps[0]
    fig, axes = plt.subplots(nrows=len(vp_block), ncols=2, figsize=(9, 1.5 * len(vp_block)))

    for idx_vp, vp in enumerate(vp_block):
        # vp = vp_block[0]
        df_vp = df.loc[df["VP"] == vp]
        df_vp_dist = df_dist.loc[df_dist["VP"] == vp]
        df_vp = df_vp.dropna(subset="phase")
        index = df_vp.first_valid_index()
        spai = df_vp.loc[index, "SPAI"]

        axes[idx_vp, 0].text(400, -870, f"VP {vp}", color="lightgrey", fontweight="bold", horizontalalignment='left')

        for idx_phase, phase in enumerate(["Habituation", "Test"]):
            # idx_phase, phase = 0, "Habituation"
            df_phase = df_vp.loc[df_vp["phase"].str.contains(phase)]
            df_phase = df_phase.sort_values(by="timestamp")
            df_phase_dist = df_vp_dist.loc[df_vp_dist["phase"].str.contains(phase)]

            walking_distance = df_phase_dist["walking_distance"].item()

            axes[idx_vp, idx_phase].hlines(y=-954, xmin=-1291, xmax=438, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].hlines(y=-409, xmin=-1291, xmax=438, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=430, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-101, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-661, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-1280, ymin=-954, ymax=-409, linewidth=2, color='lightgrey')
            axes[idx_vp, idx_phase].vlines(x=-661, ymin=-739, ymax=-614, linewidth=5, color='white')
            axes[idx_vp, idx_phase].vlines(x=-101, ymin=-554, ymax=-434, linewidth=5, color='white')

            axes[idx_vp, idx_phase].text(np.mean((-1291, 438)), -870, phase, color="k", horizontalalignment='center', fontsize="small")

            axes[idx_vp, idx_phase].text(-1251, -870, f"{round(walking_distance, 2)} m", color="lightgrey", horizontalalignment='right', fontsize="small", fontstyle="italic")

            axes[idx_vp, idx_phase].plot(df_phase["y"], df_phase["x"], lw=0.8, label=phase, c=scalarMap.to_rgba(spai))

            if phase == "Test":
                try:
                    df_cond = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Conditions3")
                    df_cond = df_cond[["VP", "Roles", "Rooms"]]
                    df_cond = df_cond.loc[df_cond["VP"] == int(vp)]
                    df_rooms = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Rooms3")
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
            axes[idx_vp, idx_phase].axis('scaled')
            axes[idx_vp, idx_phase].invert_xaxis()
            axes[idx_vp, idx_phase].invert_yaxis()
            axes[idx_vp, idx_phase].axis('off')

        axes[idx_vp, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"movement_{vp_block[0]}-{vp_block[-1]}.png"), dpi=300, bbox_inches='tight')
    plt.close()


df = pd.read_csv(os.path.join(dir_path, 'Data', 'movement.csv'), decimal='.', sep=';')
df_spai = df["SPAI"].unique()
df_spai.sort()
vps = df["VP"].unique()
vps.sort()

for cutoff in [2.79, np.median(df_spai)]:
    lsa, hsa = 0, 0
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
    for idx_row in [0, 1]:
        for idx_col in [0, 1, 2]:
            axes[idx_row, idx_col].hlines(y=-954, xmin=-1291, xmax=438, linewidth=2, color='lightgrey')
            axes[idx_row, idx_col].hlines(y=-409, xmin=-1291, xmax=438, linewidth=2, color='lightgrey')
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

    cNorm = matplotlib.colors.Normalize(vmin=df_spai.min() - 1, vmax=df_spai.max())
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('Blues'))

    idx_row = 0
    idx_col = 0

    for idx_vp, vp in enumerate(vps):
        # idx_vp = 0
        # vp = vps[idx_vp]
        df_vp = df.loc[df["VP"] == vp]
        df_vp = df_vp.dropna(subset="phase")
        index = df_vp.first_valid_index()
        spai = df_vp.loc[index, "SPAI"]
        idx_row = 1 if spai < cutoff else 0
        if spai < cutoff:
            lsa += 1
        else:
            hsa += 1

        for idx_col, phase in enumerate(["Habituation", "Test"]):
            # idx_phase, phase = 0, "Habituation"
            df_phase = df_vp.loc[df_vp["phase"].str.contains(phase)]
            df_phase = df_phase.sort_values(by="timestamp")

            if phase == "Test":
                try:
                    df_cond = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Conditions3")
                    df_cond = df_cond[["VP", "Roles", "Rooms"]]
                    df_cond = df_cond.loc[df_cond["VP"] == int(vp)]
                    df_rooms = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Rooms3")
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

    if cutoff == 2.79:
        text = f"Cutoff ({round(cutoff, 2)})"
        title = "cutoff"
    else:
        text = f"Median ({round(cutoff, 2)})"
        title = "median"

    axes[0, 0].text(510, np.mean([-954, -409]), f"≥ {text}", color="k", fontstyle="italic", verticalalignment='center', rotation=90)
    axes[0, 0].text(580, np.mean([-954, -409]), f"HSA (N = {hsa})", color="k", verticalalignment='center', rotation=90)
    axes[1, 0].text(510, np.mean([-954, -409]), f"< {text}", color="k", fontstyle="italic", verticalalignment='center', rotation=90)
    axes[1, 0].text(580, np.mean([-954, -409]), f"LSA (N = {lsa})", color="k", verticalalignment='center', rotation=90)

    axes[0, 0].set_title("Habituation", fontweight="bold")
    axes[0, 1].set_title("Test (Option 1)", fontweight="bold")
    axes[0, 2].set_title("Test (Option 2)", fontweight="bold")

    # plt.subplots_adjust(bottom=0.1, right=0.92, top=0.9)
    # cax = plt.axes([0.95, 0.1, 0.01, 0.8])
    # plt.colorbar(scalarMap, cax=cax, ticks=[0, 1, 2, 3, 4], label="SPAI")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"movement_{title}.png"), dpi=300, bbox_inches='tight')
    plt.close()


# Walking Distance
df_dist = pd.read_csv(os.path.join(dir_path, 'Data', 'walking_distance.csv'), decimal='.', sep=';')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
boxWidth = 1
pos = [1]
phases = ["Habituation", "Test"]
titles = ["Habituation", "Test"]
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']

df_dist = df_dist.sort_values(by="SPAI")
for idx_dv, dv in enumerate(['walking_distance', 'average_distance_to_start', 'maximum_distance_to_start']):
    # dv = 'walking_distance'
    formula = f"{dv} ~ phase + SPAI + phase:SPAI + (1 | VP)"

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

        x = df_phase["SPAI"].to_numpy()
        y = df_phase[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_dist["SPAI"].to_numpy()
        all_y = df_phase[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        axes[idx_dv].plot(all_x, all_y_est, '-', color=colors[idx_phase])
        axes[idx_dv].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_phase])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_phase == 0:
            axes[idx_dv].text(df_dist["SPAI"].min() + 0.01 * np.max(x), 0.95 * (df_dist[dv].max() - df_dist[dv].min()) + df_dist[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_phase])
        else:
            axes[idx_dv].text(df_dist["SPAI"].min() + 0.01 * np.max(x), 0.91 * (df_dist[dv].max() - df_dist[dv].min()) + df_dist[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_phase])

        # Plot raw data points
        axes[idx_dv].plot(x, y, 'o', ms=5, mfc=colors[idx_phase], mec=colors[idx_phase], alpha=0.6, label=titles[idx_phase])

    axes[idx_dv].set_xlabel("SPAI")
    axes[idx_dv].grid(color='lightgrey', linestyle='-', linewidth=0.3)
    axes[idx_dv].set_ylabel(f"{dv.replace('_', ' ').title()} [m]")
    axes[idx_dv].set_title(f"{dv.replace('_', ' ').title()}", fontweight="bold")
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"walking_distance_SA.png"), dpi=300)
plt.close()

df_hr = pd.read_csv(os.path.join(dir_path, 'Data', f'hr.csv'), decimal='.', sep=';')
df_hr = df_hr.loc[(df_hr["Phase"].str.contains("Habituation")) | (df_hr["Phase"].str.contains("Test"))]
df_hr.loc[df_hr["Phase"].str.contains("Habituation"), "phase"] = "Habituation"
df_hr.loc[df_hr["Phase"].str.contains("Test"), "phase"] = "Test"
df_hr = df_hr.merge(df_dist[["VP", "phase", "distance"]], on=["VP", "phase"])
df_hr = df_hr.groupby(["VP", "phase"]).mean().reset_index()
linreg = linregress(df_hr["HR (Mean)"], df_hr["distance"])
print(f"r = {linreg.rvalue}, p = {round(linreg.pvalue, 3)}")
