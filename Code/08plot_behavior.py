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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
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

# Time spent in Rooms
df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
df_subset.loc[df_subset['event'].str.contains("Office"), "room"] = "Office"
df_subset.loc[df_subset['event'].str.contains("Living"), "room"] = "Living"
df_subset.loc[df_subset['event'].str.contains("Dining"), "room"] = "Dining"
df_subset = df_subset.dropna(subset="duration")
df_subset = df_subset.groupby(["VP", "phase", "room"]).sum().reset_index()
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
        bootstrapping_dict = bootstrapping(df_phase.loc[:, "duration"].values,
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

    lm = smf.ols(formula, data=df_room).fit()
    anova = sm.stats.anova_lm(lm, typ=3)
    sum_sq_error = anova.loc["Residual", "sum_sq"]
    anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

    p = anova.loc["phase", "PR(>F)"].item()
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

lm = smf.ols(formula, data=df_crit).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

contrasts = sp.posthoc_ttest(df_crit, val_col='duration', group_col='room', p_adjust='holm')
max = df_subset["duration"].max()
p_con = contrasts.loc["Dining", "Living"].item()
if p_con < 0.05:
    ax.hlines(y=max*1.10, xmin=0.51, xmax=1.49, linewidth=0.7, color='k')
    ax.vlines(x=0.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=1.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else ""
    ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')
p_con = contrasts.loc["Dining", "Office"].item()
if p_con < 0.05:
    ax.hlines(y=max*1.10, xmin=1.51, xmax=2.49, linewidth=0.7, color='k')
    ax.vlines(x=1.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=2.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p_con < 0.001 else "**" if p_con < 0.01 else "*" if p_con < 0.05 else ""
    ax.text(2, max*1.105, p_sign, color='k', horizontalalignment='center')
p_con = contrasts.loc["Living", "Office"].item()
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
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"duration_rooms{end}"), dpi=300, bbox_inches="tight")
plt.close()


# Time spent in the different rooms of the virtual humans
df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
df_subset.loc[df_subset['event'].str.contains("Office"), "room"] = "Office"
df_subset.loc[df_subset['event'].str.contains("Living"), "room"] = "Living"
df_subset.loc[df_subset['event'].str.contains("Dining"), "room"] = "Dining"
df_subset = df_subset.dropna(subset="duration")
df_subset = df_subset.groupby(["VP", "phase", "room", "Condition"]).sum().reset_index()
df_subset = df_subset.drop(columns="SPAI")
df_subset = df_subset.merge(df[["VP", "SPAI"]].drop_duplicates(subset="VP"), on="VP")

conditions = ["friendly", "unfriendly"]
phases = ['Habituation', 'Test']
titles = ["Room with Friendly Person", "Room with Unfriendly Person"]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
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
        bootstrapping_dict = bootstrapping(df_phase.loc[:, "duration"].values,
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

    lm = smf.ols(formula, data=df_cond).fit()
    anova = sm.stats.anova_lm(lm, typ=3)
    sum_sq_error = anova.loc["Residual", "sum_sq"]
    anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

    p = anova.loc["phase", "PR(>F)"].item()
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

lm = smf.ols(formula, data=df_crit).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

max = df_subset["duration"].max()
p = anova.loc["Condition", "PR(>F)"].item()
if p < 0.05:
    ax.hlines(y=max*1.10, xmin=0.51, xmax=1.49, linewidth=0.7, color='k')
    ax.vlines(x=0.51, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=1.49, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(1, max*1.105, p_sign, color='k', horizontalalignment='center')

df_crit = df_subset.loc[df_subset["phase"].str.contains("Test")]
formula = f"duration ~ Condition + (1 | VP)"

lm = smf.ols(formula, data=df_crit).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

p = anova.loc["Condition", "PR(>F)"].item()
max = df_subset["duration"].max()
if p < 0.05:
    ax.hlines(y=max*1.10, xmin=2*boxWidth, xmax=1+2*boxWidth, linewidth=0.7, color='k')
    ax.vlines(x=2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=1+2*boxWidth, ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(np.mean([2*boxWidth, 1+2*boxWidth]), max*1.105, p_sign, color='k', horizontalalignment='center')

ax.set_xticks([x + 1 / 2 for x in range(len(conditions))])
ax.set_xticklabels([title.replace("with", "with\n") for title in titles])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Total Duration in the Rooms [s]")

fig.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
    ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
fig.subplots_adjust(right=0.76)
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"duration_test{end}"), dpi=300, bbox_inches="tight")
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
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"duration_test_SA{end}"), dpi=300)
plt.close()


# Difference
df_test = df_subset.loc[df_subset['phase'] == "Test"]
df_spai = df_test[["VP", "SPAI"]].drop_duplicates(subset="VP")
df_diff = df_test.groupby(["VP", "Condition"]).sum().reset_index()
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
    [Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7)],
    ["Avoidance", "Hypervigilance"], loc="best")

plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"duration_test-diff_SPAI{end}"), dpi=300)
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
df_grouped = df_grouped.loc[df_grouped["Condition"].isin(conditions)]
titles = ["Friendly Person", "Unfriendly Person"]

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
    bootstrapping_dict = bootstrapping(df_cond.loc[:, "distance"].values,
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

lm = smf.ols(formula, data=df_crit).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

max = df_grouped["distance"].max()
p = anova.loc["Condition", "PR(>F)"].item()
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
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"distance_test{end}"), dpi=300)
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
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"distance_test_SA{end}"), dpi=300)
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
    [Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7)],
    ["Avoidance", "Hypervigilance"], loc="best")

plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"distance_test-diff_SPAI{end}"), dpi=300)
plt.close()

# Interpersonal Distance
df = pd.read_csv(os.path.join(dir_path, 'Data', 'distance.csv'), decimal='.', sep=';')
df_spai = df.groupby(["VP"])["SPAI"].mean().reset_index()
df_spai = df_spai.sort_values(by="SPAI")
df_phase = df.loc[df["event"].str.contains("Test") & ~(df["event"].str.contains("Clicked"))]
df_grouped = df_phase.groupby(["VP", "Condition"]).min().reset_index()
df_grouped = df_grouped.drop(columns="SPAI")
df_grouped = df_grouped.merge(df_spai, on="VP")
conditions = ["friendly", "unfriendly"]
df_grouped = df_grouped.loc[df_grouped["Condition"].isin(conditions)]
titles = ["Friendly Person", "Unfriendly Person"]

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
    bootstrapping_dict = bootstrapping(df_cond.loc[:, "distance"].values,
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

lm = smf.ols(formula, data=df_crit).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

max = df_grouped["distance"].max()
p = anova.loc["Condition", "PR(>F)"].item()
if p < 0.05:
    ax.hlines(y=max*1.10, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
    ax.vlines(x=pos[0], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    ax.vlines(x=pos[1], ymin=max*1.09, ymax=max*1.10, linewidth=0.7, color='k')
    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(np.mean([pos[0], pos[1]]), max*1.105, p_sign, color='k', horizontalalignment='center')

ax.set_xticklabels([title.replace(" ", "\n") for title in titles])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(f"Minimal Distance to the Virtual Humans [cm]")
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"min_distance_test{end}"), dpi=300)
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
ax.set_ylabel(f"Minimal Distance to the Virtual Humans [cm]")
ax.legend()
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"min_distance_test_SA{end}"), dpi=300)
plt.close()
