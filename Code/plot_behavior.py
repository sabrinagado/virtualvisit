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
import matplotlib.animation as animation
from scipy.stats import linregress
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
import pymer4

from Code.toolbox import utils

from Code import preproc_scores, preproc_ratings, preproc_behavior

# % ===========================================================================
# Duration
# =============================================================================
# Time spent in rooms
def plot_time_rooms(df, SA_score="SPAI"):
    # df = df_events
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
    colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']
    red = '#E2001A'
    green = '#B1C800'
    rooms = ["Living", "Dining", "Office"]
    phases = ['Habituation', 'Test']

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
            medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_phase])
            fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_phase])
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
                       positions=[pos[idx_phase]],
                       widths=0.8 * boxWidth)

            ax.errorbar(x=pos[idx_phase], y=bootstrapping_dict['mean'],
                        yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                        elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

            if (room == "Office") & (phase == "Test"):
                x = df_phase[SA_score].to_numpy()
                y = df_phase["duration"].to_numpy()
                linreg = linregress(x, y)
                print(f"Correlation between {SA_score} and Duration in Office in Test-Phase: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

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
    print(f"ANOVA: Duration in Rooms (Phase, Room and {SA_score})")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"Room Main Effect, F({round(anova.loc['room', 'NumDF'].item(), 1)}, {round(anova.loc['room', 'DenomDF'].item(), 1)})={round(anova.loc['room', 'F-stat'].item(), 2)}, p={round(anova.loc['room', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['room', 'p_eta_2'].item(), 2)}")
    print(f"Phase Main Effect, F({round(anova.loc['phase', 'NumDF'].item(), 1)}, {round(anova.loc['phase', 'DenomDF'].item(), 1)})={round(anova.loc['phase', 'F-stat'].item(), 2)}, p={round(anova.loc['phase', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['phase', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x Room, F({round(anova.loc[f'phase:room', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:room', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:room', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:room', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:room', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x {SA_score}, F({round(anova.loc[f'phase:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:{SA_score}', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Room x {SA_score}, F({round(anova.loc[f'room:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'room:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'room:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'room:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'room:{SA_score}', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x Room x {SA_score}, F({round(anova.loc[f'phase:room:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:room:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:room:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:room:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:room:{SA_score}', 'p_eta_2'].item(), 2)}")

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


# Time spent in the different rooms of the virtual agents
def plot_time_rooms_agents_static(df, SA_score="SPAI"):
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
                red = '#E2001A'
                green = '#B1C800'
                colors = [green, red]

            # Plot raw data points
            for i in range(len(df_phase)):
                # i = 0
                x = random.uniform(pos[idx_phase] - (0.25 * boxWidth), pos[idx_phase] + (0.25 * boxWidth))
                y = df_phase.reset_index().loc[i, "duration"].item()
                ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
            fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
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
                       positions=[pos[idx_phase]],
                       widths=0.8 * boxWidth)

            ax.errorbar(x=pos[idx_phase], y=bootstrapping_dict['mean'],
                        yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                        elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

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
            p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
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
    print(f"ANOVA: Duration in Rooms (Phase, Condition and {SA_score})")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
    print(f"Phase Main Effect, F({round(anova.loc['phase', 'NumDF'].item(), 1)}, {round(anova.loc['phase', 'DenomDF'].item(), 1)})={round(anova.loc['phase', 'F-stat'].item(), 2)}, p={round(anova.loc['phase', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['phase', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x Condition, F({round(anova.loc[f'phase:Condition', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:Condition', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:Condition', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:Condition', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x {SA_score}, F({round(anova.loc[f'phase:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:{SA_score}', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x Condition x {SA_score}, F({round(anova.loc[f'phase:Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:Condition:{SA_score}', 'p_eta_2'].item(), 2)}")

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

    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.7)


# Time spent in the different rooms: Correlation with SPAI
def plot_time_test_rooms_agents_static_sad(df, SA_score="SPAI"):
    df_subset = df.loc[df["event"].str.contains("Habituation") | df["event"].str.contains("Test")]
    df_subset.loc[df_subset['event'].str.contains("Test"), "phase"] = "Test"
    df_subset.loc[df_subset['event'].str.contains("Habituation"), "phase"] = "Habituation"
    df_subset = df_subset.dropna(subset="duration")
    df_subset = df_subset.groupby(["VP", "phase", "Condition"]).sum(numeric_only=True).reset_index()
    df_subset = df_subset.drop(columns=SA_score)
    df_subset = df_subset.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

    conditions = ["friendly", "unfriendly"]
    df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]
    titles = ["Room with Friendly Agent", "Room with Unfriendly Agent"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

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
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.95 * df_test["duration"].max(), r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color=colors[idx_condition])
        else:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.91 * df_test["duration"].max(), r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color=colors[idx_condition])

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


# Time spent in the different rooms: Correlation with SPAI (Test-Habituation)
def plot_time_diff_rooms_agents_sad(df, SA_score="SPAI"):
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
    df_diff = pd.melt(df_diff, id_vars=['VP', 'SPAI'], value_vars=['friendly', 'unfriendly'], var_name="Condition", value_name="difference")
    df_diff = df_diff.sort_values(by=SA_score)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

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
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_diff[SA_score].min() + 0.01 * np.max(x), 0.95 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(), r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color=colors[idx_condition])
        else:
            ax.text(df_diff[SA_score].min() + 0.01 * np.max(x), 0.91 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(), r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}", color=colors[idx_condition])

        # Plot raw data points
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3, label=titles[idx_condition])

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


# Time spent in the different rooms
def plot_time_rooms_agents_dynamic(df, SA_score="SPAI"):
    # df = df_events
    df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("With")]
    df_test = df_test.dropna(subset="duration")
    df_test = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
    df_test = df_test.drop(columns=SA_score)
    df_test = df_test.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_test = df_test.sort_values(by=SA_score)

    conditions = ["friendly", "unfriendly"]
    titles = ["Room with Friendly Agent", "Room with Unfriendly Agent"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

    boxWidth = 1 / (len(conditions) + 1)
    pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
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
        medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
        fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
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
                   positions=[pos[idx_condition]],
                   widths=0.8 * boxWidth)

        ax.errorbar(x=pos[idx_condition], y=bootstrapping_dict['mean'],
                    yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                    elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

    df_crit = df_test.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"duration ~ Condition + {SA_score} + " \
              f"Condition:{SA_score} + (1 | VP)"

    max = df_test["duration"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    print(f"ANOVA: Duration in Rooms (Condition and {SA_score})")
    print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")

    # estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

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


def plot_time_test_look_agents_dynamic(df, SA_score="SPAI"):
    # df = df_events
    df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("Vis") & ~df["event"].str.contains("Actor")]
    df_test = df_test.loc[~(df_test['event'].str.contains("Friendly") & df_test['event'].str.contains("Unfriendly"))]
    df_test = df_test.loc[~(df_test['event'].str.contains("Neutral") | df_test['event'].str.contains("Unknown"))]
    df_test = df_test.dropna(subset="duration")
    df_test = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
    df_test = df_test.drop(columns=SA_score)
    df_test = df_test.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_test = df_test.sort_values(by=SA_score)

    conditions = ["friendly", "unfriendly"]
    titles = ["Look at Friendly Agent", "Look at Unfriendly Agent"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

    boxWidth = 1 / (len(conditions) + 1)
    pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
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
        medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
        fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
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
                   positions=[pos[idx_condition]],
                   widths=0.8 * boxWidth)

        ax.errorbar(x=pos[idx_condition], y=bootstrapping_dict['mean'],
                    yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                    elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

    df_crit = df_test.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"duration ~ Condition + {SA_score} + " \
              f"Condition:{SA_score} + (1 | VP)"

    max = df_test["duration"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    print(f"ANOVA: Look at Agents (Condition and {SA_score})")
    print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
    # estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

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


# Time spent in the different rooms: Correlation with SPAI
def plot_time_test_rooms_agents_dynamic_sad(df, SA_score="SPAI"):
    df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("With")]
    df_test = df_test.dropna(subset="duration")
    df_test = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
    df_test = df_test.drop(columns=SA_score)
    df_test = df_test.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_test = df_test.sort_values(by=SA_score)

    conditions = ["friendly", "unfriendly"]
    titles = ["Room with Friendly Agent", "Room with Unfriendly Agent"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

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


def plot_time_test_look_agents_dynamic_sad(df, SA_score="SPAI"):
    # df = df_events
    df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("Vis") & ~df["event"].str.contains("Actor")]
    df_test = df_test.loc[~(df_test['event'].str.contains("Friendly") & df_test['event'].str.contains("Unfriendly"))]
    df_test = df_test.loc[~(df_test['event'].str.contains("Neutral") | df_test['event'].str.contains("Unknown"))]
    df_test = df_test.dropna(subset="duration")
    df_test = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
    df_test = df_test.drop(columns=SA_score)
    df_test = df_test.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_test = df_test.sort_values(by=SA_score)

    conditions = ["friendly", "unfriendly"]
    titles = ["Look at Friendly Agent", "Look at Unfriendly Agent"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

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


# Difference duration
def plot_diff_duration(df, wave, SA_score="SPAI"):
    if wave == 1:
        df_test = df.loc[df["event"].str.contains("Test")]
    elif wave == 2:
        df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("Vis") & ~df["event"].str.contains("Actor")]
        df_test = df_test.loc[~(df_test['event'].str.contains("Friendly") & df_test['event'].str.contains("Unfriendly"))]
        df_test = df_test.loc[~(df_test['event'].str.contains("Neutral") | df_test['event'].str.contains("Unknown"))]
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

    df_diff = df_diff[["VP", "difference"]]
    df_diff = df_diff.rename(columns={"difference": "dur_diff"})
    df_diff = df_diff.sort_values(by="VP").reset_index(drop=True)
    return df_diff


# % ===========================================================================
# Interpersonal Distance
# =============================================================================
def plot_interpersonal_distance(df, wave, dist="avg", SA_score="SPAI"):
    # dist = "avg"
    # df = df_distance
    if dist == "avg":
        title = "Average"
    elif dist == "min":
        title = "Minimum"

    conditions = ["friendly", "unfriendly"]
    if wave == 1:
        df_test = df.loc[df["phase"].str.contains("Test")]
    elif wave == 2:
        df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("Vis") & ~df["event"].str.contains("Actor")]
        df_test = df_test.loc[~(df_test['event'].str.contains("Friendly") & df_test['event'].str.contains("Unfriendly"))]
        df_test = df_test.loc[~(df_test['event'].str.contains("Neutral") | df_test['event'].str.contains("Unknown"))]

    df_test = df_test.loc[df_test["Condition"].isin(conditions)]
    titles = ["Friendly Agent", "Unfriendly Agent"]
    if dist == "avg":
        df_grouped = df_test.groupby(["VP", "Condition"]).mean(numeric_only=True).reset_index()
    elif dist == "min":
        df_grouped = df_test.groupby(["VP", "Condition"]).min(numeric_only=True).reset_index()
    df_grouped = df_grouped.drop(columns=SA_score)
    df_grouped = df_grouped.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_grouped = df_grouped.sort_values(by=SA_score)

    red = '#E2001A'
    green = '#B1C800'
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
        medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
        fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
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
                   positions=[pos[idx_condition]],
                   widths=0.8 * boxWidth)

        ax.errorbar(x=pos[idx_condition], y=bootstrapping_dict['mean'],
                    yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                    elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

    df_crit = df_grouped.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"distance ~ Condition + {SA_score} + Condition:{SA_score} + (1 | VP)"

    max = df_grouped["distance"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    print(f"ANOVA: {title} Interpersonal Distance (Condition and {SA_score})")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
    
    # estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

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


# Interpersonal Distance: Correlation with SPAI
def plot_interpersonal_distance_sad(df, wave, dist="avg", SA_score="SPAI"):
    conditions = ["friendly", "unfriendly"]
    if wave == 1:
        df_test = df.loc[df["phase"].str.contains("Test")]
    elif wave == 2:
        df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("Vis") & ~df["event"].str.contains("Actor")]
        df_test = df_test.loc[~(df_test['event'].str.contains("Friendly") & df_test['event'].str.contains("Unfriendly"))]
        df_test = df_test.loc[~(df_test['event'].str.contains("Neutral") | df_test['event'].str.contains("Unknown"))]

    if dist == "avg":
        df_grouped = df_test.groupby(["VP", "Condition"]).mean(numeric_only=True).reset_index()
        title = "Average"
    elif dist == "min":
        df_grouped = df_test.groupby(["VP", "Condition"]).min(numeric_only=True).reset_index()
        title = "Minimum"
    df_grouped = df_grouped.drop(columns=SA_score)
    df_grouped = df_grouped.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")
    df_grouped = df_grouped.sort_values(by=SA_score)

    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
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
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3, label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"{title} Distance to the Virtual Agents [m]")
    # ax.set_title(f"{title} Interpersonal Distance", fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()


# Distance to virtual agents (Comparison to Habituation)
def plot_interpersonal_distance_diff(df, dist="avg", SA_score="SPAI"):
    if dist == "avg":
        df_subset = df.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
        title = "Average"
    elif dist == "min":
        df_subset = df.groupby(["VP", "phase", "Condition"]).min(numeric_only=True).reset_index()
        title = "Minimum"
    df_subset = df_subset.drop(columns=SA_score)
    df_subset = df_subset.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

    conditions = ["friendly", "unfriendly"]
    phases = ['Habituation', 'Test']
    titles = ["Position of\nFriendly Agent", "Position of\nUnfriendly Agent"]
    df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    boxWidth = 1 / (len(conditions) + 1)

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
                red = '#E2001A'
                green = '#B1C800'
                colors = [green, red]

            # Plot raw data points
            for i in range(len(df_phase)):
                # i = 0
                x = random.uniform(pos[idx_phase] - (0.25 * boxWidth), pos[idx_phase] + (0.25 * boxWidth))
                y = df_phase.reset_index().loc[i, "distance"].item()
                ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
            fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
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
                       positions=[pos[idx_phase]],
                       widths=0.8 * boxWidth)

            ax.errorbar(x=pos[idx_phase], y=bootstrapping_dict['mean'],
                        yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                        elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

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
    print(f"ANOVA: {title} Interpersonal Distance (Phase, Condition and {SA_score})")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
    print(f"Phase Main Effect, F({round(anova.loc['phase', 'NumDF'].item(), 1)}, {round(anova.loc['phase', 'DenomDF'].item(), 1)})={round(anova.loc['phase', 'F-stat'].item(), 2)}, p={round(anova.loc['phase', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['phase', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x Condition, F({round(anova.loc[f'phase:Condition', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:Condition', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:Condition', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:Condition', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x {SA_score}, F({round(anova.loc[f'phase:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:{SA_score}', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Phase x Condition x {SA_score}, F({round(anova.loc[f'phase:Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:Condition:{SA_score}', 'p_eta_2'].item(), 2)}")

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

    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.7)


# Interpersonal Distance: Correlation with SPAI
def plot_interpersonal_distance_diff_sad(df, dist="avg", SA_score="SPAI"):
    if dist == "avg":
        df_subset = df.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
        title = "Average"
    elif dist == "min":
        df_subset = df.groupby(["VP", "phase", "Condition"]).min(numeric_only=True).reset_index()
        title = "Minimum"
    df_subset = df_subset.drop(columns=SA_score)
    df_subset = df_subset.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

    conditions = ["friendly", "unfriendly"]
    df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]

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
    conditions = ["friendly", "unfriendly"]
    titles = ["Friendly Agent", "Unfriendly Agent"]
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

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


# Difference distance
def plot_diff_distance(df, wave, SA_score="SPAI"):
    # df = df_distance
    if wave == 1:
        df_test = df.loc[df["phase"].str.contains("Test")]
    elif wave == 2:
        df_test = df.loc[df["event"].str.contains("Test") & df["event"].str.contains("Vis") & ~df["event"].str.contains("Actor")]
        df_test = df_test.loc[~(df_test['event'].str.contains("Friendly") & df_test['event'].str.contains("Unfriendly"))]
        df_test = df_test.loc[~(df_test['event'].str.contains("Neutral") | df_test['event'].str.contains("Unknown"))]

    df_test = df_test.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
    df_test = df_test.dropna(subset="distance")
    df_spai = df_test[["VP", SA_score]].drop_duplicates(subset="VP")
    df_test = df_test.loc[df_test["Condition"].isin(["friendly", "unfriendly"])]
    df_diff = df_test.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
    df_diff = df_diff.pivot(index='VP', columns='Condition', values='distance').reset_index()
    df_diff = df_diff.fillna(0)
    df_diff["difference"] = df_diff["unfriendly"] - df_diff["friendly"]

    df_diff = df_diff[["VP", "difference"]].merge(df_spai, on="VP")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    df_diff = df_diff.sort_values(by=SA_score)
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
    c = np.where(y < 0, 'gold', 'teal')
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
    ax.set_ylabel("Difference Average Interpersonal Distance: Unfriendly-Friendly")
    ax.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7)],
        ["Approach", "Avoidance"], loc="upper right")

    plt.tight_layout()

    df_diff = df_diff[["VP", "difference"]]
    df_diff = df_diff.rename(columns={"difference": "dis_diff"})
    df_diff = df_diff.sort_values(by="VP").reset_index(drop=True)
    return df_diff


# % ===========================================================================
# Clicks
# =============================================================================
def plot_clicks(df, SA_score="SPAI"):
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
    red = '#E2001A'
    green = '#B1C800'
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
        medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
        fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
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
                   positions=[pos[idx_condition]],
                   widths=0.8 * boxWidth)

        ax.errorbar(x=pos[idx_condition], y=bootstrapping_dict['mean'],
                    yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                    elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

    df_crit = df_subset.copy()
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"click_count ~ Condition + {SA_score} + Condition:{SA_score} + (1 | VP)"

    max = df_subset["click_count"].max()
    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    print(f"ANOVA: Clicks (Condition and {SA_score})")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")

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


# Clicks: Correlation with SPAI
def plot_clicks_sad(df, SA_score="SPAI"):
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
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
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


# % ===========================================================================
# Movement
# =============================================================================
def plot_movement_single_plots(df, wave, SA_score="SPAI"):
    if wave == 2:
        df["x"] = df["x_player"]
        df["y"] = df["y_player"]
    df["distance_to_previous_scaled"] = df["distance_to_previous_player_scaled"]

    df_spai = df[SA_score].unique()
    df_spai.sort()
    df_spai = df_spai[~np.isnan(df_spai)]
    cNorm = matplotlib.colors.Normalize(vmin=np.min(df_spai) - 0.1 * np.max(df_spai), vmax=np.max(df_spai) + 0.1 * np.max(df_spai))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('viridis_r'))
    scalarMap.set_array([])

    vps = df["VP"].unique()
    vps.sort()
    vps = np.reshape(vps, (-1, 8))

    red = '#E2001A'
    green = '#B1C800'

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


# Movement per SA-Group
def plot_movement_sad(df, filepath, SA_score="SPAI", cutoff="cutoff"):
    df_spai = list(df.drop_duplicates(subset="VP")[SA_score])
    df_spai.sort()
    cNorm = matplotlib.colors.Normalize(vmin=np.min(df_spai) - 0.1 * np.max(df_spai), vmax=np.max(df_spai) + 0.1 * np.max(df_spai))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('viridis_r'))
    vps = df["VP"].unique()
    vps.sort()
    cutoff_sa = 2.79 if SA_score == "SPAI" else 30

    red = '#E2001A'
    green = '#B1C800'
    if cutoff == "cutoff":
        text = f"Cutoff ({round(cutoff_sa, 2)})"
        cutoff_value = cutoff_sa
    elif cutoff == "median":
        text = f"Median ({round(np.median(df_spai), 2)})"
        cutoff_value = np.median(df_spai)

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

    idx_row, idx_col = 0, 0
    for idx_vp, vp in enumerate(vps):
        # idx_vp = 0
        # vp = vps[idx_vp]
        df_vp = df.loc[df["VP"] == vp]
        df_vp = df_vp.dropna(subset="phase")
        index = df_vp.first_valid_index()
        spai = df_vp.loc[index, SA_score]
        idx_row = 1 if spai < cutoff_value else 0
        if spai < cutoff_value:
            lsa += 1
        else:
            hsa += 1

        for idx_col, phase in enumerate(["Habituation", "Test"]):
            # idx_phase, phase = 1, "Test"
            df_phase = df_vp.loc[df_vp["phase"].str.contains(phase)]
            df_phase = df_phase.sort_values(by="time")

            if phase == "Test":
                # Get Conditions
                df_roles = preproc_behavior.get_conditions(f"0{vp}" if vp < 10 else f"{vp}", filepath)

                if (df_roles.loc[df_roles["Role"] == "friendly", "Rooms"] == "Dining").item():
                    idx_col += 1

                for room in ["Dining", "Living"]:
                    # room = "Dining"
                    role = df_roles.loc[df_roles["Rooms"] == room, "Role"].item()
                    color = green if role == "friendly" else red
                    if room == "Dining":
                        position_x = -490
                        position_y = -1034
                    else:
                        position_x = -870
                        position_y = 262
                    circle = patches.Circle((position_y, position_x), radius=30, color=color, alpha=0.5)
                    axes[idx_row, idx_col].add_patch(circle)
            axes[idx_row, idx_col].plot(df_phase["y"], df_phase["x"], lw=0.8, label=phase,  c=scalarMap.to_rgba(spai))

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


# Animated Movement
def update_animation(num, data_hab, data_test, data_friendly, data_unfriendly, ax1, ax2, vp, spai, scalarMap):
    red = '#E2001A'
    green = '#B1C800'

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

    ax1.scatter(data_hab[0][num], data_hab[1][num], label="Habituation", color=scalarMap.to_rgba(spai), alpha=0.8, marker="o", s=20, zorder=5)
    ax1.axis('scaled')
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.axis('off')

    ax2.scatter(data_test[0][num], data_test[1][num], label="Test", color=scalarMap.to_rgba(spai), alpha=0.8, marker="o", s=20, zorder=5)
    ax2.scatter(data_friendly[0][num], data_friendly[1][num], label="Friendly", color=green, alpha=0.8, marker="s", s=15, zorder=5)
    ax2.scatter(data_unfriendly[0][num], data_unfriendly[1][num], label="Unfriendly", color=red, alpha=0.8, marker="s", s=15, zorder=5)
    ax2.axis('scaled')
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.axis('off')

    ax2.axis('off')


def animate_movement(df, vp, SA_score, save_path):
    # vp = 5
    df_spai = df.drop_duplicates(subset="VP").reset_index(drop=True)
    idx_vp = (df_spai["VP"] == vp).idxmax()
    spais = list(df_spai[SA_score])
    spais.sort()
    spai = spais[idx_vp]
    cNorm = matplotlib.colors.Normalize(vmin=np.min(spais) - 0.1 * np.max(spais), vmax=np.max(spais) + 0.1 * np.max(spais))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('viridis_r'))

    Writer = animation.writers['ffmpeg']
    Writer = Writer(fps=30, metadata=dict(title=f"Visualization of VP {vp}"), bitrate=-1)

    df_vp = df.loc[df["VP"] == vp]
    df_vp = df_vp.dropna(subset="phase")

    # Habituation:
    df_hab = df_vp.loc[df_vp["phase"].str.contains("Habituation")]
    df_hab = df_hab.sort_values(by="time")

    x = df_hab["y_player"].to_list()
    y = df_hab["x_player"].to_list()
    data_hab = np.array([x, y])

    # Test
    df_test = df_vp.loc[df_vp["phase"].str.contains("Test")]
    df_test = df_test.sort_values(by="time")

    x = df_test["y_player"].to_list()
    y = df_test["x_player"].to_list()
    data_test = np.array([x, y])

    x = df_test["y_friendly"].to_list()
    y = df_test["x_friendly"].to_list()
    data_friendly = np.array([x, y])

    x = df_test["y_unfriendly"].to_list()
    y = df_test["x_unfriendly"].to_list()
    data_unfriendly = np.array([x, y])

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 1.5))

    line_animation = animation.FuncAnimation(fig, update_animation, frames=900, fargs=(data_hab, data_test, data_friendly, data_unfriendly, ax1, ax2, vp, spai, scalarMap))
    line_animation.save(os.path.join(save_path, f"movement_{vp}.mp4"), writer=Writer, dpi=300)

    plt.close()


# % ===========================================================================
# Walking Distance
# =============================================================================
def plot_walking_distance(df, SA_score="SPAI"):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
    phases = ["Habituation", "Test"]
    titles = ["Habituation", "Test"]
    colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']

    df = df.sort_values(by=SA_score)
    for idx_dv, dv in enumerate(['walking_distance', 'average_distance_to_start', 'maximum_distance_to_start']):
        # dv = 'maximum_distance_to_start'
        formula = f"{dv} ~ phase + {SA_score} + phase:{SA_score} + (1 | VP)"

        model = pymer4.models.Lmer(formula, data=df)
        model.fit(factors={"phase": ["Habituation", "Test"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        sum_sq_error = (sum(i * i for i in model.residuals))
        anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
        print(f"ANOVA: {dv.replace('_', ' ').title()} (Phase and {SA_score})")
        print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
        print(f"Phase Main Effect, F({round(anova.loc['phase', 'NumDF'].item(), 1)}, {round(anova.loc['phase', 'DenomDF'].item(), 1)})={round(anova.loc['phase', 'F-stat'].item(), 2)}, p={round(anova.loc['phase', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['phase', 'p_eta_2'].item(), 2)}")
        print(f"Interaction Phase x {SA_score}, F({round(anova.loc[f'phase:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:{SA_score}', 'p_eta_2'].item(), 2)}")

        # estimates, contrasts = model.post_hoc(marginal_vars="phase", p_adjust="holm")

        for idx_phase, phase in enumerate(phases):
            # idx_phase = 0
            # phase = phases[idx_phase]
            df_phase = df.loc[df['phase'] == phase].reset_index(drop=True)
            df_phase = df_phase.dropna(subset=dv)

            x = df_phase[SA_score].to_numpy()
            y = df_phase[dv].to_numpy()
            linreg = linregress(x, y)
            all_x = df[SA_score].to_numpy()
            all_y = df_phase[dv].to_numpy()
            all_y_est = linreg.slope * all_x + linreg.intercept
            all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
                1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

            # Plot regression line
            axes[idx_dv].plot(all_x, all_y_est, '-', color=colors[idx_phase])
            axes[idx_dv].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_phase])

            p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
            if idx_phase == 0:
                axes[idx_dv].text(df[SA_score].min() + 0.01 * np.max(x), 0.95 * (df[dv].max() - df[dv].min()) + df[dv].min(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_phase])
            else:
                axes[idx_dv].text(df[SA_score].min() + 0.01 * np.max(x), 0.91 * (df[dv].max() - df[dv].min()) + df[dv].min(),
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


if __name__ == '__main__':
    wave = 2
    dir_path = os.getcwd()
    filepath = os.path.join(dir_path, f'Data-Wave{wave}')

    save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Behavior')
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

    df_events = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';')
    df_distance = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance_vh.csv'), decimal='.', sep=';')
    df_movement = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';')
    df_walk_dist = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'walking_distance.csv'), decimal='.', sep=';')

    SA_score = "SPAI"

    if wave == 1:
        plot_time_rooms(df_events)
        plt.savefig(os.path.join(save_path, f"duration_rooms.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plot_time_rooms_agents_static(df_events)
        plt.savefig(os.path.join(save_path, f"duration_hab-test.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plot_time_test_rooms_agents_static_sad(df_events, SA_score)
        plt.savefig(os.path.join(save_path, f"duration_test_{SA_score}.png"), dpi=300)
        plt.close()

        plot_time_diff_rooms_agents_sad(df_events, SA_score)
        plt.savefig(os.path.join(save_path, f"duration_diff_{SA_score}.png"), dpi=300)
        plt.close()

    if wave == 2:
        plot_time_rooms_agents_dynamic(df_events, SA_score)
        plt.savefig(os.path.join(save_path, f"duration_test.png"), dpi=300)
        plt.close()

        plot_time_test_rooms_agents_dynamic_sad(df_events, SA_score)
        plt.savefig(os.path.join(save_path, f"duration_test_{SA_score}.png"), dpi=300)
        plt.close()

    df_diff_dur = plot_diff_duration(df_events, SA_score)
    plt.savefig(os.path.join(save_path, f"duration_test-diff_{SA_score}.png"), dpi=300)
    plt.close()

    for dist in ["avg", "min"]:
        plot_interpersonal_distance(df_distance, "avg", SA_score)
        plt.savefig(os.path.join(save_path, f"distance_{dist}_test.png"), dpi=300)
        plt.close()

        plot_interpersonal_distance_sad(df_distance, "avg", SA_score)
        plt.savefig(os.path.join(save_path, f"distance_{dist}_test_{SA_score}.png"), dpi=300)
        plt.close()

    df_diff_dis = plot_diff_distance(df_distance, SA_score)

    if wave == 1:
        plot_interpersonal_distance_diff(df_distance, "avg", SA_score)
        plt.savefig(os.path.join(save_path, f"distance_{dist}_hab-test.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plot_interpersonal_distance_diff_sad(df_distance, "avg", SA_score)
        plt.savefig(os.path.join(save_path, f"distance_{dist}_diff_{SA_score}.png"), dpi=300)
        plt.close()

        plot_clicks(df_events, SA_score)
        plt.savefig(os.path.join(save_path, f"clicks_test.png"), dpi=300)
        plt.close()

        plot_clicks_sad(df_events, SA_score)
        plt.savefig(os.path.join(save_path, f"clicks_test_{SA_score}.png"), dpi=300)
        plt.close()

        plot_movement_single_plots(df_movement, wave, SA_score)

        cutoff = "median"
        plot_movement_sad(df_movement, SA_score, cutoff)
        plt.savefig(os.path.join(save_path, f"movement_{cutoff}_{SA_score}.png"), dpi=300, bbox_inches='tight')

    if wave == 2:
        animate_movement(df_movement, 6, SA_score, save_path)

    plot_walking_distance(df_walk_dist, SA_score)
    plt.savefig(os.path.join(save_path, f"walking_distance_grouped_{SA_score}.png"), dpi=300)
    plt.close()

    # df_hr = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', f'hr.csv'), decimal='.', sep=';')
    # df_hr = df_hr.loc[(df_hr["Phase"].str.contains("Habituation")) | (df_hr["Phase"].str.contains("Test"))]
    # df_hr.loc[df_hr["Phase"].str.contains("Habituation"), "phase"] = "Habituation"
    # df_hr.loc[df_hr["Phase"].str.contains("Test"), "phase"] = "Test"
    # df_hr = df_hr.merge(df_dist[["VP", "phase", "walking_distance"]], on=["VP", "phase"])
    # df_hr = df_hr.groupby(["VP", "phase"]).mean().reset_index()
    # linreg = linregress(df_hr["HR (Mean)"], df_hr["walking_distance"])
    # print(f"r = {linreg.rvalue}, p = {round(linreg.pvalue, 3)}")
