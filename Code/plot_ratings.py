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

from Code import preproc_scores, preproc_ratings


# Ratings VR
def plot_rating_vr(df):
    # df = df_ratings
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    phases = ['Orientation', 'Habituation', 'Test']
    colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']
    boxWidth = 1

    for idx_phase, phase in enumerate(phases):
        # idx_phase = 0
        # phase = phases[idx_phase]
        df_phase = df.loc[(df["Phase"] == phase) & (df["Object"] == "VR")]
        df_phase = df_phase.dropna(subset="Value")

        pos = idx_phase

        # Plot raw data points
        for i in range(len(df_phase)):
            # i = 0
            x = random.uniform(pos - (0.25 * boxWidth), pos + (0.25 * boxWidth))
            y = df_phase.reset_index().loc[i, "Value"].item()
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
        bootstrapping_dict = utils.bootstrapping(df_phase.loc[:, "Value"].values,
                                                 numb_iterations=5000,
                                                 alpha=alpha,
                                                 as_dict=True,
                                                 func='mean')

        ax.boxplot([df_phase.loc[:, "Value"].values],
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
                                positions=[pos],
                                widths=0.8 * boxWidth)

        ax.errorbar(x=pos, y=bootstrapping_dict['mean'],
                                 yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                                 elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

        if phase == "Test":
            linreg = linregress(df_phase.loc[:, "Value"], df_phase.loc[:, "SSQ-diff"])
            print(f"Correlation between Wellbeing after Test-Phase and SSQ-diff r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

    ax.set_xticklabels(phases, fontsize="x-large")
    ax.set_ylim(-1, 101)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel("Subjective Wellbeing", fontsize="x-large")
    plt.tight_layout()


# Ratings Rooms
def plot_rating_rooms(df, wave):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
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
            # idx_phase = 0
            # phase = phases[idx_phase]
            df_phase = df.loc[(df["Phase"] == phase) & (df["Object"] == room)]
            df_phase = df_phase.dropna(subset="Value")

            if (phase == "Test") & ((room == "Living") | (room == "Dining")) & (wave == 1):
                conditions = ['friendly', 'unfriendly']
                boxWidth_room = boxWidth/2
                pos_room = [pos[idx_phase] - boxWidth_room*0.5, pos[idx_phase] + boxWidth_room]

                for idx_condition, condition in enumerate(conditions):
                    # idx_condition = 0
                    # condition = conditions[idx_condition]
                    df_cond = df_phase.loc[df_phase["Condition"] == condition]
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
                    medianlineprops = dict(linestyle='dashed', linewidth=1, color=color)
                    fliermarkerprops = dict(marker='o', markersize=1, color=color)
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
                                            positions=[pos_room[idx_condition]],
                                            widths=0.8 * boxWidth_room)

                    ax.errorbar(x=pos_room[idx_condition], y=bootstrapping_dict['mean'],
                                             yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                                             elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

            else:
                # Plot raw data points
                for i in range(len(df_phase)):
                    # i = 0
                    x = random.uniform(pos[idx_phase] - (0.2 * boxWidth), pos[idx_phase] + (0.2 * boxWidth))
                    y = df_phase.reset_index().loc[i, "Value"].item()
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
                bootstrapping_dict = utils.bootstrapping(df_phase.loc[:, "Value"].values,
                                                         numb_iterations=5000,
                                                         alpha=alpha,
                                                         as_dict=True,
                                                         func='mean')

                ax.boxplot([df_phase.loc[:, "Value"].values],
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

        if not room == "Terrace":
            df_crit = df.loc[df["Object"] == room]
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
            anova['p_eta_2'] = anova.apply(lambda x: utils.partial_eta_squared(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)
            anova['p_eta_2_CI'] = anova.apply(lambda x: utils.partial_eta_squared_ci(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)

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
    ax.set_xticklabels(rooms, fontsize="x-large")
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Subjective Wellbeing", fontsize="x-large")
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    if wave == 1:
        fig.legend(
            [Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7),
             Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7),
             Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
             Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
            ["Habituation", "Test", "Test (friendly)", "Test (unfriendly)"], loc="center right")
    elif wave == 2:
        fig.legend(
            [Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[0], markeredgewidth=1, markerfacecolor=colors[0], alpha=.7),
             Line2D([0], [0], color="white", marker='o', markeredgecolor=colors[1], markeredgewidth=1, markerfacecolor=colors[1], alpha=.7)],
            ["Habituation", "Test"], loc="center right")
    fig.subplots_adjust(right=0.82)


# Ratings Virtual Humans
def plot_rating_agents(df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 5))
    labels = ["Likeability", "Fear", "Anger", "Attractiveness", "Behavior"]
    conditions = ['Friendly', 'Unfriendly', 'Neutral', 'Unknown']
    colors = ['#B1C800', '#E2001A', '#1F82C0',  'lightgrey']

    for idx_label, label in enumerate(labels):
        # idx_label = 0
        # label = labels[idx_label]
        # print(f"\n{label}")

        boxWidth = 1 / (len(conditions) + 2)
        pos = [idx_label + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

        for idx_condition, condition in enumerate(conditions):
            # idx_condition = 1
            # condition = conditions[idx_condition]
            df_cond = df.loc[(df["Criterion"] == label) & (df["Condition"] == condition.lower())]
            df_cond = df_cond.dropna(subset="Value")

            # Plot raw data points
            for i in range(len(df_cond)):
                # i = 0
                x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
                y = df_cond.reset_index().loc[i, "Value"].item()
                ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3, label=label)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
            fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
            boxprops = dict(color=colors[idx_condition])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, "Value"].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            ax.boxplot([df_cond.loc[:, "Value"].values],
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

            ax.errorbar(x=pos[idx_condition], y=bootstrapping_dict['mean'], yerr=bootstrapping_dict['mean']-bootstrapping_dict['lower'],
                        elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

        df_crit = df.loc[df["Criterion"] == label]
        df_crit = df_crit.loc[~(df_crit["Condition"] == "unknown")]
        formula = f"Value ~ Condition + (1 | VP)"
        model = pymer4.models.Lmer(formula, data=df_crit)
        model.fit(factors={"Condition": ["friendly", "neutral", "unfriendly"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        anova['p_eta_2'] = anova.apply(lambda x: utils.partial_eta_squared(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)
        anova['p_eta_2_CI'] = anova.apply(lambda x: utils.partial_eta_squared_ci(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)

        estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")
        p = anova.loc["Condition", "P-val"].item()

        # print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")

        max = 100  # df_crit["Value"].max()
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
    ax.set_xticklabels(labels, fontsize="x-large")
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    fig.legend(
        [Line2D([0], [0], marker='s', ms=15, markeredgecolor=colors[0], markeredgewidth=2, markerfacecolor="white", lw=0),
         Line2D([0], [0], marker='s', ms=15, markeredgecolor=colors[1], markeredgewidth=2, markerfacecolor="white", lw=0),
         Line2D([0], [0], marker='s', ms=15, markeredgecolor=colors[2], markeredgewidth=2, markerfacecolor="white", lw=0),
         Line2D([0], [0], marker='s', ms=15, markeredgecolor=colors[3], markeredgewidth=2, markerfacecolor="white", lw=0)],
        ["Friendly\nAgent", "Unfriendly\nAgent", "Neutral\nAgent", "Unknown\nAgent"], loc="center right", fontsize="x-large")
    fig.subplots_adjust(right=0.87)
    # plt.tight_layout()


# Correlation of Ratings
def corr_ratings(df):
    df_like = df.loc[(df["Criterion"] == "Likeability"), ["VP", "Phase", "Object", "Criterion", "Value"]]
    df_fear = df.loc[(df["Criterion"] == "Fear"), ["VP", "Phase", "Object", "Criterion", "Value"]]
    df_anger = df.loc[(df["Criterion"] == "Anger"), ["VP", "Phase", "Object", "Criterion", "Value"]]
    df_reliability_ratings = df_like.merge(df_fear, on=["VP", "Phase", "Object"], suffixes=('_like', '_fear'))
    df_reliability_ratings = df_reliability_ratings.merge(df_anger, on=["VP", "Phase", "Object"])
    df_reliability_ratings = df_reliability_ratings.rename(columns={"Value": "Value_anger"})
    x = df_reliability_ratings["Value_like"].to_numpy()
    y = df_reliability_ratings["Value_fear"].to_numpy()
    linreg = linregress(x, y)
    print(f"Likeability x Fear: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

    x = df_reliability_ratings["Value_like"].to_numpy()
    y = df_reliability_ratings["Value_anger"].to_numpy()
    linreg = linregress(x, y)
    print(f"Likeability x Anger: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

    x = df_reliability_ratings["Value_fear"].to_numpy()
    y = df_reliability_ratings["Value_anger"].to_numpy()
    linreg = linregress(x, y)
    print(f"Fear x Anger: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")


# Ratings Virtual Humans, Relationship with Social Anxiety
def plot_rating_agents_scale(df, save_path, scale="SPAI"):
    # df = df_ratings
    wave = 1 if "Wave1" in save_path else 2

    figure_heigth = 5.25 if wave == 2 else 5
    legend = True if wave == 2 else False

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, figure_heigth))
    labels = ["Likeability", "Fear", "Anger", "Attractiveness", "Behavior"]
    conditions = ['Unknown', 'Neutral', 'Friendly', 'Unfriendly']
    colors = ['lightgrey', '#1F82C0', '#B1C800', '#E2001A']

    anovas = pd.DataFrame()

    for idx_label, label in enumerate(labels):
        # idx_label = 0
        # label = labels[idx_label]
        print(f"\n{label}")
        df_crit = df.loc[df["Criterion"] == label]
        df_crit = df_crit.sort_values(by=scale)
        # df_crit = df_crit.loc[~(df_crit["Condition"] == "unknown")]

        df_lmm = df_crit.copy()
        df_lmm = df_lmm.loc[df_lmm["Condition"].isin(["friendly", "unfriendly"])]
        df_lmm[scale] = (df_lmm[scale] - df_lmm[scale].mean()) / df_lmm[scale].std()
        df_lmm = df_lmm.rename(columns={scale: "scale"})
        formula = (f"Value ~  Condition + scale + Condition:scale + (1 | VP)")
        # formula = (f"Value ~  {SA_score} + Condition + click_count "
        #            f"+ Condition:{SA_score} + Condition:click_count + click_count:{SA_score} + Condition:{SA_score}:click_count + (1 | VP)")
        model = pymer4.models.Lmer(formula, data=df_lmm)
        model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        anova['p_eta_2'] = anova.apply(lambda x: utils.partial_eta_squared(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)
        anova['p_eta_2_CI'] = anova.apply(lambda x: utils.partial_eta_squared_ci(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)
        # estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

        print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
        print(f"{scale} Main Effect, F({round(anova.loc['scale', 'NumDF'].item(), 1)}, {round(anova.loc['scale', 'DenomDF'].item(), 1)})={round(anova.loc['scale', 'F-stat'].item(), 2)}, p={round(anova.loc['scale', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['scale', 'p_eta_2'].item(), 2)}")
        print(f"Interaction, F({round(anova.loc[f'Condition:scale', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:scale', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:scale', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:scale', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:scale', 'p_eta_2'].item(), 2)}")

        anova['NumDF'] = anova['NumDF'].round().astype("str")
        anova['DenomDF'] = anova['DenomDF'].round().astype("str")
        anova["df"] = anova['NumDF'].str.cat(anova['DenomDF'], sep=', ')
        anova['F-stat'] = anova['F-stat'].round(2).astype("str")
        anova['P-val'] = anova['P-val'].round(3).astype("str")
        anova.loc[anova['P-val'] == "0.0", "P-val"] = "< .001"
        anova['P-val'] = anova['P-val'].replace({"0.": "."})
        anova['p_eta_2'] = anova['p_eta_2'].round(2).astype("str")

        anova["label"] = label
        anova = anova.reset_index(names=['factor'])
        anova = anova[["label", "factor", "F-stat", "df", "P-val", "p_eta_2", "p_eta_2_CI"]].reset_index()
        anova = anova.drop(columns="index")
        anovas = pd.concat([anovas, anova])

        if idx_label > 2:
            continue

        for idx_condition, condition in enumerate(conditions):
            # idx_condition = 0
            # condition = conditions[idx_condition]
            df_spai = df_crit.groupby(["VP"])[scale].mean(numeric_only=True).reset_index()

            df_cond = df_crit.loc[df_crit['Condition'] == condition.lower()].reset_index(drop=True)
            df_cond = df_cond.dropna(subset="Value")
            df_cond = df_cond.groupby(["VP"]).mean(numeric_only=True).reset_index()
            df_cond = df_cond.sort_values(by=scale)

            x = df_cond[scale].to_numpy()
            y = df_cond["Value"].to_numpy()
            linreg = linregress(x, y)
            all_x = df_crit[scale].to_numpy()
            all_y = df_crit["Value"].to_numpy()
            all_y_est = linreg.slope * all_x + linreg.intercept
            all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
                1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

            # Plot regression line
            axes[idx_label].plot(all_x, all_y_est, '-', color=colors[idx_condition])
            axes[idx_label].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

            p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
            if idx_condition == 2:
                axes[idx_label].text(df_crit[scale].min() + 0.01 * np.max(x), 0.95 * df_crit["Value"].max(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])
            elif idx_condition == 3:
                axes[idx_label].text(df_crit[scale].min() + 0.01 * np.max(x), 0.91 * df_crit["Value"].max(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])
            elif idx_condition == 1:
                axes[idx_label].text(df_crit[scale].min() + 0.01 * np.max(x), 0.87 * df_crit["Value"].max(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])
            elif idx_condition == 0:
                axes[idx_label].text(df_crit[scale].min() + 0.01 * np.max(x), 0.83 * df_crit["Value"].max(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])

            # Plot raw data points
            axes[idx_label].plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3,
                    label=conditions[idx_condition])

        # Style Plot
        axes[idx_label].set_ylim([-2, df_crit["Value"].max()+2])
        axes[idx_label].set_title(f"{label}", fontweight='bold', fontsize="xx-large")  # (N = {len(df_cond['VP'].unique())})
        axes[idx_label].set_ylabel(label, fontsize="x-large")
        if "SPAI" in scale:
            axes[idx_label].set_xticks(range(0, 6))
        elif "SIAS" in scale:
            axes[idx_label].set_xticks(range(5, 65, 5))
        elif "IPQ" in scale:
            axes[idx_label].set_xticks(range(0, 7))
        elif "MPS" in scale:
            axes[idx_label].set_xticks(range(1, 6))
        axes[idx_label].set_xlabel(scale, fontsize="x-large")
        axes[idx_label].grid(color='lightgrey', linestyle='-', linewidth=0.3)


    # axes[2].legend(loc="upper right")
    # axes[2].legend(
    #     [Line2D([0], [0], color="white", marker='o', markeredgecolor='#B1C800', markeredgewidth=1, markerfacecolor='#B1C800', alpha=.7),
    #      Line2D([0], [0], color="white", marker='o', markeredgecolor='#E2001A', markeredgewidth=1, markerfacecolor='#E2001A', alpha=.7),
    #      Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
    #      Line2D([0], [0], color="white", marker='o', markeredgecolor='lightgrey', markeredgewidth=1, markerfacecolor='lightgrey', alpha=.7)],
    #     ["Friendly", "Unfriendly", "Neutral", "Unknown"], loc="upper right")
    plt.tight_layout()
    if legend:
        fig.legend(
            [(Line2D([0], [0], color='#B1C800', linewidth=2, alpha=1), Line2D([0], [0], color='#B1C800', linewidth=15, alpha=0.2)),
             (Line2D([0], [0], color='#E2001A', linewidth=2, alpha=1), Line2D([0], [0], color='#E2001A', linewidth=15, alpha=0.2)),
             (Line2D([0], [0], color='#1F82C0', linewidth=2, alpha=1), Line2D([0], [0], color='#1F82C0', linewidth=15, alpha=0.2)),
             (Line2D([0], [0], color='lightgrey', linewidth=2, alpha=1), Line2D([0], [0], color='lightgrey', linewidth=15, alpha=0.2)),],
            ["Friendly Agent", "Unfriendly Agent", "Neutral Agent", "Unknown Agent"],
            loc='lower center', ncols=4, frameon=False, fontsize="xx-large")
        fig.subplots_adjust(bottom=0.22)

    anovas.to_csv(os.path.join(save_path, f'lmms_ratings_{scale}.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')


# Ratings Virtual Humans, Relationship with Social Anxiety and Clicks
def plot_rating_agents_sad_clicks(df, df_events, SA_score="SPAI"):
    df_clicks = df_events.loc[df_events["event"].str.contains("Clicked")]
    df_clicks = df_clicks.groupby(["VP", "Condition"])["event"].count().reset_index()
    df_clicks = df_clicks.rename(columns={"event": "click_count"})

    df = df.merge(df_clicks, on=["VP", "Condition"], how="outer")
    df["click_count"] = df["click_count"].fillna(0)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    labels = ["Likeability", "Fear", "Anger"]  # , "Attractiveness", "Behavior"]
    conditions = ['Friendly', 'Unfriendly']
    colors = [['#9baf00', '#6f7d00', '#424b00'], ['#e2001a', '#a90014', '#55000a']]
    legend_labels = []
    for idx_label, label in enumerate(labels):
        # idx_label = 2
        # label = labels[idx_label]
        print(label)
        df_crit = df.loc[df["Criterion"] == label]
        df_crit = df_crit.sort_values(by=SA_score)

        df_ancova = df_crit.copy()
        df_ancova = df_ancova.loc[df_ancova["Condition"].isin(["friendly", "unfriendly"])]
        df_ancova[SA_score] = (df_ancova[SA_score] - df_ancova[SA_score].mean()) / df_ancova[SA_score].std()
        formula = (f"Value ~  {SA_score} + Condition + click_count + "
                   f"Condition:{SA_score} + Condition:click_count + click_count:{SA_score} + "
                   f"Condition:{SA_score}:click_count + (1 | VP)")
        model = pymer4.models.Lmer(formula, data=df_ancova)
        model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        anova['p_eta_2'] = anova.apply(lambda x: utils.partial_eta_squared(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)
        anova['p_eta_2_CI'] = anova.apply(lambda x: utils.partial_eta_squared_ci(x['F-stat'], x['NumDF'], x['DenomDF']), axis=1)

        estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

        for idx_condition, condition in enumerate(conditions):
            # idx_condition = 1
            # condition = conditions[idx_condition]
            df_spai = df_crit.groupby(["VP"])[SA_score].mean(numeric_only=True).reset_index()
            df_spai = df_spai.sort_values(by=SA_score)

            df_cond = df_crit.loc[df_crit['Condition'] == condition.lower()].reset_index(drop=True)
            df_cond = df_cond.dropna(subset="Value")
            df_cond = df_cond.groupby(["VP"]).mean(numeric_only=True).reset_index()
            df_cond = df_cond.sort_values(by=SA_score)

            for click_count in [0, 1, 2]:
                # click_count = 2
                if click_count == 0:
                    df_click = df_cond.loc[df_cond["click_count"] == click_count]
                    click_label = "0 Clicks"
                elif click_count == 1:
                    df_click = df_cond.loc[df_cond["click_count"] == click_count]
                    click_label = "1 Click"
                elif click_count == 2:
                    df_click = df_cond.loc[df_cond["click_count"] >= 1]
                    click_label = ">1 Clicks"

                x = df_click[SA_score].to_numpy()
                y = df_click["Value"].to_numpy()
                linreg = linregress(x, y)
                all_x = df_crit[SA_score].to_numpy()
                all_y = df_crit["Value"].to_numpy()
                all_y_est = linreg.slope * all_x + linreg.intercept
                all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
                    1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

                # Plot regression line
                axes[idx_label].plot(all_x, all_y_est, '-', color=colors[idx_condition][click_count])
                axes[idx_label].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.4, color=colors[idx_condition][click_count])

                # Plot raw data points
                if idx_label == 0:
                    axes[idx_label].plot(x, y, 'o', ms=5, mfc=colors[idx_condition][click_count], mec=colors[idx_condition][click_count], alpha=0.4,
                                         label=f"{conditions[idx_condition]}, {click_label}")
                    legend_labels.append(axes[idx_label].get_legend_handles_labels()[1])
                else:
                    axes[idx_label].plot(x, y, 'o', ms=5, mfc=colors[idx_condition][click_count],
                                        mec=colors[idx_condition][click_count], alpha=0.4, label=None)

        # Style Plot
        axes[idx_label].set_ylim([-2, df_crit["Value"].max()+2])
        axes[idx_label].set_title(f"{label}", fontweight='bold')  # (N = {len(df_cond['VP'].unique())})
        axes[idx_label].set_ylabel(label)
        if "SPAI" in SA_score:
            axes[idx_label].set_xticks(range(0, 6))
        elif "SIAS" in SA_score:
            axes[idx_label].set_xticks(range(5, 65, 5))
        axes[idx_label].set_xlabel(SA_score)
        axes[idx_label].grid(color='lightgrey', linestyle='-', linewidth=0.3)

    fig.legend(
        [Line2D([0], [0], color=colors[0][0], marker='o', markeredgecolor=colors[0][0], markeredgewidth=1, markerfacecolor=colors[0][0], alpha=.7),
         Line2D([0], [0], color=colors[0][1], marker='o', markeredgecolor=colors[0][1], markeredgewidth=1, markerfacecolor=colors[0][1], alpha=.7),
         Line2D([0], [0], color=colors[0][2], marker='o', markeredgecolor=colors[0][2], markeredgewidth=1, markerfacecolor=colors[0][2], alpha=.7),
         Line2D([0], [0], color=colors[1][0], marker='o', markeredgecolor=colors[1][0], markeredgewidth=1, markerfacecolor=colors[1][0], alpha=.7),
         Line2D([0], [0], color=colors[1][1], marker='o', markeredgecolor=colors[1][1], markeredgewidth=1, markerfacecolor=colors[1][1], alpha=.7),
         Line2D([0], [0], color=colors[1][2], marker='o', markeredgecolor=colors[1][2], markeredgewidth=1, markerfacecolor=colors[1][2], alpha=.7)],
        legend_labels[-1], loc="center right")
    plt.tight_layout()
    fig.subplots_adjust(right=0.84)


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
    file_name_labbook = [item for item in os.listdir(filepath) if (item.endswith(".xlsx") and "Labbook" in item)][0]
    df_labbook = pd.read_excel(os.path.join(filepath, file_name_labbook), sheet_name=f"Wave{wave}")
    df_scores, problematic_subjects = preproc_scores.create_scores(df_scores_raw, df_labbook, problematic_subjects)

    start = 1
    vp_folder = [int(item.split("_")[1]) for item in os.listdir(filepath) if ("VP" in item)]
    end = np.max(vp_folder)
    vps = np.arange(start, end + 1)
    vps = [vp for vp in vps if not vp in problematic_subjects]

    df_ratings, problematic_subjects = preproc_ratings.create_ratings(vps, filepath, problematic_subjects, df_scores)

    # df_ratings = df_ratings.loc[~(df_ratings["VP"].isin(problematic_subjects))]
    # df_ratings_fear = df_ratings.loc[df_ratings["Criterion"] == "Fear"]
    # df_ratings_fear = df_ratings_fear[["VP", "Condition", "Value", "SPAI"]]
    # df_ratings_fear.columns = ["VP", "Condition", "Rating", "SPAI"]
    # df_ratings_fear.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'fear_ratings.csv'), decimal='.', sep=';', index=False)
    #
    # df_fear_wide = df_ratings_fear.pivot(index=['VP', 'SPAI'], columns="Condition", values='Rating').reset_index()
    # df_fear_wide.columns = ["VP", "SPAI", "Condition_friendly", "Condition_neutral", "Condition_unfriendly", "Condition_unknown"]
    # df_fear_wide.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'fear_ratings_wide.csv'), decimal='.', sep=';', index=False)
