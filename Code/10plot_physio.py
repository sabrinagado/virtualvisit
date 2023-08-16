# =============================================================================
# Physiology
# sensor: HMD & Unreal Engine (Log Writer), movisens
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
save_path = os.path.join(dir_path, 'Plots', 'Physiology')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

red = '#E2001A'
green = '#B1C800'
blue = '#1F82C0'
colors = [green, blue, red]

# Acquisition
ylabels = ["Pupil Diameter [mm]", "Skin Conductance Level [µS]", "Heart Rate (BPM)"]
for physiology, column_name, ylabel in zip(["pupil", "eda", "hr"], ["pupil", "EDA", "ECG"], ylabels):
    # physiology = "pupil"
    # column_name = "pupil"
    # ylabel = "Heart Rate (BPM)"
    df = pd.read_csv(os.path.join(dir_path, 'Data', f'{physiology}_interaction.csv'), decimal='.', sep=';')
    if physiology == "hr":
        df = df.loc[(df[column_name] >= df[column_name].mean() - 3 * df[column_name].std()) & (df[column_name] <= df[column_name].mean() + 3 * df[column_name].std())]  # exclude outliers

    phases = ["FriendlyInteraction", "NeutralInteraction", "UnfriendlyInteraction"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    titles = ["Friendly Interaction", "Neutral Interaction", "Unfriendly Interaction"]
    for idx_phase, phase in enumerate(phases):
        # idx_phase = 0
        # phase = phases[idx_phase]
        df_phase = df.loc[df['event'] == phase]

        times = df_phase["time"].unique()
        mean = df_phase.groupby("time")[column_name].mean()
        sem = df_phase.groupby("time")[column_name].sem()

        # Plot line
        ax.plot(times, mean, '-', color=colors[idx_phase], label=titles[idx_phase])
        ax.fill_between(times, mean + sem, mean - sem, alpha=0.2, color=colors[idx_phase])

    # df_resample = df.copy()
    # df_resample["second"] = df_resample["time"].astype("int")
    # df_resample = df_resample.drop(columns="time")
    # df_resample = df_resample.groupby(["VP", "event", "second"]).mean().reset_index()
    y_pos = ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for timepoint in df["time"].unique():
        # timepoint = 0
        df_tp = df.loc[(df["time"] == timepoint)]

        df_tp = df_tp.loc[~(df_tp["event"].str.contains("Neutral"))]
        formula = f"{column_name} ~ event + (1 | VP)"

        lm = smf.ols(formula, data=df_tp).fit()
        anova = sm.stats.anova_lm(lm, typ=3)
        sum_sq_error = anova.loc["Residual", "sum_sq"]
        anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

        p = anova.loc["event", "PR(>F)"].item()
        if p < 0.05:
            ax.hlines(y=y_pos, xmin=timepoint, xmax=timepoint+0.1, linewidth=5, color='gold')

    # Style Plot
    ax.set_xlim([0, 5])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel.split(' [')[0]} (N = {len(df['VP'].unique())})", fontweight='bold')
    ax.set_xlabel("Seconds after Interaction Onset")
    ax.legend(loc="upper right")
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.legend()

    plt.tight_layout()
    for end in (['.png']):  # '.pdf',
        plt.savefig(os.path.join(save_path, f"{physiology}_acq{end}"), dpi=300)
    plt.close()

# Test Phase
red = '#E2001A'
green = '#B1C800'
colors = [green, red]
ylabels = ["Pupil Diameter [mm]", "Skin Conductance Level [µS]", "Heart Rate (BPM)"]
dvs = ["Pupil Dilation (Mean)", "SCL (Mean)", "HR (Mean)"]
for physiology, ylabel, dv in zip(["pupil", "eda", "hr"], ylabels, dvs):
    # physiology = "pupil"
    # ylabel = "Pupil Diameter [mm]"
    # dv = "Pupil Dilation (Mean)"
    df = pd.read_csv(os.path.join(dir_path, 'Data', f'{physiology}.csv'), decimal='.', sep=';')

    df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
    df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
    df_subset.loc[df_subset['Phase'].str.contains("Habituation"), "phase"] = "Habituation"
    df_subset.loc[df_subset['Phase'].str.contains("Office"), "room"] = "Office"
    df_subset.loc[df_subset['Phase'].str.contains("Living"), "room"] = "Living"
    df_subset.loc[df_subset['Phase'].str.contains("Dining"), "room"] = "Dining"

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

            if phase == "Habituation":
                colors = ['#1F82C0', '#1F82C0']
            else:
                colors = [green, red]

            # Plot raw data points
            for i in range(len(df_phase)):
                # i = 0
                x = random.uniform(pos[idx_phase] - (0.25 * boxWidth), pos[idx_phase] + (0.25 * boxWidth))
                y = df_phase.reset_index().loc[i, dv].item()
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
            bootstrapping_dict = bootstrapping(df_phase.loc[:, dv].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            ax.boxplot([df_phase.loc[:, dv].values],
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

        df_cond = df_cond.rename(columns={dv: physiology})
        formula = f"{physiology} ~ phase + (1 | VP)"

        lm = smf.ols(formula, data=df_cond).fit()
        anova = sm.stats.anova_lm(lm, typ=3)
        sum_sq_error = anova.loc["Residual", "sum_sq"]
        anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

        p = anova.loc["phase", "PR(>F)"].item()
        max = df_subset[dv].max()
        if p < 0.05:
            ax.hlines(y=max * 1.05, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
            ax.vlines(x=pos[0], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
            ax.vlines(x=pos[1], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
            p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

    # df_crit = df_subset.copy()
    # formula = f"{dv} ~ phase + Condition + SPAI + " \
    #           f"phase:Condition + phase:SPAI + Condition:SPAI +" \
    #           f"phase:Condition:SPAI + (1 | VP)"
    #
    # lm = smf.ols(formula, data=df_crit).fit()
    # anova = sm.stats.anova_lm(lm, typ=3)
    # sum_sq_error = anova.loc["Residual", "sum_sq"]
    # anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)
    #
    # max = df_subset[dv].max()
    # p = anova.loc["Condition", "PR(>F)"].item()
    # if p < 0.05:
    #     ax.hlines(y=max * 1.10, xmin=0.51, xmax=1.49, linewidth=0.7, color='k')
    #     ax.vlines(x=0.51, ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
    #     ax.vlines(x=1.49, ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
    #     p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    #     ax.text(1, max * 1.105, p_sign, color='k', horizontalalignment='center')

    df_crit = df_subset.loc[df_subset["phase"].str.contains("Test")]
    df_crit = df_crit.rename(columns={dv: physiology})
    formula = f"{physiology} ~ Condition + (1 | VP)"

    lm = smf.ols(formula, data=df_crit).fit()
    anova = sm.stats.anova_lm(lm, typ=3)
    sum_sq_error = anova.loc["Residual", "sum_sq"]
    anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

    p = anova.loc["Condition", "PR(>F)"].item()
    max = df_subset[dv].max()
    if p < 0.05:
        ax.hlines(y=max * 1.10, xmin=2 * boxWidth, xmax=1 + 2 * boxWidth, linewidth=0.7, color='k')
        ax.vlines(x=2 * boxWidth, ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
        ax.vlines(x=1 + 2 * boxWidth, ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
        p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(np.mean([2 * boxWidth, 1 + 2 * boxWidth]), max * 1.105, p_sign, color='k', horizontalalignment='center')

    ax.set_xticks([x + 1 / 2 for x in range(len(conditions))])
    ax.set_xticklabels([title.replace("with", "with\n") for title in titles])
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(ylabel)

    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1,
                markerfacecolor='#1F82C0', alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green,
                alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red,
                alpha=.7)],
        ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.76)
    for end in (['.png']):  # '.pdf',
        plt.savefig(os.path.join(save_path, f"{physiology}_test{end}"), dpi=300, bbox_inches="tight")
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
        y = df_cond[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_test["SPAI"].to_numpy()
        all_y = df_cond[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_test["SPAI"].min() + 0.01 * np.max(x), 0.95 * (df_test[dv].max()-df_test[dv].min()) + df_test[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        else:
            ax.text(df_test["SPAI"].min() + 0.01 * np.max(x), 0.91 * (df_test[dv].max()-df_test[dv].min()) + df_test[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])

        # Plot raw data points
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6,
                label=titles[idx_condition])

    ax.set_xlabel("SPAI")
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"{ylabel} in Test Phase")
    ax.legend()
    plt.tight_layout()
    for end in (['.png']):  # '.pdf',
        plt.savefig(os.path.join(save_path, f"{physiology}_test_SA{end}"), dpi=300)
    plt.close()

for physiology in ["pupil", "eda", "hr"]:
    # physiology = "hr"
    if physiology == "pupil":
        dv = "Pupil Dilation (Mean)"
    elif physiology == "eda":
        dv = "SCL (Mean)"
    elif physiology == "hr":
        dv = "HR (Mean)"
    df = pd.read_csv(os.path.join(dir_path, 'Data', f'{physiology}.csv'), decimal='.', sep=';')

    df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
    df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
    df_subset.loc[df_subset['Phase'].str.contains("Habituation"), "phase"] = "Habituation"
    df_subset.loc[df_subset['Phase'].str.contains("Office"), "room"] = "Office"
    df_subset.loc[df_subset['Phase'].str.contains("Living"), "room"] = "Living"
    df_subset.loc[df_subset['Phase'].str.contains("Dining"), "room"] = "Dining"
    df_subset = df_subset.rename(columns={dv: physiology})
    df_subset["SPAI"] = (df_subset["SPAI"] - df_subset["SPAI"].mean()) / df_subset["SPAI"].std()

    formula = f"{physiology} ~ Phase + Condition + SPAI + " \
              f"Phase:Condition + Phase:SPAI + Condition:SPAI +" \
              f"Phase:Condition:SPAI +" \
              f"(1 | VP)"

    lm = smf.ols(formula, data=df_subset).fit()
    anova = sm.stats.anova_lm(lm, typ=3)
    sum_sq_error = anova.loc["Residual", "sum_sq"]
    anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

