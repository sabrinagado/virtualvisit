# =============================================================================
# Physiology
# sensor: HMD & Unreal Engine (Log Writer), movisens
# study: Virtual Visit
# =============================================================================
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress, t, ttest_rel, percentileofscore
from scipy import signal
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
import pymer4
from tqdm import tqdm
from Code.toolbox import utils


dir_path = os.getcwd()
save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Physiology')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

red = '#E2001A'
green = '#B1C800'
blue = '#1F82C0'
SA_score = "SPAI"

# Acquisition
ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]"]
colors = [green, red, blue]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(["hr", "eda", "pupil"], ["ECG", "EDA", "pupil"], ylabels)):
    # physio_idx = 0
    # physiology = "hr"
    # column_name = "ECG"
    # ylabel = "Heart Rate [BPM]"
    df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', f'{physiology}_interaction.csv'), decimal='.', sep=';')
    if physiology == "hr":
        df = df.loc[(df[column_name] >= df[column_name].mean() - 3 * df[column_name].std()) & (df[column_name] <= df[column_name].mean() + 3 * df[column_name].std())]  # exclude outliers
    elif physiology == "eda":
        for vp in df["VP"].unique():
            # vp = 2
            df_vp = df.loc[df["VP"] == vp]
            for event in df_vp["event"].unique():
                # event = "FriendlyInteraction"
                df_event = df_vp.loc[df_vp["event"] == event]
                eda_signal = df_event["EDA"].to_numpy()

                # Get local minima and maxima
                local_maxima = signal.argrelextrema(eda_signal, np.greater)[0]
                peak_values = list(eda_signal[local_maxima])
                peak_times = list(local_maxima)

                local_minima = signal.argrelextrema(eda_signal, np.less)[0]
                onset_values = list(eda_signal[local_minima])
                onset_times = list(local_minima)

                scr_onsets = []
                scr_peaks = []
                scr_amplitudes = []
                scr_risetimes = []
                amplitude_min = 0.02
                for onset_idx, onset in enumerate(onset_times):
                    # onset_idx = 0
                    # onset = onset_times[onset_idx]
                    subsequent_peak_times = [peak_time for peak_time in peak_times if (onset - peak_time) < 0]
                    if len(subsequent_peak_times) > 0:
                        peak_idx = list(peak_times).index(min(subsequent_peak_times, key=lambda x: abs(x - onset)))
                        rise_time = (peak_times[peak_idx] - onset) / 10
                        amplitude = peak_values[peak_idx] - onset_values[onset_idx]
                        if (rise_time > 0.1) & (rise_time < 10) & (amplitude >= amplitude_min):
                            scr_onsets.append(onset)
                            scr_peaks.append(peak_times[peak_idx])
                            scr_amplitudes.append(amplitude)
                            scr_risetimes.append(rise_time)
                if len(scr_amplitudes) == 0:
                    df = df.loc[~((df["VP"] == vp) & (df["event"] == event))]

    phases = ["FriendlyInteraction", "UnfriendlyInteraction"]  # "NeutralInteraction",
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    titles = ["Friendly Interaction", "Unfriendly Interaction"]  # "Neutral Interaction",
    for idx_phase, phase in enumerate(phases):
        # idx_phase = 0
        # phase = phases[idx_phase]
        df_phase = df.loc[df['event'] == phase]

        times = df_phase["time"].unique()
        mean = df_phase.groupby("time")[column_name].mean()
        sem = df_phase.groupby("time")[column_name].sem()

        # Plot line
        axes[physio_idx].plot(times, mean, '-', color=colors[idx_phase], label=titles[idx_phase])
        axes[physio_idx].fill_between(times, mean + sem, mean - sem, alpha=0.2, color=colors[idx_phase])

    y_pos = axes[physio_idx].get_ylim()[0] + 0.02 * (axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])

    df_diff = df.loc[df["event"].isin(phases)]
    df_diff = df_diff.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
    t_vals = pd.DataFrame()
    t_vals["t"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["FriendlyInteraction"], df_diff["UnfriendlyInteraction"], nan_policy="omit").statistic)
    t_vals["df"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["FriendlyInteraction"], df_diff["UnfriendlyInteraction"], nan_policy="omit").df)
    t_vals["threshold"] = t_vals.apply(lambda x: t.ppf(1 - 0.025, x["df"]), axis=1)  # two-sided test
    t_vals["t"] = t_vals["t"].abs()
    t_vals["sig"] = t_vals["t"] > t_vals["threshold"]
    t_vals = t_vals.reset_index()

    t_vals["idx_cluster"] = 0
    idx_cluster = 1
    for idx_row, row in t_vals.iterrows():
        # idx_row = 0
        # row = t_vals.iloc[idx_row, ]
        if row["sig"]:
            t_vals.loc[idx_row, "idx_cluster"] = idx_cluster
            if (idx_row < len(t_vals) - 1) and not (t_vals.loc[idx_row + 1, "sig"]):
                idx_cluster += 1

    cluster_mass = t_vals.groupby("idx_cluster")["t"].sum().reset_index()
    cluster_mass = cluster_mass.loc[cluster_mass["idx_cluster"] != 0].reset_index()

    pd.options.mode.chained_assignment = None
    cluster_distribution = []
    for i in tqdm(np.arange(0, 1000)):
        df_shuffle = pd.DataFrame()
        df_subset = df.loc[df["event"].isin(phases)]
        for vp in df_subset["VP"].unique():
            # vp = df_subset["VP"].unique()[0]
            df_vp = df_subset.loc[df_subset["VP"] == vp]
            rand_int = np.random.randint(0, 2)
            if rand_int == 0:
                df_vp["event"] = df_vp["event"].replace({"FriendlyInteraction": "UnfriendlyInteraction",
                                                         "UnfriendlyInteraction": "FriendlyInteraction"})
            df_shuffle = pd.concat([df_shuffle, df_vp])

        df_shuffle = df_shuffle.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
        t_vals_shuffle = pd.DataFrame()
        t_vals_shuffle["t"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle["FriendlyInteraction"], df_shuffle["UnfriendlyInteraction"], nan_policy="omit").statistic)
        t_vals_shuffle["df"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle["FriendlyInteraction"], df_shuffle["UnfriendlyInteraction"], nan_policy="omit").df)
        t_vals_shuffle["threshold"] = t_vals_shuffle.apply(lambda x: t.ppf(1 - 0.025, x["df"]), axis=1)  # two-sided test
        t_vals_shuffle["t"] = t_vals_shuffle["t"].abs()
        t_vals_shuffle["sig"] = t_vals_shuffle["t"] > t_vals_shuffle["threshold"]
        t_vals_shuffle = t_vals_shuffle.reset_index()

        t_vals_shuffle["idx_cluster"] = 0
        idx_cluster = 1
        for idx_row, row in t_vals_shuffle.iterrows():
            # idx_row = 0
            # row = t_vals_shuffle.iloc[idx_row, ]
            if row["sig"]:
                t_vals_shuffle.loc[idx_row, "idx_cluster"] = idx_cluster
                if (idx_row < len(t_vals_shuffle) - 1) and not (t_vals_shuffle.loc[idx_row + 1, "sig"]):
                    idx_cluster += 1

        cluster_mass_shuffle = t_vals_shuffle.groupby("idx_cluster")["t"].sum().reset_index()
        cluster_mass_shuffle = cluster_mass_shuffle.loc[cluster_mass_shuffle["idx_cluster"] != 0].reset_index()
        if len(cluster_mass_shuffle) > 0:
            cluster_distribution.append(cluster_mass_shuffle["t"].max())

    cluster_mass["p-val"] = cluster_mass.apply(lambda x: 1 - percentileofscore(cluster_distribution, x["t"]) / 100, axis=1)
    t_vals = t_vals.merge(cluster_mass[["idx_cluster", "p-val"]], on='idx_cluster', how="left")

    for timepoint in t_vals["time"].unique():
        # timepoint = 0
        p = t_vals.loc[(t_vals["time"] == timepoint), "p-val"].item()
        if p < 0.05:
            axes[physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint+0.1, linewidth=5, color='gold')

    # Style Plot
    axes[physio_idx].set_xlim([0, 5])
    axes[physio_idx].set_ylabel(ylabel)
    axes[physio_idx].set_title(f"{ylabel.split(' [')[0].replace(' (BPM)', '')} (N = {len(df['VP'].unique())})", fontweight='bold')
    axes[physio_idx].set_xlabel("Seconds after Interaction Onset")
    axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

axes[2].legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"physiology_acq.png"), dpi=300)
plt.close()


# Clicks
ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]"]
colors = [green, red]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(["hr", "eda", "pupil"], ["ECG", "EDA", "pupil"], ylabels)):
    # physiology = "hr"
    # column_name = "ECG"
    # ylabel = "Heart Rate (BPM)"
    df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', f'{physiology}_interaction.csv'), decimal='.', sep=';')
    if physiology == "hr":
        df = df.loc[(df[column_name] >= df[column_name].mean() - 3 * df[column_name].std()) & (df[column_name] <= df[column_name].mean() + 3 * df[column_name].std())]  # exclude outliers
    elif physiology == "eda":
        for vp in df["VP"].unique():
            # vp = 2
            df_vp = df.loc[df["VP"] == vp]
            for event in df_vp["event"].unique():
                # event = "FriendlyInteraction"
                df_event = df_vp.loc[df_vp["event"] == event]
                eda_signal = df_event["EDA"].to_numpy()
                # Get local minima and maxima
                local_maxima = signal.argrelextrema(eda_signal, np.greater)[0]
                peak_values = list(eda_signal[local_maxima])
                peak_times = list(local_maxima)

                local_minima = signal.argrelextrema(eda_signal, np.less)[0]
                onset_values = list(eda_signal[local_minima])
                onset_times = list(local_minima)

                scr_onsets = []
                scr_peaks = []
                scr_amplitudes = []
                scr_risetimes = []
                amplitude_min = 0.02
                for onset_idx, onset in enumerate(onset_times):
                    # onset_idx = 0
                    # onset = onset_times[onset_idx]
                    subsequent_peak_times = [peak_time for peak_time in peak_times if (onset - peak_time) < 0]
                    if len(subsequent_peak_times) > 0:
                        peak_idx = list(peak_times).index(min(subsequent_peak_times, key=lambda x: abs(x - onset)))
                        rise_time = (peak_times[peak_idx] - onset) / 10
                        amplitude = peak_values[peak_idx] - onset_values[onset_idx]
                        if (rise_time > 0.1) & (rise_time < 10) & (amplitude >= amplitude_min):
                            scr_onsets.append(onset)
                            scr_peaks.append(peak_times[peak_idx])
                            scr_amplitudes.append(amplitude)
                            scr_risetimes.append(rise_time)
                if len(scr_amplitudes) == 0:
                    df = df.loc[~((df["VP"] == vp) & (df["event"] == event))]

    df = df.loc[df["time"] < 3]

    phases = ["Test_FriendlyWasClicked", "Test_UnfriendlyWasClicked"]
    titles = ["Friendly Clicked", "Unfriendly Clicked"]
    for idx_phase, phase in enumerate(phases):
        # idx_phase = 0
        # phase = phases[idx_phase]
        df_phase = df.loc[df['event'] == phase]

        times = df_phase["time"].unique()
        mean = df_phase.groupby("time")[column_name].mean()
        sem = df_phase.groupby("time")[column_name].sem()

        # Plot line
        axes[physio_idx].plot(times, mean, '-', color=colors[idx_phase], label=titles[idx_phase])
        axes[physio_idx].fill_between(times, mean + sem, mean - sem, alpha=0.2, color=colors[idx_phase])

    y_pos = axes[physio_idx].get_ylim()[0] + 0.02 * (axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])

    for timepoint in df["time"].unique():
        # timepoint = 0
        df_tp = df.loc[(df["time"] == timepoint)]
        df_tp = df_tp.loc[df_tp["event"].isin(phases)]
        formula = f"{column_name} ~ event + (1 | VP)"

        model = pymer4.models.Lmer(formula, data=df_tp)
        model.fit(factors={"event": ["Test_FriendlyWasClicked", "Test_UnfriendlyWasClicked"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)

        p = anova.loc["event", "P-val"].item()
        if p < 0.05:
            axes[physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint+0.1, linewidth=5, color='gold')

    # Style Plot
    axes[physio_idx].set_xlim([0, 2.9])
    axes[physio_idx].set_ylabel(ylabel)
    axes[physio_idx].set_title(f"{ylabel.split(' [')[0]} (N = {len(df['VP'].unique())})", fontweight='bold')
    axes[physio_idx].set_xlabel("Seconds after Click")
    axes[physio_idx].legend(loc="upper right")
    axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"physiology_click.png"), dpi=300)
plt.close()


# Test Phase
red = '#E2001A'
green = '#B1C800'
colors = [green, red]
ylabels = ["Pupil Diameter [mm]", "Skin Conductance Level [µS]", "Heart Rate [BPM]", "Heart Rate Variability\n(High Frequency)", "Heart Rate Variability (RMSSD)"]
dvs = ["Pupil Dilation (Mean)", "SCL (Mean)", "HR (Mean)", "HRV (HF_nu)", "HRV (RMSSD)"]
for physiology, ylabel, dv in zip(["pupil", "eda", "hr", "hrv_hf", "hrv_rmssd"], ylabels, dvs):
    # physiology = "hrv_hf"
    # ylabel, dv = ylabels[3], dvs[3]
    if "hrv" in physiology:
        df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', f'hr.csv'), decimal='.', sep=';')
    else:
        df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', f'{physiology}.csv'), decimal='.', sep=';')

    df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
    df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
    df_subset.loc[df_subset['Phase'].str.contains("Habituation"), "phase"] = "Habituation"
    df_subset.loc[df_subset['Phase'].str.contains("Office"), "room"] = "Office"
    df_subset.loc[df_subset['Phase'].str.contains("Living"), "room"] = "Living"
    df_subset.loc[df_subset['Phase'].str.contains("Dining"), "room"] = "Dining"

    df_subset = df_subset.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
    df_subset = df_subset.dropna(subset=dv)

    conditions = ["friendly", "unfriendly"]
    phases = ['Habituation', 'Test']
    titles = ["Room with Friendly Person", "Room with Unfriendly Person"]
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
            bootstrapping_dict = utils.bootstrapping(df_phase.loc[:, dv].values,
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

        model = pymer4.models.Lmer(formula, data=df_cond)
        model.fit(factors={"phase": ['Habituation', 'Test']}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        estimates, contrasts = model.post_hoc(marginal_vars="phase", p_adjust="holm")

        p = anova.loc["phase", "P-val"].item()
        max = df_subset[dv].max()
        if p < 0.05:
            ax.hlines(y=max * 1.05, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
            ax.vlines(x=pos[0], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
            ax.vlines(x=pos[1], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
            p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

    df_crit = df_subset.copy()
    df_crit = df_crit.loc[df_crit["Condition"].isin(conditions)]
    df_crit = df_crit.rename(columns={dv: physiology})
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

    formula = f"{physiology} ~ phase + Condition + {SA_score} + " \
              f"phase:Condition + phase:{SA_score} + Condition:{SA_score} +" \
              f"phase:Condition:{SA_score} + (1 | VP)"

    model = pymer4.models.Lmer(formula, data=df_crit)
    model.fit(factors={"phase": ["Habituation", "Test"], "Condition": ["friendly", "unfriendly"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="phase", p_adjust="holm")

    p = contrasts.loc[contrasts["phase"] == "Test", "P-val"].item()
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
    ax.set_title(ylabel.split("[")[0], fontweight='bold')

    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1, markerfacecolor='#1F82C0', alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.7)
    plt.savefig(os.path.join(save_path, f"{physiology}_hab-test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Correlation with SPAI
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1
    pos = [1]

    titles = ["Room with Friendly Person", "Room with Unfriendly Person"]
    df_test = df_subset.loc[df_subset['phase'] == "Test"]
    df_test = df_test.sort_values(by=SA_score)
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)

        x = df_cond[SA_score].to_numpy()
        y = df_cond[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_test[SA_score].to_numpy()
        all_y = df_cond[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.95 * (df_test[dv].max()-df_test[dv].min()) + df_test[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])
        else:
            ax.text(df_test[SA_score].min() + 0.01 * np.max(x), 0.91 * (df_test[dv].max()-df_test[dv].min()) + df_test[dv].min(),
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_condition])

        # Plot raw data points
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"{ylabel} in Test Phase")
    ax.set_title(ylabel.split("[")[0], fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{physiology}_test_{SA_score}.png"), dpi=300)
    plt.close()

    # Correlation with SPAI (Test-Habituation)
    df_spai = df[["VP", SA_score]].drop_duplicates(subset="VP")
    df_spai = df_spai.sort_values(by=SA_score)
    df_diff = df_subset.pivot(index='VP', columns=['phase', "Condition"], values=dv).reset_index()
    df_diff = df_diff.dropna()
    df_diff["friendly"] = df_diff[("Test"), ("friendly")] - df_diff[("Habituation"), ("friendly")]
    df_diff["unfriendly"] = df_diff[("Test"), ("unfriendly")] - df_diff[("Habituation"), ("unfriendly")]
    df_diff.columns = df_diff.columns.droplevel(level=1)
    df_diff = df_diff[["VP", "friendly", "unfriendly"]]
    df_diff = df_diff.merge(df_spai, on="VP")
    df_diff = pd.melt(df_diff, id_vars=['VP', 'SPAI'], value_vars=['friendly', 'unfriendly'], var_name="Condition", value_name="difference")
    df_diff = df_diff.sort_values(by=SA_score)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
    boxWidth = 1
    pos = [1]
    conditions = ["friendly", "unfriendly"]
    titles = ["Room with Friendly Person", "Room with Unfriendly Person"]

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df_diff.loc[df_diff['Condition'] == condition].reset_index(drop=True)
        df_cond = df_cond.dropna(subset="difference")
        df_cond = df_cond.sort_values(by=SA_score)

        x = df_cond[SA_score].to_numpy()
        y = df_cond["difference"].to_numpy()
        linreg = linregress(x, y)
        all_x = df_spai[SA_score].to_numpy()
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
        ax.plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"Difference (Test - Habituation) in {ylabel}")
    ax.set_title(ylabel.split("[")[0], fontweight='bold')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{physiology}_diff_{SA_score}.png"), dpi=300)
    plt.close()


for physiology in ["pupil", "eda", "hr"]:
    # physiology = "hr"
    if physiology == "pupil":
        dv = "Pupil Dilation (Mean)"
    elif physiology == "eda":
        dv = "SCL (Mean)"
    elif physiology == "hr":
        dv = "HR (Mean)"

    df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', f'{physiology}.csv'), decimal='.', sep=';')

    df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
    df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
    df_subset.loc[df_subset['Phase'].str.contains("Habituation"), "phase"] = "Habituation"
    df_subset.loc[df_subset['Phase'].str.contains("Office"), "room"] = "Office"
    df_subset.loc[df_subset['Phase'].str.contains("Living"), "room"] = "Living"
    df_subset.loc[df_subset['Phase'].str.contains("Dining"), "room"] = "Dining"
    df_subset = df_subset.rename(columns={dv: physiology})
    df_subset[SA_score] = (df_subset[SA_score] - df_subset[SA_score].mean()) / df_subset[SA_score].std()
    df_subset = df_subset.loc[df_subset["Condition"].isin(conditions)]

    formula = f"{physiology} ~ Phase + Condition + {SA_score} + " \
              f"Phase:Condition + Phase:{SA_score} + Condition:{SA_score} +" \
              f"Phase:Condition:{SA_score} +" \
              f"(1 | VP)"

    model = pymer4.models.Lmer(formula, data=df_subset)
    model.fit(factors={"Condition": ['friendly', 'unfriendly'],
                       "Phase": ['Habituation_DiningRoom', 'Test_DiningRoom', 'Habituation_LivingRoom', 'Test_LivingRoom']}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="Phase", p_adjust="holm")
