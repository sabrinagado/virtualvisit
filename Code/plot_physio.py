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


# Acquisition
def plot_physio_acq(filepath, split=False, median=None, SA_score="SPAI", permutations=1000):
    physiologies = ["hr", "eda", "pupil", "hrv_hf"]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]"]
    dvs = ["ECG", "EDA", "pupil"]
    red = '#E2001A'
    green = '#B1C800'
    blue = '#1F82C0'
    colors = [green, red, blue]
    if split:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
        for idx_group, SA_group in enumerate(["HSA", "LSA"]):
            # idx_group = 0
            # SA_group = "HSA"
            for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(physiologies, dvs, ylabels)):
                # physio_idx = 2
                # physiology, column_name, ylabel = physiologies[physio_idx], dvs[physio_idx], ylabels[physio_idx]
                df = pd.read_csv(os.path.join(filepath, f'{physiology}_interaction.csv'), decimal='.', sep=';')
                if SA_group == "LSA":
                    df_group = df.loc[df[SA_score] < median]
                else:
                    df_group = df.loc[df[SA_score] >= median]
                print(f"{physiology.upper()}: N - {SA_group} = {len(df_group['VP'].unique())}")
                if physiology == "eda":
                    for vp in df_group["VP"].unique():
                        # vp = 2
                        df_vp = df_group.loc[df_group["VP"] == vp]
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
                                    peak_idx = list(peak_times).index(
                                        min(subsequent_peak_times, key=lambda x: abs(x - onset)))
                                    rise_time = (peak_times[peak_idx] - onset) / 10
                                    amplitude = peak_values[peak_idx] - onset_values[onset_idx]
                                    if (rise_time > 0.1) & (rise_time < 10) & (amplitude >= amplitude_min):
                                        scr_onsets.append(onset)
                                        scr_peaks.append(peak_times[peak_idx])
                                        scr_amplitudes.append(amplitude)
                                        scr_risetimes.append(rise_time)
                            if len(scr_amplitudes) == 0:
                                df_group = df_group.loc[~((df_group["VP"] == vp) & (df_group["event"] == event))]

                phases = ["FriendlyInteraction", "UnfriendlyInteraction"]  # "NeutralInteraction",
                titles = ["Friendly Interaction", "Unfriendly Interaction"]  # "Neutral Interaction",
                for idx_phase, phase in enumerate(phases):
                    # idx_phase = 0
                    # phase = phases[idx_phase]
                    df_phase = df_group.loc[df_group['event'] == phase]

                    times = df_phase["time"].unique()
                    mean = df_phase.groupby("time")[column_name].mean()
                    sem = df_phase.groupby("time")[column_name].sem()

                    # Plot line
                    axes[idx_group, physio_idx].plot(times, mean, '-', color=colors[idx_phase], label=titles[idx_phase])
                    axes[idx_group, physio_idx].fill_between(times, mean + sem, mean - sem, alpha=0.2, color=colors[idx_phase])

                y_pos = axes[idx_group, physio_idx].get_ylim()[0] + 0.02 * (
                        axes[idx_group, physio_idx].get_ylim()[1] - axes[idx_group, physio_idx].get_ylim()[0])

                # Calculate differences for values
                df_diff = df_group.loc[df_group["event"].isin(phases)]
                df_diff = df_diff.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()

                # Calculate t-values for pairwise comparisons
                t_vals = pd.DataFrame()
                t_vals["t"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["FriendlyInteraction"], df_diff["UnfriendlyInteraction"], nan_policy="omit").statistic)
                t_vals["df"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["FriendlyInteraction"], df_diff["UnfriendlyInteraction"], nan_policy="omit").df)
                t_vals["threshold"] = t_vals.apply(lambda x: t.ppf(1 - 0.025, x["df"]), axis=1)  # two-sided test
                t_vals["t"] = t_vals["t"].abs()
                t_vals["sig"] = t_vals["t"] > t_vals["threshold"]
                t_vals = t_vals.reset_index()

                # Add cluster IDs
                t_vals["idx_cluster"] = 0
                idx_cluster = 1
                for idx_row, row in t_vals.iterrows():
                    # idx_row = 0
                    # row = t_vals.iloc[idx_row, ]
                    if row["sig"]:
                        t_vals.loc[idx_row, "idx_cluster"] = idx_cluster
                        if (idx_row < len(t_vals) - 1) and not (t_vals.loc[idx_row + 1, "sig"]):
                            idx_cluster += 1

                # Get "cluster mass" = sum of t-values per cluster
                cluster_mass = t_vals.groupby("idx_cluster")["t"].sum().reset_index()
                cluster_mass = cluster_mass.loc[cluster_mass["idx_cluster"] != 0].reset_index()

                pd.options.mode.chained_assignment = None

                # Create distribution of possible cluster masses using permutations
                cluster_distribution = []
                for i in tqdm(np.arange(0, permutations)):
                    df_shuffle = pd.DataFrame()
                    df_subset = df_group.loc[df_group["event"].isin(phases)]
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
                    else:
                        cluster_distribution.append(0)

                # Find percentile of t-values of identified clusters and get corresponding p-value
                cluster_mass["p-val"] = cluster_mass.apply(lambda x: 1 - percentileofscore(cluster_distribution, x["t"]) / 100, axis=1)
                t_vals = t_vals.merge(cluster_mass[["idx_cluster", "p-val"]], on='idx_cluster', how="left")

                # If p-value of cluster < .05 add cluster to plot
                for timepoint in t_vals["time"].unique():
                    # timepoint = 0
                    p = t_vals.loc[(t_vals["time"] == timepoint), "p-val"].item()
                    if p < 0.05:
                        axes[idx_group, physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint + 0.1, linewidth=5, color='gold')

                # Style Plot
                axes[idx_group, physio_idx].set_xlim([0, 5])
                axes[idx_group, physio_idx].set_ylabel(ylabel)
                if idx_group == 0:
                    axes[idx_group, physio_idx].set_title(f"{ylabel.split(' [')[0].replace(' (BPM)', '')}",fontweight='bold')  # (N = {len(df['VP'].unique())})
                elif idx_group == 1:
                    axes[idx_group, physio_idx].set_xlabel("Seconds after Interaction Onset")
                axes[idx_group, physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

            axes[idx_group, 2].legend(loc="upper right")
            axes[idx_group, 0].text(-0.7, np.mean(axes[idx_group, 0].get_ylim()), f"{SA_group}", color="k", fontweight="bold", fontsize="large", verticalalignment='center', rotation=90)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        for physio_idx, (physiology, column_name, ylabel) in enumerate(
                zip(["hr", "eda", "pupil"], ["ECG", "EDA", "pupil"], ylabels)):
            # physio_idx = 0
            # physiology, column_name, ylabel = physiologies[physio_idx], dvs[physio_idx], ylabels[physio_idx]
            df = pd.read_csv(os.path.join(filepath, f'{physiology}_interaction.csv'), decimal='.', sep=';')
            print(f"{physiology.upper()}: N = {len(df['VP'].unique())}")
            if physiology == "eda":
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

            y_pos = axes[physio_idx].get_ylim()[0] + 0.02 * (
                        axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])

            # Calculate differences for values
            df_diff = df.loc[df["event"].isin(phases)]
            df_diff = df_diff.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()

            # Calculate t-values for pairwise comparisons
            t_vals = pd.DataFrame()
            t_vals["t"] = df_diff.groupby("time").apply(
                lambda df_diff: ttest_rel(df_diff["FriendlyInteraction"], df_diff["UnfriendlyInteraction"],
                                          nan_policy="omit").statistic)
            t_vals["df"] = df_diff.groupby("time").apply(
                lambda df_diff: ttest_rel(df_diff["FriendlyInteraction"], df_diff["UnfriendlyInteraction"],
                                          nan_policy="omit").df)
            t_vals["threshold"] = t_vals.apply(lambda x: t.ppf(1 - 0.025, x["df"]), axis=1)  # two-sided test
            t_vals["t"] = t_vals["t"].abs()
            t_vals["sig"] = t_vals["t"] > t_vals["threshold"]
            t_vals = t_vals.reset_index()

            # Add cluster IDs
            t_vals["idx_cluster"] = 0
            idx_cluster = 1
            for idx_row, row in t_vals.iterrows():
                # idx_row = 0
                # row = t_vals.iloc[idx_row, ]
                if row["sig"]:
                    t_vals.loc[idx_row, "idx_cluster"] = idx_cluster
                    if (idx_row < len(t_vals) - 1) and not (t_vals.loc[idx_row + 1, "sig"]):
                        idx_cluster += 1

            # Get "cluster mass" = sum of t-values per cluster
            cluster_mass = t_vals.groupby("idx_cluster")["t"].sum().reset_index()
            cluster_mass = cluster_mass.loc[cluster_mass["idx_cluster"] != 0].reset_index()

            pd.options.mode.chained_assignment = None

            # Create distribution of possible cluster masses using permutations
            cluster_distribution = []
            for i in tqdm(np.arange(0, permutations)):
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
                t_vals_shuffle["t"] = df_shuffle.groupby("time").apply(
                    lambda df_shuffle: ttest_rel(df_shuffle["FriendlyInteraction"], df_shuffle["UnfriendlyInteraction"],
                                                 nan_policy="omit").statistic)
                t_vals_shuffle["df"] = df_shuffle.groupby("time").apply(
                    lambda df_shuffle: ttest_rel(df_shuffle["FriendlyInteraction"], df_shuffle["UnfriendlyInteraction"],
                                                 nan_policy="omit").df)
                t_vals_shuffle["threshold"] = t_vals_shuffle.apply(lambda x: t.ppf(1 - 0.025, x["df"]),
                                                                   axis=1)  # two-sided test
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
                else:
                    cluster_distribution.append(0)

            # Find percentile of t-values of identified clusters and get corresponding p-value
            cluster_mass["p-val"] = cluster_mass.apply(lambda x: 1 - percentileofscore(cluster_distribution, x["t"]) / 100,
                                                       axis=1)
            t_vals = t_vals.merge(cluster_mass[["idx_cluster", "p-val"]], on='idx_cluster', how="left")

            # If p-value of cluster < .05 add cluster to plot
            for timepoint in t_vals["time"].unique():
                # timepoint = 0
                p = t_vals.loc[(t_vals["time"] == timepoint), "p-val"].item()
                if p < 0.05:
                    axes[physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint + 0.1, linewidth=5, color='gold')

            # Style Plot
            axes[physio_idx].set_xlim([0, 5])
            axes[physio_idx].set_ylabel(ylabel)
            axes[physio_idx].set_title(f"{ylabel.split(' [')[0].replace(' (BPM)', '')}", fontweight='bold')  # (N = {len(df['VP'].unique())})
            axes[physio_idx].set_xlabel("Seconds after Interaction Onset")
            axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

        axes[2].legend(loc="upper right")
    plt.tight_layout()


# Clicks
def plot_physio_click(filepath, split=False, median=None, SA_score="SPAI", permutations=1000):
    physiologies = ["hr", "eda", "pupil", "hrv_hf"]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]"]
    dvs = ["ECG", "EDA", "pupil"]
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

    if split:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
        for idx_group, SA_group in enumerate(["HSA", "LSA"]):
            # idx_group = 0
            # SA_group = "LSA"
            for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(physiologies, dvs, ylabels)):
                # physio_idx = 0
                # physiology, column_name, ylabel = physiologies[physio_idx], dvs[physio_idx], ylabels[physio_idx]
                df = pd.read_csv(os.path.join(filepath, f'{physiology}_interaction.csv'), decimal='.', sep=';')
                if SA_group == "LSA":
                    df_group = df.loc[df[SA_score] < median]
                else:
                    df_group = df.loc[df[SA_score] >= median]
                print(f"{physiology.upper()}: N - {SA_group} = {len(df_group['VP'].unique())}")
                if physiology == "eda":
                    for vp in df_group["VP"].unique():
                        # vp = 2
                        df_vp = df_group.loc[df_group["VP"] == vp]
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
                                subsequent_peak_times = [peak_time for peak_time in peak_times if
                                                         (onset - peak_time) < 0]
                                if len(subsequent_peak_times) > 0:
                                    peak_idx = list(peak_times).index(
                                        min(subsequent_peak_times, key=lambda x: abs(x - onset)))
                                    rise_time = (peak_times[peak_idx] - onset) / 10
                                    amplitude = peak_values[peak_idx] - onset_values[onset_idx]
                                    if (rise_time > 0.1) & (rise_time < 10) & (amplitude >= amplitude_min):
                                        scr_onsets.append(onset)
                                        scr_peaks.append(peak_times[peak_idx])
                                        scr_amplitudes.append(amplitude)
                                        scr_risetimes.append(rise_time)
                            if len(scr_amplitudes) == 0:
                                df = df.loc[~((df["VP"] == vp) & (df["event"] == event))]

                df_group = df_group.loc[df_group["time"] < 3]

                phases = ["Test_FriendlyWasClicked", "Test_UnfriendlyWasClicked"]
                titles = ["Friendly Clicked", "Unfriendly Clicked"]
                for idx_phase, phase in enumerate(phases):
                    # idx_phase = 0
                    # phase = phases[idx_phase]
                    df_phase = df_group.loc[df_group['event'] == phase]

                    times = df_phase["time"].unique()
                    mean = df_phase.groupby("time")[column_name].mean()
                    sem = df_phase.groupby("time")[column_name].sem()

                    # Plot line
                    axes[idx_group, physio_idx].plot(times, mean, '-', color=colors[idx_phase], label=titles[idx_phase])
                    axes[idx_group, physio_idx].fill_between(times, mean + sem, mean - sem, alpha=0.2, color=colors[idx_phase])

                y_pos = axes[idx_group, physio_idx].get_ylim()[0] + 0.02 * (
                        axes[idx_group, physio_idx].get_ylim()[1] - axes[idx_group, physio_idx].get_ylim()[0])

                df_diff = df_group.loc[df_group["event"].isin(phases)]
                df_diff = df_diff.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
                df_diff = df_diff.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
                t_vals = pd.DataFrame()
                t_vals["t"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["Test_FriendlyWasClicked"], df_diff["Test_UnfriendlyWasClicked"], nan_policy="omit").statistic)
                t_vals["df"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["Test_FriendlyWasClicked"], df_diff["Test_UnfriendlyWasClicked"], nan_policy="omit").df)
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
                for i in tqdm(np.arange(0, permutations)):
                    df_shuffle = pd.DataFrame()
                    df_subset = df_group.loc[df_group["event"].isin(phases)]
                    for vp in df_subset["VP"].unique():
                        # vp = df_subset["VP"].unique()[0]
                        df_vp = df_subset.loc[df_subset["VP"] == vp]
                        rand_int = np.random.randint(0, 2)
                        if rand_int == 0:
                            df_vp["event"] = df_vp["event"].replace(
                                {"Test_FriendlyWasClicked": "Test_UnfriendlyWasClicked",
                                 "Test_UnfriendlyWasClicked": "Test_FriendlyWasClicked"})
                        df_shuffle = pd.concat([df_shuffle, df_vp])

                    df_shuffle = df_shuffle.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
                    df_shuffle = df_shuffle.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
                    t_vals_shuffle = pd.DataFrame()
                    t_vals_shuffle["t"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle["Test_FriendlyWasClicked"], df_shuffle["Test_UnfriendlyWasClicked"], nan_policy="omit").statistic)
                    t_vals_shuffle["df"] = df_shuffle.groupby("time").apply( lambda df_shuffle: ttest_rel(df_shuffle["Test_FriendlyWasClicked"], df_shuffle["Test_UnfriendlyWasClicked"], nan_policy="omit").df)
                    t_vals_shuffle["threshold"] = t_vals_shuffle.apply(lambda x: t.ppf(1 - 0.025, x["df"]),
                                                                       axis=1)  # two-sided test
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
                    else:
                        cluster_distribution.append(0)

                cluster_mass["p-val"] = cluster_mass.apply(lambda x: 1 - percentileofscore(cluster_distribution, x["t"]) / 100, axis=1)
                t_vals = t_vals.merge(cluster_mass[["idx_cluster", "p-val"]], on='idx_cluster', how="left")

                for timepoint in t_vals["time"].unique():
                    # timepoint = 0
                    p = t_vals.loc[(t_vals["time"] == timepoint), "p-val"].item()
                    if p < 0.05:
                        axes[idx_group, physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint + 0.1, linewidth=5, color='gold')

                # Style Plot
                axes[idx_group, physio_idx].set_xlim([0, 2.9])
                axes[idx_group, physio_idx].set_ylabel(ylabel)
                if idx_group == 0:
                    axes[idx_group, physio_idx].set_title(f"{ylabel.split(' [')[0]}", fontweight='bold')
                elif idx_group == 1:
                    axes[idx_group, physio_idx].set_xlabel("Seconds after Click")
                axes[idx_group, physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

            axes[idx_group, 2].legend(loc="upper right")
            axes[idx_group, 0].text(-0.7, np.mean(axes[idx_group, 0].get_ylim()), f"{SA_group}", color="k", fontweight="bold", fontsize="large", verticalalignment='center', rotation=90)

    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(physiologies, dvs, ylabels)):
            # physio_idx = 0
            # physiology, column_name, ylabel = physiologies[physio_idx], dvs[physio_idx], ylabels[physio_idx]
            df = pd.read_csv(os.path.join(filepath, f'{physiology}_interaction.csv'), decimal='.', sep=';')
            print(f"{physiology.upper()}: N = {len(df['VP'].unique())}")
            if physiology == "eda":
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

            y_pos = axes[physio_idx].get_ylim()[0] + 0.02 * (
                        axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])

            df_diff = df.loc[df["event"].isin(phases)]
            df_diff = df_diff.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
            df_diff = df_diff.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
            t_vals = pd.DataFrame()
            t_vals["t"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["Test_FriendlyWasClicked"], df_diff["Test_UnfriendlyWasClicked"], nan_policy="omit").statistic)
            t_vals["df"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff["Test_FriendlyWasClicked"], df_diff["Test_UnfriendlyWasClicked"], nan_policy="omit").df)
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
            for i in tqdm(np.arange(0, permutations)):
                df_shuffle = pd.DataFrame()
                df_subset = df.loc[df["event"].isin(phases)]
                for vp in df_subset["VP"].unique():
                    # vp = df_subset["VP"].unique()[0]
                    df_vp = df_subset.loc[df_subset["VP"] == vp]
                    rand_int = np.random.randint(0, 2)
                    if rand_int == 0:
                        df_vp["event"] = df_vp["event"].replace({"Test_FriendlyWasClicked": "Test_UnfriendlyWasClicked",
                                                                 "Test_UnfriendlyWasClicked": "Test_FriendlyWasClicked"})
                    df_shuffle = pd.concat([df_shuffle, df_vp])

                df_shuffle = df_shuffle.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
                df_shuffle = df_shuffle.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
                t_vals_shuffle = pd.DataFrame()
                t_vals_shuffle["t"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle["Test_FriendlyWasClicked"], df_shuffle["Test_UnfriendlyWasClicked"], nan_policy="omit").statistic)
                t_vals_shuffle["df"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle["Test_FriendlyWasClicked"], df_shuffle["Test_UnfriendlyWasClicked"], nan_policy="omit").df)
                t_vals_shuffle["threshold"] = t_vals_shuffle.apply(lambda x: t.ppf(1 - 0.025, x["df"]),axis=1)  # two-sided test
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
                else:
                    cluster_distribution.append(0)

            cluster_mass["p-val"] = cluster_mass.apply(lambda x: 1 - percentileofscore(cluster_distribution, x["t"]) / 100, axis=1)
            t_vals = t_vals.merge(cluster_mass[["idx_cluster", "p-val"]], on='idx_cluster', how="left")

            for timepoint in t_vals["time"].unique():
                # timepoint = 0
                p = t_vals.loc[(t_vals["time"] == timepoint), "p-val"].item()
                if p < 0.05:
                    axes[physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint + 0.1, linewidth=5, color='gold')

            # Style Plot
            axes[physio_idx].set_xlim([0, 2.9])
            axes[physio_idx].set_ylabel(ylabel)
            axes[physio_idx].set_title(f"{ylabel.split(' [')[0]}", fontweight='bold')
            axes[physio_idx].set_xlabel("Seconds after Click")
            axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

        axes[2].legend()
    plt.tight_layout()


def plot_physio_visible(filepath, split=False, median=None, SA_score="SPAI", permutations=1000):
    physiologies = ["hr", "eda", "pupil", "hrv_hf"]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]"]
    dvs = ["ECG", "EDA", "pupil"]
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

    if split:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
        for idx_group, SA_group in enumerate(["HSA", "LSA"]):
            # idx_group = 0
            # SA_group = "LSA"
            for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(physiologies, dvs, ylabels)):
                # physio_idx = 0
                # physiology, column_name, ylabel = physiologies[physio_idx], dvs[physio_idx], ylabels[physio_idx]
                df = pd.read_csv(os.path.join(filepath, f'{physiology}_interaction.csv'), decimal='.', sep=';')
                if SA_group == "LSA":
                    df_group = df.loc[df[SA_score] < median]
                else:
                    df_group = df.loc[df[SA_score] >= median]
                print(f"{physiology.upper()}: N - {SA_group} = {len(df_group['VP'].unique())}")
                if physiology == "eda":
                    for vp in df["VP"].unique():
                        # vp = 2
                        df_vp = df_group.loc[df_group["VP"] == vp]
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
                                subsequent_peak_times = [peak_time for peak_time in peak_times if
                                                         (onset - peak_time) < 0]
                                if len(subsequent_peak_times) > 0:
                                    peak_idx = list(peak_times).index(
                                        min(subsequent_peak_times, key=lambda x: abs(x - onset)))
                                    rise_time = (peak_times[peak_idx] - onset) / 10
                                    amplitude = peak_values[peak_idx] - onset_values[onset_idx]
                                    if (rise_time > 0.1) & (rise_time < 10) & (amplitude >= amplitude_min):
                                        scr_onsets.append(onset)
                                        scr_peaks.append(peak_times[peak_idx])
                                        scr_amplitudes.append(amplitude)
                                        scr_risetimes.append(rise_time)
                            if len(scr_amplitudes) == 0:
                                df = df.loc[~((df["VP"] == vp) & (df["event"] == event))]

                df_group = df_group.loc[df_group["time"] < 6]
                df_group = df_group.loc[df_group["event"].str.contains("Visible") & ~(df_group["event"].str.contains("Actor"))]
                df_group = df_group.loc[~(df_group['event'].str.contains("Friendly") & df_group['event'].str.contains("Unfriendly"))]

                phases = ["Test_FriendlyVisible", "Test_UnfriendlyVisible"]
                titles = ["Friendly Agent Visible", "Unfriendly Agent Visbile"]
                for idx_phase, phase in enumerate(phases):
                    # idx_phase = 0
                    # phase = phases[idx_phase]
                    df_phase = df_group.loc[df_group['event'] == phase]

                    times = df_phase["time"].unique()
                    mean = df_phase.groupby("time")[column_name].mean()
                    sem = df_phase.groupby("time")[column_name].sem()

                    # Plot line
                    axes[idx_group, physio_idx].plot(times, mean, '-', color=colors[idx_phase], label=titles[idx_phase])
                    axes[idx_group, physio_idx].fill_between(times, mean + sem, mean - sem, alpha=0.2, color=colors[idx_phase])

                y_pos = axes[idx_group, physio_idx].get_ylim()[0] + 0.02 * (axes[idx_group, physio_idx].get_ylim()[1] - axes[idx_group, physio_idx].get_ylim()[0])

                df_diff = df_group.loc[df_group["event"].isin(phases)]
                df_diff = df_diff.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
                df_diff = df_diff.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
                t_vals = pd.DataFrame()
                t_vals["t"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff[phases[0]], df_diff[phases[1]], nan_policy="omit").statistic)
                t_vals["df"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff[phases[0]], df_diff[phases[1]], nan_policy="omit").df)
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
                for i in tqdm(np.arange(0, permutations)):
                    df_shuffle = pd.DataFrame()
                    df_subset = df_group.loc[df_group["event"].isin(phases)]
                    for vp in df_subset["VP"].unique():
                        # vp = df_subset["VP"].unique()[0]
                        df_vp = df_subset.loc[df_subset["VP"] == vp]
                        rand_int = np.random.randint(0, 2)
                        if rand_int == 0:
                            df_vp["event"] = df_vp["event"].replace({phases[0]: phases[1], phases[1]: phases[0]})
                        df_shuffle = pd.concat([df_shuffle, df_vp])

                    df_shuffle = df_shuffle.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
                    df_shuffle = df_shuffle.pivot(index=['VP', 'time'], columns=['event'],
                                                  values=column_name).reset_index()
                    t_vals_shuffle = pd.DataFrame()
                    t_vals_shuffle["t"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle[phases[0]], df_shuffle[phases[1]], nan_policy="omit").statistic)
                    t_vals_shuffle["df"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle[phases[0]], df_shuffle[phases[1]], nan_policy="omit").df)
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
                    cluster_mass_shuffle = cluster_mass_shuffle.loc[
                        cluster_mass_shuffle["idx_cluster"] != 0].reset_index()
                    if len(cluster_mass_shuffle) > 0:
                        cluster_distribution.append(cluster_mass_shuffle["t"].max())
                    else:
                        cluster_distribution.append(0)

                cluster_mass["p-val"] = cluster_mass.apply(lambda x: 1 - percentileofscore(cluster_distribution, x["t"]) / 100,axis=1)
                t_vals = t_vals.merge(cluster_mass[["idx_cluster", "p-val"]], on='idx_cluster', how="left")

                for timepoint in t_vals["time"].unique():
                    # timepoint = 0
                    p = t_vals.loc[(t_vals["time"] == timepoint), "p-val"].item()
                    if p < 0.05:
                        axes[idx_group, physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint + 0.1, linewidth=5, color='gold')

                # Style Plot
                axes[idx_group, physio_idx].set_xlim([0, 5])
                axes[idx_group, physio_idx].set_ylabel(ylabel)
                if idx_group == 0:
                    axes[idx_group, physio_idx].set_title(f"{ylabel.split(' [')[0]}", fontweight='bold')
                elif idx_group == 1:
                    axes[idx_group, physio_idx].set_xlabel("Seconds after agent showed up")
                axes[idx_group, physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

            axes[idx_group, 2].legend(loc="upper right")
            axes[idx_group, 0].text(-0.7, np.mean(axes[idx_group, 0].get_ylim()), f"{SA_group}", color="k", fontweight="bold", fontsize="large", verticalalignment='center', rotation=90)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(physiologies, dvs, ylabels)):
            # physio_idx = 0
            # physiology, column_name, ylabel = physiologies[physio_idx], dvs[physio_idx], ylabels[physio_idx]
            df = pd.read_csv(os.path.join(filepath, f'{physiology}_interaction.csv'), decimal='.', sep=';')
            print(f"{physiology.upper()}: N = {len(df['VP'].unique())}")
            if physiology == "eda":
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

            df = df.loc[df["time"] < 6]
            df = df.loc[df["event"].str.contains("Visible")]
            df = df.loc[~(df['event'].str.contains("Friendly") & df['event'].str.contains("Unfriendly"))]

            phases = ["Test_FriendlyVisible", "Test_UnfriendlyVisible"]
            titles = ["Friendly Agent Visible", "Unfriendly Agent Visbile"]
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
            df_diff = df_diff.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
            df_diff = df_diff.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
            t_vals = pd.DataFrame()
            t_vals["t"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff[phases[0]], df_diff[phases[1]], nan_policy="omit").statistic)
            t_vals["df"] = df_diff.groupby("time").apply(lambda df_diff: ttest_rel(df_diff[phases[0]], df_diff[phases[1]], nan_policy="omit").df)
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
            for i in tqdm(np.arange(0, permutations)):
                df_shuffle = pd.DataFrame()
                df_subset = df.loc[df["event"].isin(phases)]
                for vp in df_subset["VP"].unique():
                    # vp = df_subset["VP"].unique()[0]
                    df_vp = df_subset.loc[df_subset["VP"] == vp]
                    rand_int = np.random.randint(0, 2)
                    if rand_int == 0:
                        df_vp["event"] = df_vp["event"].replace({phases[0]: phases[1],
                                                                 phases[1]: phases[0]})
                    df_shuffle = pd.concat([df_shuffle, df_vp])

                df_shuffle = df_shuffle.groupby(['VP', 'event', 'time']).mean(numeric_only=True).reset_index()
                df_shuffle = df_shuffle.pivot(index=['VP', 'time'], columns=['event'], values=column_name).reset_index()
                t_vals_shuffle = pd.DataFrame()
                t_vals_shuffle["t"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle[phases[0]], df_shuffle[phases[1]], nan_policy="omit").statistic)
                t_vals_shuffle["df"] = df_shuffle.groupby("time").apply(lambda df_shuffle: ttest_rel(df_shuffle[phases[0]], df_shuffle[phases[1]], nan_policy="omit").df)
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
                else:
                    cluster_distribution.append(0)

            cluster_mass["p-val"] = cluster_mass.apply(lambda x: 1 - percentileofscore(cluster_distribution, x["t"]) / 100,
                                                       axis=1)
            t_vals = t_vals.merge(cluster_mass[["idx_cluster", "p-val"]], on='idx_cluster', how="left")

            for timepoint in t_vals["time"].unique():
                # timepoint = 0
                p = t_vals.loc[(t_vals["time"] == timepoint), "p-val"].item()
                if p < 0.05:
                    axes[physio_idx].hlines(y=y_pos, xmin=timepoint, xmax=timepoint + 0.1, linewidth=5, color='gold')

            # Style Plot
            axes[physio_idx].set_xlim([0, 5])
            axes[physio_idx].set_ylabel(ylabel)
            axes[physio_idx].set_title(f"{ylabel.split(' [')[0]}", fontweight='bold')
            axes[physio_idx].set_xlabel("Seconds after agent showed up")
            axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)
        axes[2].legend()
    plt.tight_layout()


# Test Phase
def plot_physio_test(filepath, save_path, SA_score="SPAI"):
    physiologies = ["hr", "eda", "pupil", "hrv_hf", "hrv_rmssd"]
    physiologies = physiologies[0:3]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]",
               "Heart Rate Variability\n(High Frequency)", "Heart Rate Variability (RMSSD)"]
    dvs = ["HR (Mean)", "SCL (Mean)", "Pupil Dilation (Mean)", "HRV (HF_nu)", "HRV (RMSSD)"]

    fig, axes = plt.subplots(nrows=1, ncols=len(physiologies), figsize=(18, 5))
    for physio_idx, (physiology, ylabel, dv) in enumerate(zip(physiologies, ylabels[0:len(physiologies)], dvs[0:len(physiologies)])):
        # physio_idx = 0
        # physiology, ylabel, dv = physiologies[physio_idx], ylabels[physio_idx], dvs[physio_idx]
        if "hrv" in physiology:
            df = pd.read_csv(os.path.join(filepath, f'hr.csv'), decimal='.', sep=';')
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
        print(f"{physiology.upper()}: N = {len(df['VP'].unique())}")
        # Baseline Correction
        df_baseline = df.loc[df["Phase"].str.contains("Orientation")]
        df_baseline["Baseline"] = df_baseline[dv]
        df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
        df_subset = df_subset.merge(df_baseline[["VP", "Baseline"]], on="VP", how="left")
        df_subset[dv] = df_subset[dv] - df_subset["Baseline"]

        df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
        df_test = df_subset.copy()
        if df_test['Phase'].str.contains("Visible").any():
            df_test = df_test.loc[df_test['Phase'].str.contains("Visible") & ~df_test['Phase'].str.contains("Actor")]
            titles = ["Friendly Agent\nVisible", "Unfriendly Agent\nVisible"]
            df_test = df_test.loc[~(df_test['Phase'].str.contains("Friendly") & df_test['Phase'].str.contains("Unfriendly"))]
        else:
            titles = ["Room with\nFriendly Agent", "Room with\nUnfriendly Agent"]

        df_test = df_test.dropna(subset=dv)
        df_test = df_test.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()

        conditions = ["friendly", "unfriendly"]

        for idx_condition, condition in enumerate(conditions):
            # idx_condition = 0
            # condition = conditions[idx_condition]
            df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)

            boxWidth = 1 / (len(conditions) + 1)
            pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

            red = '#E2001A'
            green = '#B1C800'
            colors = [green, red]

            # Plot raw data points
            for i in range(len(df_cond)):
                # i = 0
                x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
                y = df_cond.reset_index().loc[i, dv].item()
                axes[physio_idx].plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
            fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
            boxprops = dict(color=colors[idx_condition])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, dv].values,
                                                     numb_iterations=5000,
                                                     alpha=alpha,
                                                     as_dict=True,
                                                     func='mean')

            axes[physio_idx].boxplot([df_cond.loc[:, dv].values],
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

            axes[physio_idx].errorbar(x=pos[idx_condition], y=bootstrapping_dict['mean'],
                                      yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                                      elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

        df_crit = df_test.copy()
        df_crit = df_crit.loc[df_crit["Condition"].isin(conditions)]
        df_crit = df_crit.rename(columns={dv: physiology})
        df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

        formula = f"{physiology} ~ Condition + {SA_score} + " \
                  f"Condition:{SA_score} + (1 | VP)"

        model = pymer4.models.Lmer(formula, data=df_crit)
        model.fit(factors={"Condition": ["friendly", "unfriendly"]}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        sum_sq_error = (sum(i * i for i in model.residuals))
        anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
        print(f"ANOVA: Physio Test (Condition, Phase and {SA_score})")
        print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
        print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
        print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
        estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

        p = anova.loc["Condition", "P-val"].item()
        max = df_test[dv].max()
        if p < 0.05:
            axes[physio_idx].hlines(y=max * 1.10, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
            axes[physio_idx].vlines(x=pos[0], ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
            axes[physio_idx].vlines(x=pos[1], ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
            p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"." if p < 0.1 else ""
            axes[physio_idx].text(np.mean([pos[0], pos[1]]), max * 1.105, p_sign, color='k', horizontalalignment='center')

        anova['NumDF'] = anova['NumDF'].round().astype("str")
        anova['DenomDF'] = anova['DenomDF'].round(2).astype("str")
        anova["df"] = anova['NumDF'].str.cat(anova['DenomDF'], sep=', ')
        anova['F-stat'] = anova['F-stat'].round(2).astype("str")
        anova['P-val'] = anova['P-val'].round(3).astype("str")
        anova.loc[anova['P-val'] == "0.0", "P-val"] = "< .001"
        anova['P-val'] = anova['P-val'].replace({"0.": "."})
        anova['p_eta_2'] = anova['p_eta_2'].round(2).astype("str")

        anova = anova.reset_index(names=['factor'])
        anova = anova[["factor", "F-stat", "df", "P-val", "p_eta_2"]].reset_index()
        anova = anova.drop(columns="index")
        anova.to_csv(os.path.join(save_path, f'lmms_{physiology}_test.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

        axes[physio_idx].set_xticks(pos)
        axes[physio_idx].set_xticklabels([title.replace(" Agent", "\nAgent") for title in titles])
        axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)
        axes[physio_idx].set_ylabel(ylabel)
        axes[physio_idx].set_title(f"{ylabel.split('[')[0]}", fontweight='bold')  # (N = {len(df_subset['VP'].unique())})

    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        titles, loc='center right', bbox_to_anchor=(1, 0.5))
    # fig.subplots_adjust(right=0.7)


# Correlation with SPAI
def plot_physio_test_sad(filepath, SA_score="SPAI"):
    physiologies = ["hr", "eda", "pupil", "hrv_hf", "hrv_rmssd"]
    physiologies = physiologies[0:3]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]",
               "Heart Rate Variability\n(High Frequency)", "Heart Rate Variability (RMSSD)"]
    dvs = ["HR (Mean)", "SCL (Mean)", "Pupil Dilation (Mean)", "HRV (HF_nu)", "HRV (RMSSD)"]

    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]
    conditions = ["friendly", "unfriendly"]
    fig, axes = plt.subplots(nrows=1, ncols=len(physiologies), figsize=(18, 5))
    for physio_idx, (physiology, ylabel, dv) in enumerate(zip(physiologies, ylabels[0:len(physiologies)], dvs[0:len(physiologies)])):
        # physio_idx = 0
        # physiology, ylabel, dv = physiologies[physio_idx], ylabels[physio_idx], dvs[physio_idx]
        if "hrv" in physiology:
            df = pd.read_csv(os.path.join(filepath, f'hr.csv'), decimal='.', sep=';')
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')

        df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
        df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
        df_subset.loc[df_subset['Phase'].str.contains("Habituation"), "phase"] = "Habituation"
        df_subset.loc[df_subset['Phase'].str.contains("Office"), "room"] = "Office"
        df_subset.loc[df_subset['Phase'].str.contains("Living"), "room"] = "Living"
        df_subset.loc[df_subset['Phase'].str.contains("Dining"), "room"] = "Dining"

        df_test = df_subset.loc[df_subset['phase'] == "Test"]
        if df_test['Phase'].str.contains("Visible").any():
            df_test = df_test.loc[df_test['Phase'].str.contains("Visible") & ~df_test['Phase'].str.contains("Actor")]
            titles = ["Friendly Agent\nVisible", "Unfriendly Agent\nVisible"]
            df_test = df_test.loc[~(df_test['Phase'].str.contains("Friendly") & df_test['Phase'].str.contains("Unfriendly"))]
        else:
            titles = ["Room with\nFriendly Agent", "Room with\nUnfriendly Agent"]

        df_test = df_test.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
        df_test = df_test.dropna(subset=dv)
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
            axes[physio_idx].plot(all_x, all_y_est, '-', color=colors[idx_condition])
            axes[physio_idx].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

            p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
            if idx_condition == 0:
                axes[physio_idx].text(df_test[SA_score].min() + 0.01 * np.max(x),
                        0.95 * (df_test[dv].max() - df_test[dv].min()) + df_test[dv].min(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])
            else:
                axes[physio_idx].text(df_test[SA_score].min() + 0.01 * np.max(x),
                        0.91 * (df_test[dv].max() - df_test[dv].min()) + df_test[dv].min(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])

            # Plot raw data points
            axes[physio_idx].plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6,
                    label=titles[idx_condition])

        axes[physio_idx].set_xlabel(SA_score)
        axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)
        axes[physio_idx].set_ylabel(f"{ylabel} in Test Phase")
        axes[physio_idx].set_title(ylabel.split("[")[0], fontweight='bold')
    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        titles, loc='center right', bbox_to_anchor=(1, 0.5))


# Test-Habituation
def plot_physio_diff(filepath, save_path, SA_score="SPAI"):
    physiologies = ["hr", "eda", "pupil", "hrv_hf", "hrv_rmssd"]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]",
               "Heart Rate Variability\n(High Frequency)", "Heart Rate Variability (RMSSD)"]
    dvs = ["HR (Mean)", "SCL (Mean)", "Pupil Dilation (Mean)", "HRV (HF_nu)", "HRV (RMSSD)"]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    for physio_idx, (physiology, ylabel, dv) in enumerate(zip(physiologies[0:3], ylabels[0:3], dvs[0:3])):  # , "hrv_hf", "hrv_rmssd"
        # physio_idx = 0
        # physiology, ylabel, dv = physiologies[physio_idx], ylabels[physio_idx], dvs[physio_idx]
        if "hrv" in physiology:
            df = pd.read_csv(os.path.join(filepath, f'hr.csv'), decimal='.', sep=';')
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')

        print(f"{physiology.upper()}: N = {len(df['VP'].unique())}")

        # Baseline Correction
        df_baseline = df.loc[df["Phase"].str.contains("Orientation")]
        df_baseline["Baseline"] = df_baseline[dv]
        df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
        df_subset = df_subset.merge(df_baseline[["VP", "Baseline"]], on="VP", how="left")
        df_subset[dv] = df_subset[dv] - df_subset["Baseline"]

        df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
        df_subset.loc[df_subset['Phase'].str.contains("Habituation"), "phase"] = "Habituation"
        df_subset.loc[df_subset['Phase'].str.contains("Office"), "room"] = "Office"
        df_subset.loc[df_subset['Phase'].str.contains("Living"), "room"] = "Living"
        df_subset.loc[df_subset['Phase'].str.contains("Dining"), "room"] = "Dining"

        phases = ['Habituation', 'Test']

        if df_subset['Phase'].str.contains("Visible").any():
            df_subset = df_subset.loc[(df_subset['Phase'].str.contains("Visible") & ~df_subset['Phase'].str.contains("Actor")) | df_subset['Phase'].str.contains("Habituation")]
            titles = ["Habituation", "Friendly Agent\nVisible", "Unfriendly Agent\nVisible"]
            df_subset = df_subset.loc[~(df_subset['Phase'].str.contains("Friendly") & df_subset['Phase'].str.contains("Unfriendly"))]
            df_subset = df_subset.loc[~(df_subset['Phase'].str.contains("Neutral") | df_subset['Phase'].str.contains("Unknown"))]
            df_subset.loc[df_subset["Phase"].str.contains("Test"), "phase"] = df_subset["Phase"]
            df_subset.loc[df_subset["Phase"].str.contains("Habituation"), "Condition"] = "neutral"
        else:
            titles = ["Room with\nFriendly Agent", "Room with\nUnfriendly Agent"]
        df_subset = df_subset.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
        df_subset = df_subset.dropna(subset=dv)

        if df_subset['phase'].str.contains("Visible").any():
            conditions = ["neutral", "friendly", "unfriendly"]

            boxWidth = 1 / (len(conditions) + 1)
            pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]
            pos[0] = pos[0] - boxWidth/2

            for idx_condition, condition in enumerate(conditions):
                # idx_condition = 1
                # condition = conditions[idx_condition]
                df_cond = df_subset.loc[df_subset['Condition'] == condition].reset_index(drop=True)

                blue = '#1F82C0'
                red = '#E2001A'
                green = '#B1C800'
                colors = [blue, green, red]

                # Plot raw data points
                for i in range(len(df_cond)):
                    # i = 0
                    x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
                    y = df_cond.reset_index().loc[i, dv].item()
                    axes[physio_idx].plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

                # Plot boxplots
                meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
                medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
                fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
                whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
                capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
                boxprops = dict(color=colors[idx_condition])

                fwr_correction = True
                alpha = (1 - (0.05))
                bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, dv].values,
                                                         numb_iterations=5000,
                                                         alpha=alpha,
                                                         as_dict=True,
                                                         func='mean')

                axes[physio_idx].boxplot([df_cond.loc[:, dv].values],
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

                axes[physio_idx].errorbar(x=pos[idx_condition], y=bootstrapping_dict['mean'],
                                          yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                                          elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

            df_crit = df_subset.copy()
            df_crit.loc[df_crit["phase"].str.contains("Test"), "phase"] = "Test"
            df_crit = df_crit.groupby(["VP", "phase"]).mean(numeric_only=True).reset_index()
            df_crit = df_crit.rename(columns={dv: physiology})
            df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

            formula = f"{physiology} ~ phase + (1 | VP)"

            model = pymer4.models.Lmer(formula, data=df_crit)
            model.fit(factors={"phase": ['Habituation', 'Test']}, summarize=False)
            anova = model.anova(force_orthogonal=True)
            sum_sq_error = (sum(i * i for i in model.residuals))
            anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)

            anova['NumDF'] = anova['NumDF'].round().astype("str")
            anova['DenomDF'] = anova['DenomDF'].round(2).astype("str")
            anova["df"] = anova['NumDF'].str.cat(anova['DenomDF'], sep=', ')
            anova['F-stat'] = anova['F-stat'].round(2).astype("str")
            anova['P-val'] = anova['P-val'].round(3).astype("str")
            anova.loc[anova['P-val'] == "0.0", "P-val"] = "< .001"
            anova['P-val'] = anova['P-val'].replace({"0.": "."})
            anova['p_eta_2'] = anova['p_eta_2'].round(2).astype("str")

            anova = anova.reset_index(names=['factor'])
            anova = anova[["factor", "F-stat", "df", "P-val", "p_eta_2"]].reset_index()
            anova = anova.drop(columns="index")
            anova1 = anova.copy()

            df_crit = df_subset.loc[df_subset["phase"].str.contains("Test")]
            df_crit = df_crit.loc[df_crit["Condition"].isin(conditions)]
            df_crit = df_crit.rename(columns={dv: physiology})
            df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

            formula = f"{physiology} ~ Condition + {SA_score} + " \
                      f"Condition:{SA_score} + (1 | VP)"

            model = pymer4.models.Lmer(formula, data=df_crit)
            model.fit(factors={"Condition": ['friendly', 'unfriendly']}, summarize=False)
            anova = model.anova(force_orthogonal=True)
            sum_sq_error = (sum(i * i for i in model.residuals))
            anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
            print(f"ANOVA: Physio ({physiology}) Test (Condition and {SA_score})")
            print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
            print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
            print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
            estimates, contrasts = model.post_hoc(marginal_vars="Condition", p_adjust="holm")

            p = contrasts.loc[contrasts["Contrast"] == "friendly - unfriendly", "P-val"].item()
            max = df_subset[dv].max()
            if p < 0.05:
                axes[physio_idx].hlines(y=max * 1.10, xmin=pos[1], xmax=pos[2], linewidth=0.7, color='k')
                axes[physio_idx].vlines(x=pos[1], ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
                axes[physio_idx].vlines(x=pos[2], ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
                p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                axes[physio_idx].text(np.mean([pos[1], pos[2]]), max * 1.105, p_sign, color='k', horizontalalignment='center')

            anova['NumDF'] = anova['NumDF'].round().astype("str")
            anova['DenomDF'] = anova['DenomDF'].round(2).astype("str")
            anova["df"] = anova['NumDF'].str.cat(anova['DenomDF'], sep=', ')
            anova['F-stat'] = anova['F-stat'].round(2).astype("str")
            anova['P-val'] = anova['P-val'].round(3).astype("str")
            anova.loc[anova['P-val'] == "0.0", "P-val"] = "< .001"
            anova['P-val'] = anova['P-val'].replace({"0.": "."})
            anova['p_eta_2'] = anova['p_eta_2'].round(2).astype("str")

            anova = anova.reset_index(names=['factor'])
            anova = anova[["factor", "F-stat", "df", "P-val", "p_eta_2"]].reset_index()
            anova = anova.drop(columns="index")
            anova = pd.concat([anova1, anova])
            anova.to_csv(os.path.join(save_path, f'lmms_{physiology}.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

            axes[physio_idx].set_xticks(pos)
            axes[physio_idx].set_xticklabels([title.replace(" Agent", "\nAgent") for title in titles])
            axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)
            axes[physio_idx].set_ylabel(ylabel)
            axes[physio_idx].set_title(f"{ylabel.split('[')[0]}", fontweight='bold')  # (N = {len(df_subset['VP'].unique())})

        else:
            conditions = ["friendly", "unfriendly"]
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
                        red = '#E2001A'
                        green = '#B1C800'
                        colors = [green, red]

                    # Plot raw data points
                    for i in range(len(df_phase)):
                        # i = 0
                        x = random.uniform(pos[idx_phase] - (0.25 * boxWidth), pos[idx_phase] + (0.25 * boxWidth))
                        y = df_phase.reset_index().loc[i, dv].item()
                        axes[physio_idx].plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition],
                                              alpha=0.3)

                    # Plot boxplots
                    meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
                    medianlineprops = dict(linestyle='dashed', linewidth=1, color=colors[idx_condition])
                    fliermarkerprops = dict(marker='o', markersize=1, color=colors[idx_condition])
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

                    axes[physio_idx].boxplot([df_phase.loc[:, dv].values],
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

                    axes[physio_idx].errorbar(x=pos[idx_phase], y=bootstrapping_dict['mean'],
                                              yerr=bootstrapping_dict['mean'] - bootstrapping_dict['lower'],
                                              elinewidth=2, ecolor="dimgrey", marker="s", ms=6, mfc="dimgrey", mew=0)

                df_cond = df_cond.rename(columns={dv: physiology})
                formula = f"{physiology} ~ phase + (1 | VP)"

                model = pymer4.models.Lmer(formula, data=df_cond)
                model.fit(factors={"phase": ['Habituation', 'Test']}, summarize=False)
                anova = model.anova(force_orthogonal=True)
                estimates, contrasts = model.post_hoc(marginal_vars="phase", p_adjust="holm")

                p = anova.loc["phase", "P-val"].item()
                max = df_subset[dv].max()
                if p < 0.05:
                    axes[physio_idx].hlines(y=max * 1.05, xmin=pos[0], xmax=pos[1], linewidth=0.7, color='k')
                    axes[physio_idx].vlines(x=pos[0], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
                    axes[physio_idx].vlines(x=pos[1], ymin=max * 1.04, ymax=max * 1.05, linewidth=0.7, color='k')
                    p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    axes[physio_idx].text(np.mean([pos[0], pos[1]]), max * 1.055, p_sign, color='k', horizontalalignment='center')

            df_crit = df_subset.copy()
            df_crit = df_crit.loc[df_crit["Condition"].isin(conditions)]
            df_crit = df_crit.rename(columns={dv: physiology})
            df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()

            formula = f"{physiology} ~ phase + Condition + {SA_score} + " \
                      f"Condition:{SA_score} + phase:Condition + phase:{SA_score} + " \
                      f"phase:Condition:{SA_score} + (1 | VP)"

            model = pymer4.models.Lmer(formula, data=df_crit)
            model.fit(factors={"phase": ["Habituation", "Test"], "Condition": ["friendly", "unfriendly"]}, summarize=False)
            anova = model.anova(force_orthogonal=True)
            sum_sq_error = (sum(i * i for i in model.residuals))
            anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
            print(f"ANOVA: Physio ({physiology}) Test (Condition, Phase and {SA_score})")
            print(f"Phase Main Effect, F({round(anova.loc['phase', 'NumDF'].item(), 1)}, {round(anova.loc['phase', 'DenomDF'].item(), 1)})={round(anova.loc['phase', 'F-stat'].item(), 2)}, p={round(anova.loc['phase', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['phase', 'p_eta_2'].item(), 2)}")
            print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
            print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
            print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
            print(f"Interaction Condition x Phase, F({round(anova.loc[f'phase:Condition', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:Condition', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:Condition', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:Condition', 'p_eta_2'].item(), 2)}")
            print(f"Interaction {SA_score} x Phase, F({round(anova.loc[f'phase:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:{SA_score}', 'p_eta_2'].item(), 2)}")
            print(f"Interaction Condition x {SA_score} x Phase, F({round(anova.loc[f'phase:Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'phase:Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'phase:Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'phase:Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'phase:Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
            estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="phase", p_adjust="holm")
            print(f"Condition Effect in Test Phase (post-hoc), t({round(contrasts.loc[contrasts['phase'] == 'Test', 'DF'].item(), 2)})={round(contrasts.loc[contrasts['phase'] == 'Test', 'T-stat'].item(), 2)}, p={round(contrasts.loc[contrasts['phase'] == 'Test', 'P-val'].item(), 3)}")

            p = contrasts.loc[contrasts["phase"] == "Test", "P-val"].item()
            max = df_subset[dv].max()
            if p < 0.05:
                axes[physio_idx].hlines(y=max * 1.10, xmin=2 * boxWidth, xmax=1 + 2 * boxWidth, linewidth=0.7, color='k')
                axes[physio_idx].vlines(x=2 * boxWidth, ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
                axes[physio_idx].vlines(x=1 + 2 * boxWidth, ymin=max * 1.09, ymax=max * 1.10, linewidth=0.7, color='k')
                p_sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                axes[physio_idx].text(np.mean([2 * boxWidth, 1 + 2 * boxWidth]), max * 1.105, p_sign, color='k', horizontalalignment='center')

            anova['NumDF'] = anova['NumDF'].round().astype("str")
            anova['DenomDF'] = anova['DenomDF'].round(2).astype("str")
            anova["df"] = anova['NumDF'].str.cat(anova['DenomDF'], sep=', ')
            anova['F-stat'] = anova['F-stat'].round(2).astype("str")
            anova['P-val'] = anova['P-val'].round(3).astype("str")
            anova.loc[anova['P-val'] == "0.0", "P-val"] = "< .001"
            anova['P-val'] = anova['P-val'].replace({"0.": "."})
            anova['p_eta_2'] = anova['p_eta_2'].round(2).astype("str")

            anova = anova.reset_index(names=['factor'])
            anova = anova[["factor", "F-stat", "df", "P-val", "p_eta_2"]].reset_index()
            anova = anova.drop(columns="index")
            anova.to_csv(os.path.join(save_path, f'lmms_{physiology}.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

            axes[physio_idx].set_xticks([x + 1 / 2 for x in range(len(conditions))])
            axes[physio_idx].set_xticklabels(titles)
            axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)
            axes[physio_idx].set_ylabel(ylabel)
            axes[physio_idx].set_title(f"{ylabel.split('[')[0]}", fontweight='bold')  # (N = {len(df_subset['VP'].unique())})

    fig.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor='#1F82C0', markeredgewidth=1,markerfacecolor='#1F82C0', alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
        ["Habituation", "Test (friendly)", "Test (unfriendly)"], loc='center right', bbox_to_anchor=(1, 0.5))
    # fig.subplots_adjust(right=0.7)


# Correlation with SPAI (Test-Habituation)
def plot_physio_diff_sad(filepath, SA_score="SPAI"):
    if not "Wave1" in filepath:
        return

    physiologies = ["hr", "eda", "pupil", "hrv_hf", "hrv_rmssd"]
    physiologies = physiologies[0:3]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance Level [µS]", "Pupil Diameter [mm]",
               "Heart Rate Variability\n(High Frequency)", "Heart Rate Variability (RMSSD)"]
    dvs = ["HR (Mean)", "SCL (Mean)", "Pupil Dilation (Mean)", "HRV (HF_nu)", "HRV (RMSSD)"]
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]

    conditions = ["friendly", "unfriendly"]
    titles = ["Room with\nFriendly Agent", "Room with\nUnfriendly Agent"]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    for physio_idx, (physiology, ylabel, dv) in enumerate(zip(physiologies, ylabels[0:len(physiologies)], dvs[0:len(physiologies)])):
        # physio_idx = 0
        # physiology, ylabel, dv = physiologies[physio_idx], ylabels[physio_idx], dvs[physio_idx]
        if "hrv" in physiology:
            df = pd.read_csv(os.path.join(filepath, f'hr.csv'), decimal='.', sep=';')
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
        df_spai = df[["VP", SA_score]].drop_duplicates(subset="VP")
        df_spai = df_spai.sort_values(by=SA_score)

        df_subset = df.loc[df["Phase"].str.contains("Habituation") | df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
        df_subset.loc[df_subset['Phase'].str.contains("Test"), "phase"] = "Test"
        df_subset.loc[df_subset['Phase'].str.contains("Habituation"), "phase"] = "Habituation"
        df_subset.loc[df_subset['Phase'].str.contains("Office"), "room"] = "Office"
        df_subset.loc[df_subset['Phase'].str.contains("Living"), "room"] = "Living"
        df_subset.loc[df_subset['Phase'].str.contains("Dining"), "room"] = "Dining"

        df_subset = df_subset.groupby(["VP", "phase", "Condition"]).mean(numeric_only=True).reset_index()
        df_subset = df_subset.dropna(subset=dv)
        df_diff = df_subset.pivot(index='VP', columns=['phase', "Condition"], values=dv).reset_index()
        df_diff = df_diff.dropna()
        df_diff["friendly"] = df_diff[("Test"), ("friendly")] - df_diff[("Habituation"), ("friendly")]
        df_diff["unfriendly"] = df_diff[("Test"), ("unfriendly")] - df_diff[("Habituation"), ("unfriendly")]
        df_diff.columns = df_diff.columns.droplevel(level=1)
        df_diff = df_diff[["VP", "friendly", "unfriendly"]]
        df_diff = df_diff.merge(df_spai, on="VP")
        df_diff = pd.melt(df_diff, id_vars=['VP', 'SPAI'], value_vars=['friendly', 'unfriendly'], var_name="Condition", value_name="difference")
        df_diff = df_diff.sort_values(by=SA_score)

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
            axes[physio_idx].plot(all_x, all_y_est, '-', color=colors[idx_condition])
            axes[physio_idx].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

            p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
            if idx_condition == 0:
                axes[physio_idx].text(df_diff[SA_score].min() + 0.01 * np.max(x),
                        0.95 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])
            else:
                axes[physio_idx].text(df_diff[SA_score].min() + 0.01 * np.max(x),
                        0.91 * (df_diff["difference"].max() - df_diff["difference"].min()) + df_diff["difference"].min(),
                        r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                        color=colors[idx_condition])

            # Plot raw data points
            axes[physio_idx].plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])

        axes[physio_idx].set_xlabel(SA_score)
        if "SPAI" in SA_score:
            axes[physio_idx].set_xticks(range(0, 6))
        elif "SIAS" in SA_score:
            axes[physio_idx].set_xticks(range(5, 65, 5))
        axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)
        axes[physio_idx].set_ylabel(f"Difference (Test - Habituation) in {ylabel}")
        axes[physio_idx].set_title(ylabel.split("[")[0], fontweight='bold')
    fig.legend([Line2D([0], [0], color="white", marker='o', markeredgecolor=green, markeredgewidth=1, markerfacecolor=green, alpha=.7),
                Line2D([0], [0], color="white", marker='o', markeredgecolor=red, markeredgewidth=1, markerfacecolor=red, alpha=.7)],
               titles, loc='center right', bbox_to_anchor=(1, 0.5))


if __name__ == '__main__':
    wave = 2
    dir_path = os.getcwd()
    filepath = os.path.join(dir_path, f'Data-Wave{wave}')

    for physiology in ("hr", "eda"):
        df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
        df_check = df.loc[df["Proportion Usable Data"] >= .75]
        print(f"Participants included for {physiology}-analysis: {len(df_check['VP'].unique())}")

    save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Physiology')
    if not os.path.exists(save_path):
        print('creating path for saving')
        os.makedirs(save_path)

    plot_physio_acq(filepath)
    plt.savefig(os.path.join(save_path, f"physiology_acq.png"), dpi=300)
    plt.close()

    if wave == 1:
        plot_physio_click(filepath)
        plt.savefig(os.path.join(save_path, f"physiology_click.png"), dpi=300)
        plt.close()

    elif wave == 2:
        plot_physio_visible(filepath)
        plt.savefig(os.path.join(save_path, f"physiology_vis.png"), dpi=300)
        plt.close()

    SA_score = "SPAI"
    plot_physio_test(filepath, SA_score)
    plt.savefig(os.path.join(save_path, f"physiology_hab-test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plot_physio_test_sad(filepath, SA_score)
    plt.savefig(os.path.join(save_path, f"physiology_test_{SA_score}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plot_physio_diff(filepath, SA_score)
    plt.savefig(os.path.join(save_path, f"physiology_hab-test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if wave == 1:
        plot_physio_diff_sad(filepath, SA_score)
        plt.savefig(os.path.join(save_path, f"physiology_hab-test_{SA_score}.png"), dpi=300, bbox_inches="tight")
        plt.close()
