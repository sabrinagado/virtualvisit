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
from scipy.stats import linregress, t, f, ttest_rel, percentileofscore
from scipy import signal
import statsmodels.api as sm
from statsmodels.formula.api import ols
import rle
from collections import defaultdict
from rpy2.situation import (get_r_home)

os.environ["R_HOME"] = get_r_home()
import pymer4
from tqdm import tqdm
from Code.toolbox import utils


def perform_lmm(df, dv, id, factors):
    if len(factors) == 1:
        formula = f"{dv} ~ {factors[0]} + (1 | {id})"
    elif len(factors) == 2:
        formula = f"{dv} ~ {factors[0]} + {factors[1]} + {factors[0]}:{factors[1]} + (1 | {id})"
    elif len(factors) == 3:
        formula = (f"{dv} ~ {factors[0]} + {factors[1]} + {factors[2]} + "
                   f"{factors[0]}:{factors[1]} + {factors[0]}:{factors[2]} + {factors[1]}:{factors[2]} + "
                   f"{factors[0]}:{factors[1]}:{factors[2]}+ (1 | {id})")
    factor_dict = {}
    for factor in factors:
        if 1 < len(df[factor].unique()) < 10:
            factor_dict[factor] = list(df[factor].unique())

    model = pymer4.models.Lmer(formula, data=df)
    model.fit(factors=factor_dict, summarize=False)
    anova = model.anova()
    return anova[["F-stat"]].transpose()


def perform_rmANCOVA(df, dv, id, factor_cat, factor_cont):
    """
    Calculate Repeated Measures ANOVA for a dataset and return F-statistics as a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - dv: String, name of the dependent variable column.
    - id: String, name of the column with subject IDs.
    - factor_cat: String, name of the column with the categorical factor.
    - factor_cont: String, name of the column with the continuous factor.


    Returns:
    - DataFrame with F-statistics for the main effect of the categorical factor, the main effect of the continuous factor, and their interaction.
    """
    # Formula to include both main effects and their interaction
    formula = f"{dv} ~ C({factor_cat}) * {factor_cont} + (1|{id})"

    # Fit the model
    model = ols(formula, data=df).fit()

    # Perform ANOVA
    aov_table = sm.stats.anova_lm(model, typ=2)

    # Creating a DataFrame and return results
    return pd.DataFrame({factor_cat: [aov_table['PR(>F)'].loc[[f'C({factor_cat})']].item()],
                         factor_cont: [aov_table['PR(>F)'].loc[[f'{factor_cont}']].item()],
                         f"{factor_cat}:{factor_cont}": [aov_table['PR(>F)'].loc[[f'C({factor_cat}):{factor_cont}']].item()]})


# Calculate the statistic for each time point
def Ftest_p(df, id, dv, factors, time, method="anova"):
    if method == "anova":
        df_stat = df.groupby(time).apply(lambda x: perform_rmANCOVA(x, dv, id, factors[0], factors[1])).reset_index()
    elif method == "lmm":
        utils.blockPrint()
        df_stat = df.groupby(time).apply(lambda x: perform_lmm(x, dv, id, factors)).reset_index()
        utils.enablePrint()
    df_stat = df_stat.drop(columns="level_1")
    return df_stat


def perform_t_test(df, dv, id, factor):
    # df_test = df.loc[df["time"] == 4]
    factor_levels = list(df[factor].unique())
    df_diff = df.pivot(index=id, columns=factor, values=dv).reset_index()
    ttest = ttest_rel(df_diff[factor_levels[0]], df_diff[factor_levels[1]], nan_policy="omit")
    return ttest.pvalue


# Calculate the statistic for each time point
def ttest_p(df, id, dv, factor, time):
    utils.blockPrint()
    df_stat = df.groupby(time).apply(lambda x: perform_t_test(x, dv, id, factor)).reset_index()
    df_stat.columns = [time, factor]
    return df_stat


def calculate_cluster_length(x, p_value):
    # Find clusters in the vector that are above/below the threshold
    x = rle.encode(abs(x) < p_value)

    # Extract only clusters that are above the threshold
    cluster_lengths = [i for (i, v) in zip(x[1], x[0]) if v]

    # Find the start and end of clusters via the cumsum of cluster lengths. For
    # cluster start, we'll put a 0/FALSE in front, so that we don't run into
    # indexing vec[0]
    cluster_start = [i for (i, v) in zip([0] + list(np.cumsum(x[1])), x[0] + [False]) if v]
    cluster_end = [i for (i, v) in zip(np.cumsum(x[1]), x[0]) if v]

    cluster = pd.DataFrame({'cluster': list(np.arange(0, len(cluster_lengths))),
                            'start': cluster_start,
                            'end': cluster_end,
                            'length': cluster_lengths})
    return cluster


# Calculation of the null distribution of cluster lengths
def null_distribution_cluster_length(df, dv, factors, test, time, id, nperm=1000, method=None):
    null_distribution = defaultdict(list)
    n_columns = 39

    # Create a random permutation of the conditions
    for i in tqdm(np.arange(0, nperm)):
        shuffled_factors = []
        df = df.iloc[:, 0:n_columns]
        for factor in factors:
            # factor = factors[1]
            df_shuffle = df.groupby([id, factor]).mean(numeric_only=True).reset_index()
            df_shuffle = df_shuffle[[id, factor]]
            if len(df_shuffle) > len(df_shuffle[id].unique()):
                df_shuffle[f"shuffled_{factor}"] = df_shuffle.groupby(id)[factor].transform(np.random.permutation)
            else:
                df_shuffle[f"shuffled_{factor}"] = np.random.permutation(df_shuffle[factor])
            df = df.merge(df_shuffle, on=[id, factor])
            shuffled_factors.append(f"shuffled_{factor}")

        if test == "F":
            dist_statistics = Ftest_p(df, dv=dv, id=id, factors=shuffled_factors, time=time, method=method)
        elif test == "t":
            dist_statistics = ttest_p(df, dv=dv, id=id, factor=shuffled_factors[0], time=time)

        for effect in np.arange(0, len(dist_statistics.columns) - 1):
            # effect = 0
            if effect < len(factors):
                cluster = calculate_cluster_length(dist_statistics[f"shuffled_{factors[effect]}"], 0.05)
                length = max(cluster["length"]) if len(cluster) > 0 else 0
                null_distribution[factors[effect]].append(length)
            else:
                cluster = calculate_cluster_length(dist_statistics[f"shuffled_{factors[0]}:shuffled_{factors[1]}"], 0.05)
                length = max(cluster["length"]) if len(cluster) > 0 else 0
                null_distribution[f"{factors[0]}:{factors[1]}"].append(length)

    return pd.DataFrame(null_distribution)


# Acquisition
def plot_physio_acq(filepath, save_path, test="F", SA_score="SPAI", permutations=1000):
    physiologies = ["hr", "eda", "pupil", "hrv_hf"]
    ylabels = ["Heart Rate [BPM]", "Skin Conductance [µS]", "Pupil Diameter [mm]"]
    dvs = ["ECG", "EDA", "pupil"]
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]
    sampling_rate = 10
    sampling_rate_new = 2

    fig, axes = plt.subplots(nrows=1, ncols=len(dvs), figsize=(6*len(dvs), 5))
    for physio_idx, (physiology, column_name, ylabel) in enumerate(zip(physiologies, dvs, ylabels)):
        # physio_idx = 0
        # physiology, column_name, ylabel = physiologies[physio_idx], dvs[physio_idx], ylabels[physio_idx]
        df = pd.read_csv(os.path.join(filepath, f'{physiology}_interaction.csv'), decimal='.', sep=';')
        if not physiology == "pupil":
            problematic_subjects = check_physio(filepath, physiology)
            df = df.loc[~df["VP"].isin(problematic_subjects)]
        df.loc[df["event"].str.contains("Friendly"), "condition"] = "friendly"
        df.loc[df["event"].str.contains("Unfriendly"), "condition"] = "unfriendly"
        df.loc[df["event"].str.contains("Neutral"), "condition"] = "neutral"

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

        phases = ["FriendlyInteraction", "UnfriendlyInteraction"]
        titles = ["Friendly Interaction", "Unfriendly Interaction"]
        df = df.loc[df["event"].isin(phases)]
        df = df.loc[df["time"] <= 5]
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

        if test == "t":
            null_distribution = null_distribution_cluster_length(df, column_name, id="VP", factors=["condition"], test="t", time="time", nperm=permutations)
            critical_lengths_null = null_distribution.quantile(.95, axis=0)
            t_tests = ttest_p(df, dv=column_name, id="VP", factor="condition", time="time")
            cluster_condition = calculate_cluster_length(t_tests["condition"], 0.05)

            cluster_condition["critical_length"] = critical_lengths_null["condition"]
            cluster_condition["p-val"] = cluster_condition.apply(lambda x: 1 - percentileofscore(null_distribution["condition"], x["length"]) / 100, axis=1)
            cluster_condition = cluster_condition.loc[cluster_condition["p-val"] < .05]
            cluster_condition["times_start"] = cluster_condition["start"] / sampling_rate
            cluster_condition["times_end"] = cluster_condition["end"] / sampling_rate
            cluster_condition["effect"] = "condition"
            cluster_condition = cluster_condition[["effect", "cluster", "times_start", "times_end", "p-val"]]
            cluster_condition.to_csv(os.path.join(save_path, f'cluster_{physiology}_t_acq.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

            # If p-value of cluster < .05 add cluster to plot
            y_condition = axes[physio_idx].get_ylim()[0] + 0.02 * (axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])
            for idx_row, row in cluster_condition.iterrows():
                axes[physio_idx].hlines(y=y_condition, xmin=row["times_start"], xmax=row["times_end"], linewidth=5, color='gold')

        elif test == "F":
            null_distribution = null_distribution_cluster_length(df, column_name, id="VP", factors=["condition", SA_score], test="F", method="anova", time="time", nperm=permutations)
            critical_lengths_null = null_distribution.quantile(.95, axis=0)

            F_tests = Ftest_p(df, dv=column_name, id="VP", factors=["condition", SA_score], time="time", method="anova")
            cluster_condition = calculate_cluster_length(F_tests["condition"], 0.05)
            cluster_condition["critical_length"] = critical_lengths_null["condition"]
            cluster_condition["p-val"] = cluster_condition.apply(lambda x: 1 - percentileofscore(null_distribution["condition"], x["length"]) / 100, axis=1)
            cluster_condition = cluster_condition.loc[cluster_condition["p-val"] < .05]
            cluster_condition["times_start"] = cluster_condition["start"] / sampling_rate
            cluster_condition["times_end"] = cluster_condition["end"] / sampling_rate
            cluster_condition["effect"] = "condition"
            cluster_condition = cluster_condition[["effect", "cluster", "times_start", "times_end", "p-val"]]

            cluster_SA = calculate_cluster_length(F_tests["SPAI"], 0.05)
            cluster_SA["critical_length"] = critical_lengths_null["SPAI"]
            cluster_SA["p-val"] = cluster_SA.apply(lambda x: 1 - percentileofscore(null_distribution["SPAI"], x["length"]) / 100, axis=1)
            cluster_SA = cluster_SA.loc[cluster_SA["p-val"] < .05]
            cluster_SA["times_start"] = cluster_SA["start"] / sampling_rate
            cluster_SA["times_end"] = cluster_SA["end"] / sampling_rate
            cluster_SA["effect"] = SA_score
            cluster_SA = cluster_SA[["effect", "cluster", "times_start", "times_end", "p-val"]]

            cluster_int = calculate_cluster_length(F_tests[f"condition:{SA_score}"], 0.05)
            cluster_int["critical_length"] = critical_lengths_null[f"condition:{SA_score}"]
            cluster_int["p-val"] = cluster_int.apply(lambda x: 1 - percentileofscore(null_distribution[f"condition:{SA_score}"], x["length"]) / 100, axis=1)
            cluster_int = cluster_int.loc[cluster_int["p-val"] < .05]
            cluster_int["times_start"] = cluster_int["start"] / sampling_rate
            cluster_int["times_end"] = cluster_int["end"] / sampling_rate
            cluster_int["effect"] = f"condition:{SA_score}"
            cluster_int = cluster_int[["effect", "cluster", "times_start", "times_end", "p-val"]]

            cluster = pd.concat([cluster_condition, cluster_SA, cluster_int])
            cluster.to_csv(os.path.join(save_path, f'cluster_{physiology}_acq.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

            # If p-value of cluster < .05 add cluster to plot
            y_condition = axes[physio_idx].get_ylim()[0] - 0.01 * (axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])
            for idx_row, row in cluster_condition.iterrows():
                axes[physio_idx].hlines(y=y_condition, xmin=row["times_start"], xmax=row["times_end"], linewidth=3, color='gold')

            y_spai = axes[physio_idx].get_ylim()[0] - 0.015 * (axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])
            for idx_row, row in cluster_SA.iterrows():
                axes[physio_idx].hlines(y=y_spai, xmin=row["times_start"], xmax=row["times_end"], linewidth=3, color='dodgerblue')

            y_int = axes[physio_idx].get_ylim()[0] - 0.02 * (axes[physio_idx].get_ylim()[1] - axes[physio_idx].get_ylim()[0])
            for idx_row, row in cluster_int.iterrows():
                axes[physio_idx].hlines(y=y_int, xmin=row["times_start"], xmax=row["times_end"], linewidth=3, color='darkviolet')

            df_resample = df.copy()
            df_resample["time"] = pd.to_timedelta(df_resample["time"], 's')
            df_resample = df_resample.set_index("time")
            df_resample = df_resample.groupby(["VP", "condition"]).resample("0.5S").mean(numeric_only=True)
            df_resample = df_resample.drop(columns="VP")
            df_resample = df_resample.reset_index()
            df_resample["time"] = df_resample["time"].dt.total_seconds()

            formula = (f"{column_name} ~ time + condition + {SA_score} + " \
                       f"time:condition + time:{SA_score} + condition:{SA_score} + " \
                       f"time:condition:{SA_score} + (1 | VP)")

            model = pymer4.models.Lmer(formula, data=df_resample)
            model.fit(factors={"condition": ["friendly", "unfriendly"]}, summarize=False)
            anova = model.anova(force_orthogonal=True)
            sum_sq_error = (sum(i * i for i in model.residuals))
            anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
            print(f"ANOVA: Physio Test (Condition, Phase and {SA_score})")
            print(f"Condition Main Effect, F({round(anova.loc['condition', 'NumDF'].item(), 1)}, {round(anova.loc['condition', 'DenomDF'].item(), 1)})={round(anova.loc['condition', 'F-stat'].item(), 2)}, p={round(anova.loc['condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['condition', 'p_eta_2'].item(), 2)}")
            print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
            print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'condition:{SA_score}', 'p_eta_2'].item(), 2)}")
            estimates, contrasts = model.post_hoc(marginal_vars="condition", p_adjust="holm")

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
            anova.to_csv(os.path.join(save_path, f'lmms_{physiology}_acq.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

        # Style Plot
        axes[physio_idx].set_ylabel(ylabel)
        axes[physio_idx].set_title(f"{ylabel.split(' [')[0].replace(' (BPM)', '')}", fontweight='bold')  # (N = {len(df['VP'].unique())})
        axes[physio_idx].set_xlabel("Seconds after Interaction Onset")
        axes[physio_idx].grid(color='lightgrey', linestyle='-', linewidth=0.3)

    axes[2].legend(loc="upper right")
    plt.tight_layout()
    fig.legend(
        [Line2D([0], [0], color="gold", linewidth=3, alpha=.7),
         Line2D([0], [0], color="dodgerblue", linewidth=3, alpha=.7),
         Line2D([0], [0], color="darkviolet", linewidth=3, alpha=.7)],
        ["Main Effect of Condition", "Main Effect of Social Anxiety", "Interaction of Condition and Social Anxiety"],
        loc='lower center', ncols=3, frameon=False)
    fig.subplots_adjust(bottom=0.17)


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
            problematic_subjects = check_physio(filepath, "hr")
            df = df.loc[~df["VP"].isin(problematic_subjects)]
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
            if not "pupil" in physiology:
                problematic_subjects = check_physio(filepath, physiology)
                df = df.loc[~df["VP"].isin(problematic_subjects)]
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
            problematic_subjects = check_physio(filepath, "hr")
            df = df.loc[~df["VP"].isin(problematic_subjects)]
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
            if not "pupil" in physiology:
                problematic_subjects = check_physio(filepath, physiology)
                df = df.loc[~df["VP"].isin(problematic_subjects)]

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
        # physio_idx = 3
        # physiology, ylabel, dv = physiologies[physio_idx], ylabels[physio_idx], dvs[physio_idx]
        if "hrv" in physiology:
            df = pd.read_csv(os.path.join(filepath, f'hr.csv'), decimal='.', sep=';')
            problematic_subjects = check_physio(filepath, "hr")
            df = df.loc[~df["VP"].isin(problematic_subjects)]
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
            if not "pupil" in physiology:
                problematic_subjects = check_physio(filepath, physiology)
                df = df.loc[~df["VP"].isin(problematic_subjects)]

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
            problematic_subjects = check_physio(filepath, "hr")
            df = df.loc[~df["VP"].isin(problematic_subjects)]
        else:
            df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
            if not "pupil" in physiology:
                problematic_subjects = check_physio(filepath, physiology)
                df = df.loc[~df["VP"].isin(problematic_subjects)]
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


def check_physio(filepath, physiology):
    df = pd.read_csv(os.path.join(filepath, f'{physiology}.csv'), decimal='.', sep=';')
    df = df.loc[~(df["Phase"].str.contains("Interaction") | df["Phase"].str.contains("Clicked") | df["Phase"].str.contains("resting") | df["Phase"].str.contains("Visible"))]
    df.loc[df["Phase"].str.contains("Orientation"), "phase"] = "Orientation"
    df.loc[df["Phase"].str.contains("Habituation"), "phase"] = "Habituation"
    df.loc[df["Phase"].str.contains("Test"), "phase"] = "Test"

    df_grouped = df.groupby(["VP", "phase"]).sum(numeric_only=True).reset_index()
    df_grouped.loc[df_grouped["phase"].str.contains("Orientation"), "total_duration"] = 30
    df_grouped.loc[df_grouped["phase"].str.contains("Habituation"), "total_duration"] = 180
    df_grouped.loc[df_grouped["phase"].str.contains("Test"), "total_duration"] = 180
    df_grouped["prop_duration"] = df_grouped["Duration"]/df_grouped["total_duration"]
    df_grouped.loc[df_grouped["prop_duration"] > 1, "prop_duration"] = 1

    df_grouped = df_grouped.groupby(["VP"]).mean(numeric_only=True).reset_index()

    exclude_vp = list(df_grouped.loc[df_grouped["prop_duration"] < .75, "VP"].unique())
    return exclude_vp


if __name__ == '__main__':
    wave = 1
    dir_path = os.getcwd()
    filepath = os.path.join(dir_path, f'Data-Wave{wave}')

    save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Physiology')
    if not os.path.exists(save_path):
        print('creating path for saving')
        os.makedirs(save_path)

    SA_score = "SPAI"
