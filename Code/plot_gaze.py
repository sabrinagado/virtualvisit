# =============================================================================
# Eye_tracking and Gaze: Proportion of Gaze on Social vs. Non-Social Stimuli
# sensor: HMD & Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.stats import linregress
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
import pymer4

from Code.toolbox import utils

from Code import preproc_scores, preproc_ratings, preproc_gaze


# Visualize ET Validation
def plot_et_validation(vps, filepath):
    df_cal_all = pd.DataFrame(columns=["VP", "time", "position", "x", "y"])
    for vp in vps:
        # vp = vps[1]
        vp = f"0{vp}" if vp < 10 else f"{vp}"
        # print(f"VP: {vp}")

        try:
            files = [item for item in os.listdir(os.path.join(filepath, 'VP_' + vp)) if (item.endswith(".csv"))]
            file = [file for file in files if "etcalibration" in file][0]
            df_cal = pd.read_csv(os.path.join(filepath, 'VP_' + vp, file), sep=';', decimal='.')
        except:
            print("no ET calibration file")
            continue

        for idx_row, row in df_cal.iterrows():
            # idx_row = 0
            # row = df_cal.iloc[idx_row, :]
            position = row["position"]
            x = float(position.split("=")[1].split(",")[0]) + row["x_divergence"]
            y = float(position.split("=")[2]) + row["y_divergence"]
            df_cal_all = pd.concat([df_cal_all, pd.DataFrame({"VP": [vp], "time": [row["time"]], "position": [position], "x": [x], "y": [y]})])

    df_cal_all = df_cal_all.reset_index(drop=True)
    df_cal_all_agg = df_cal_all.groupby(["time", "position"])[["x", "y"]].agg(["mean", "std"]).reset_index()
    df_cal_all_agg = df_cal_all_agg.sort_index()

    points_start = pd.DataFrame(columns=["VP", "x", "y", "color"])
    points_end = pd.DataFrame(columns=["VP", "x", "y", "color"])
    for vp in vps:
        # vp = vps[0]
        vp = f"0{vp}" if vp < 10 else f"{vp}"
        # print(f"VP: {vp}")

        try:
            files = [item for item in os.listdir(os.path.join(filepath, 'VP_' + vp)) if (item.endswith(".csv"))]
            file = [file for file in files if "etcalibration" in file][0]
            df_cal = pd.read_csv(os.path.join(filepath, 'VP_' + vp, file), sep=';', decimal='.')
        except:
            print("no ET calibration file")
            continue

        for idx_row, row in df_cal.loc[df_cal["time"] == "Start"].iterrows():
            # idx_row = 0
            # row = df_cal.iloc[idx_row, :]
            position = row["position"]
            reference = df_cal_all_agg.loc[(df_cal_all_agg[("time",)] == "Start") & (df_cal_all_agg[("position",)] == position)]
            x = float(position.split("=")[1].split(",")[0]) + row["x_divergence"]
            y = float(position.split("=")[2]) + row["y_divergence"]

            if (((x > reference[("x", "mean")] + 3 * reference[("x", "std")]) | (x < reference[("x", "mean")] - 3 * reference[("x", "std")])) | (
                    (y > reference[("y", "mean")] + 3 * reference[("y", "std")]) | (y < reference[("y", "mean")] - 3 * reference[("y", "std")]))).item():
                color = "red"
            else:
                color = "black"
            points_start = pd.concat([points_start, pd.DataFrame({"VP": [vp], "x": [x], "y": [y], "color": [color]})])

        for idx_row, row in df_cal.loc[df_cal["time"] == "End"].iterrows():
            # idx_row = 0
            # row = df_cal.iloc[idx_row, :]
            position = row["position"]
            reference = df_cal_all_agg.loc[(df_cal_all_agg[("time",)] == "End") & (df_cal_all_agg[("position",)] == position)]
            x = float(position.split("=")[1].split(",")[0]) + row["x_divergence"]
            y = float(position.split("=")[2]) + row["y_divergence"]
            if (((x > reference[("x", "mean")] + 3 * reference[("x", "std")]) | (x < reference[("x", "mean")] - 3 * reference[("x", "std")])) | (
                        (y > reference[("y", "mean")] + 3 * reference[("y", "std")]) | (y < reference[("y", "mean")] - 3 * reference[("y", "std")]))).item():
                color = "red"
            else:
                color = "black"
            points_end = pd.concat([points_end, pd.DataFrame({"VP": [vp], "x": [x], "y": [y], "color": [color]})])

    points_cal = pd.DataFrame(columns=["x", "y"])
    for idx_row, row in df_cal.loc[df_cal["time"] == "Start"].iterrows():
        # idx_row = 0
        # row = df_cal.iloc[idx_row, :]
        position = row["position"]
        x = float(position.split("=")[1].split(",")[0])
        y = float(position.split("=")[2])
        points_cal = pd.concat([points_cal, pd.DataFrame({"x": [x], "y": [y]})])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    for idx_row, row in points_cal.iterrows():
        # idx_row = 0
        # row = points_cal.iloc[idx_row, :]
        x = row["x"]
        y = row["y"]
        circle_start = Circle((x, y), 15, fill=False, linestyle="--", linewidth=0.8, color="darkgrey")
        axes[0].add_artist(circle_start)
        circle_end = Circle((x, y), 15, fill=False, linestyle="--", linewidth=0.8, color="darkgrey")
        axes[1].add_artist(circle_end)

    for idx_points, (points, title) in enumerate(zip([points_start, points_end], ["Start", "End"])):
        # idx_points = 0
        # points = points_start
        axes[idx_points].scatter(points["x"], points["y"], marker='+', s=20, color="black", linewidths=0.8)
        axes[idx_points].scatter(points_cal["x"], points_cal["y"], marker='+', s=100, color="blue", linewidths=1)
        axes[idx_points].set_title(title)
        axes[idx_points].set_ylim(points_cal["y"].min()-20, points_cal["y"].max()+20)
        axes[idx_points].set_xlim(points_cal["x"].min() - 20, points_cal["x"].max() + 20)

    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.tight_layout()


# Test-Phase
def plot_gaze_test(df, save_path, wave, dv="Gaze Proportion", SA_score="SPAI"):
    # df = df_gaze
    if dv == "Gaze Proportion":
        y_label = "Proportional Dwell Time on Virtual Agent"
    elif dv == "Switches":
        y_label = "Shifts of Visual Attention Towards Virtual Agent"

    df_grouped = df.groupby(["VP", "Condition"]).sum(numeric_only=True).reset_index()
    df_grouped = df_grouped.drop(columns=SA_score)
    df_grouped = df_grouped.merge(df[["VP", SA_score]].drop_duplicates(subset="VP"), on="VP")

    max = round(df_grouped[dv].max(), 2) * 1.1
    conditions = ["friendly", "unfriendly"]
    labels = ["Friendly\nAgent", "Unfriendly\nAgent"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    boxWidth = 1 / (len(conditions) + 1)
    pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 1
        # condition = conditions[idx_condition]
        df_cond = df_grouped.loc[df_grouped['Condition'] == condition].reset_index(drop=True)

        red = '#E2001A'
        green = '#B1C800'
        blue = '#1F82C0'
        colors = [green, red]

        # Plot raw data points
        for i in range(len(df_cond)):
            # i = 0
            x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
            y = df_cond.loc[i, dv].item()
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
        bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, dv].values,
                                                 numb_iterations=5000,
                                                 alpha=alpha,
                                                 as_dict=True,
                                                 func='mean')

        ax.boxplot([df_cond.loc[:, dv].values],
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

    df_crit = df_grouped.loc[df_grouped["Condition"].isin(conditions)]
    df_crit[SA_score] = (df_crit[SA_score] - df_crit[SA_score].mean()) / df_crit[SA_score].std()
    df_crit = df_crit.rename(columns={dv: "gaze"})

    formula = f"gaze ~ Condition + {SA_score} + Condition:{SA_score} + (1 | VP)"

    max = df_crit["gaze"].max()
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

    ax.set_xticklabels(labels)
    # ax.set_title(f"Spontaneous Fixations", fontweight='bold')  # (N = {len(df_cond['VP'].unique())})
    max = round(df_crit["gaze"].max(), 2) * 1.17
    ax.set_ylim([0, max])
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_ylabel(f"{y_label}s")

    anova['NumDF'] = anova['NumDF'].round().astype("str")
    anova['DenomDF'] = anova['DenomDF'].round(2).astype("str")
    anova["df"] = anova['NumDF'].str.cat(anova['DenomDF'], sep=', ')
    anova['F-stat'] = anova['F-stat'].round(2).astype("str")
    anova['P-val'] = anova['P-val'].round(3).astype("str")
    anova.loc[anova['P-val'] == "0.0", "P-val"] = "< .001"
    anova['P-val'] = anova['P-val'].replace({"0.": "."})
    anova['p_eta_2'] = anova['p_eta_2'].round(3).astype("str")

    anova = anova.reset_index(names=['factor'])
    anova = anova[["factor", "F-stat", "df", "P-val", "p_eta_2"]].reset_index()
    anova = anova.drop(columns="index")
    anova.to_csv(os.path.join(save_path, f'lmms_gaze_{dv}_test.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

    plt.tight_layout()


# Test, Relationship SPAI, ROI
def plot_gaze_test_roi(df, save_path, dv="Gaze Proportion", SA_score="SPAI"):
    # df = df_gaze
    if dv == "Gaze Proportion":
        y_label = "Proportional Dwell Time on Virtual Agent"
    elif dv == "Switches":
        y_label = "Shifts of Visual Attention Towards Virtual Agent"
    df = df.sort_values(by=SA_score)

    max = round(df[dv].max(), 2) * 1.1

    conditions = ["friendly", "unfriendly"]
    # titles = ["Spontaneous Fixations on\nFriendly Agent", "Spontaneous Fixations on\nUnfriendly Agent"]
    fig, axes = plt.subplots(nrows=1, ncols=len(conditions), figsize=(3.5 * len(conditions), 5))
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = "friendly"
        rois = ["body", "head"]
        labels = ["Body", "Head"]
        df_condition = df.loc[df['Condition'] == condition]
        df_condition = df_condition.loc[df_condition['ROI'] != "other"].reset_index(drop=True)

        colors = ['#183DB2', '#7FCEBC']

        for idx_roi, roi in enumerate(rois):
            # idx_roi = 1
            # roi = rois[idx_roi]
            df_roi = df_condition.loc[df_condition['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)

            x = df_roi[SA_score].to_numpy()
            y = df_roi[dv].to_numpy()
            linreg = linregress(x, y)
            all_x = df[SA_score].to_numpy()
            all_y = df[dv].to_numpy()
            all_y_est = linreg.slope * all_x + linreg.intercept
            all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
                1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

            # Plot regression line
            axes[idx_condition].plot(all_x, all_y_est, '-', color=colors[idx_roi])
            axes[idx_condition].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_roi])

            if "Wave1" in save_path and dv == "Switches":
                p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
                if idx_roi == 0:
                    axes[idx_condition].text(df[SA_score].min() + 0.01 * np.max(x), 0.95 * max,
                                         r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                         color=colors[idx_roi])
                else:
                    axes[idx_condition].text(df[SA_score].min() + 0.01 * np.max(x), 0.91 * max,
                                         r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                         color=colors[idx_roi])

            # Plot raw data points
            axes[idx_condition].plot(x, y, 'o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.3, label=roi.capitalize())

        axes[idx_condition].legend(loc="upper right")
        axes[idx_condition].set_title(f"{condition.capitalize()} Agent", fontweight='bold')  # (N = {len(df_condition['VP'].unique())})
        axes[idx_condition].set_ylim([0, max])
        axes[idx_condition].set_xlabel(SA_score)
        axes[idx_condition].grid(color='lightgrey', linestyle='-', linewidth=0.3)
        axes[idx_condition].set_ylabel(y_label)

    plt.tight_layout()

    df = df.rename(columns={dv: "gaze"})
    df = df.loc[df["Condition"].str.contains("friendly")]
    df[SA_score] = (df[SA_score] - df[SA_score].mean()) / df[SA_score].std()

    formula = f"gaze ~ Condition + {SA_score} + ROI +" \
              f"Condition:{SA_score} + Condition:ROI + {SA_score}:ROI +" \
              f"Condition:{SA_score}:ROI + (1 | VP)"

    model = pymer4.models.Lmer(formula, data=df)
    model.fit(factors={"Condition": ["friendly", "unfriendly"], "ROI": ["body", "head"]}, summarize=False)
    anova = model.anova(force_orthogonal=True)
    sum_sq_error = (sum(i * i for i in model.residuals))
    anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
    print(f"ANOVA: Gaze Test (Condition, ROI and {SA_score})")
    print(f"Condition Main Effect, F({round(anova.loc['Condition', 'NumDF'].item(), 1)}, {round(anova.loc['Condition', 'DenomDF'].item(), 1)})={round(anova.loc['Condition', 'F-stat'].item(), 2)}, p={round(anova.loc['Condition', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['Condition', 'p_eta_2'].item(), 2)}")
    print(f"{SA_score} Main Effect, F({round(anova.loc[SA_score, 'NumDF'].item(), 1)}, {round(anova.loc[SA_score, 'DenomDF'].item(), 1)})={round(anova.loc[SA_score, 'F-stat'].item(), 2)}, p={round(anova.loc[SA_score, 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[SA_score, 'p_eta_2'].item(), 2)}")
    print(f"ROI Main Effect, F({round(anova.loc['ROI', 'NumDF'].item(), 1)}, {round(anova.loc['ROI', 'DenomDF'].item(), 1)})={round(anova.loc['ROI', 'F-stat'].item(), 2)}, p={round(anova.loc['ROI', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc['ROI', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score}, F({round(anova.loc[f'Condition:{SA_score}', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x ROI, F({round(anova.loc[f'Condition:ROI', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:ROI', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:ROI', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:ROI', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:ROI', 'p_eta_2'].item(), 2)}")
    print(f"Interaction {SA_score} x ROI, F({round(anova.loc[f'{SA_score}:ROI', 'NumDF'].item(), 1)}, {round(anova.loc[f'{SA_score}:ROI', 'DenomDF'].item(), 1)})={round(anova.loc[f'{SA_score}:ROI', 'F-stat'].item(), 2)}, p={round(anova.loc[f'{SA_score}:ROI', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'{SA_score}:ROI', 'p_eta_2'].item(), 2)}")
    print(f"Interaction Condition x {SA_score} x ROI, F({round(anova.loc[f'Condition:{SA_score}:ROI', 'NumDF'].item(), 1)}, {round(anova.loc[f'Condition:{SA_score}:ROI', 'DenomDF'].item(), 1)})={round(anova.loc[f'Condition:{SA_score}:ROI', 'F-stat'].item(), 2)}, p={round(anova.loc[f'Condition:{SA_score}:ROI', 'P-val'].item(), 3)}, p_eta_2={round(anova.loc[f'Condition:{SA_score}:ROI', 'p_eta_2'].item(), 2)}")
    estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="ROI", p_adjust="holm")

    anova['NumDF'] = anova['NumDF'].round().astype("str")
    anova['DenomDF'] = anova['DenomDF'].round(2).astype("str")
    anova["df"] = anova['NumDF'].str.cat(anova['DenomDF'], sep=', ')
    anova['F-stat'] = anova['F-stat'].round(2).astype("str")
    anova['P-val'] = anova['P-val'].round(3).astype("str")
    anova.loc[anova['P-val'] == "0.0", "P-val"] = "< .001"
    anova['P-val'] = anova['P-val'].replace({"0.": "."})
    anova['p_eta_2'] = anova['p_eta_2'].round(3).astype("str")

    anova = anova.reset_index(names=['factor'])
    anova = anova[["factor", "F-stat", "df", "P-val", "p_eta_2"]].reset_index()
    anova = anova.drop(columns="index")
    anova.to_csv(os.path.join(save_path, f'lmms_gaze_{dv}_test_roi.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')


# Test, Relationship SPAI
def plot_gaze_test_sad(df, dv="Gaze Proportion", SA_score="SPAI"):
    # df = df_gaze
    if dv == "Gaze Proportion":
        y_label = "Proportional Dwell Time on Virtual Agent"
    elif dv == "Switches":
        y_label = "Shifts of Visual Attention Towards Virtual Agent"

    df = df.sort_values(by=SA_score)

    conditions = ["friendly", "unfriendly"]
    titles = ["Friendly Agent", "Unfriendly Agent"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    red = '#E2001A'
    green = '#B1C800'
    colors = [green, red]
    for idx_condition, condition in enumerate(conditions):
        # idx_condition = 0
        # condition = conditions[idx_condition]
        df_cond = df.loc[df['Condition'] == condition].reset_index(drop=True)

        x = df_cond[SA_score].to_numpy()
        y = df_cond[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df[SA_score].to_numpy()
        all_y = df_cond[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        ax.plot(all_x, all_y_est, '-', color=colors[idx_condition])
        ax.fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_condition == 0:
            ax.text(df[SA_score].min() + 0.01 * np.max(x), 0.95 * (df[dv].max() - df[dv].min()) + df[dv].min(),
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_condition])
        else:
            ax.text(df[SA_score].min() + 0.01 * np.max(x), 0.91 * (df[dv].max() - df[dv].min()) + df[dv].min(),
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
    ax.set_ylabel(y_label)
    # ax.set_title(f"Time Spent Close to Virtual Agents", fontweight='bold')
    ax.legend(loc="upper right")
    plt.tight_layout()


# Difference
def plot_diff_gaze(df, SA_score="SPAI"):
    df_test = df.loc[df["Phase"].str.contains("Test") & ~(df["Phase"].str.contains("Clicked"))]
    df_spai = df_test[["VP", SA_score]].drop_duplicates(subset="VP")
    df_diff = df_test.groupby(["VP", "Person", "Condition"]).sum(numeric_only=True).reset_index()
    df_diff = df_diff.pivot(index='VP', columns='Condition', values='Gaze Proportion').reset_index()
    df_diff["difference"] = df_diff["unfriendly"] - df_diff["friendly"]

    df_diff = df_diff[["VP", "difference"]].merge(df_spai, on="VP")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
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

    ax.set_title(f"Avoidance vs. Hypervigilance (N = {len(df_diff['VP'].unique())})", fontweight='bold')
    # ax.set_ylim([0, max])
    ax.set_xlabel(SA_score)
    if "SPAI" in SA_score:
        ax.set_xticks(range(0, 6))
    elif "SIAS" in SA_score:
        ax.set_xticks(range(5, 65, 5))
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.axhline(0, linewidth=0.8, color="k", linestyle="dashed")
    ax.set_ylabel("Difference Gaze Proportion: Unfriendly-Friendly")
    ax.legend(
        [Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7),
         Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7)],
        ["Hypervigilance", "Avoidance"], loc="upper right")

    plt.tight_layout()

    df_diff = df_diff[["VP", "difference"]]
    df_diff = df_diff.rename(columns={"difference": "gaze_diff"})
    df_diff = df_diff.sort_values(by="VP").reset_index(drop=True)
    return df_diff


if __name__ == '__main__':
    wave = 2
    dir_path = os.getcwd()
    filepath = os.path.join(dir_path, f'Data-Wave{wave}')

    save_path = os.path.join(dir_path, f'Plots-Wave{wave}', 'Gaze')
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
    vps = [vp for vp in vps if not vp in problematic_subjects]

    plot_et_validation(vps, filepath)
    plt.savefig(os.path.join(save_path, f"et_calibration.png"), dpi=300)
    plt.close()

    SA_score = "SPAI"
    df_gaze = pd.read_csv(os.path.join(filepath, 'gaze.csv'), decimal='.', sep=';')
    dvs = ["Gaze Proportion", "Switches"]
    dv = dvs[0]
    plot_gaze_acq(df_gaze, dv="Gaze Proportion", SA_score="SPAI")
    plt.savefig(os.path.join(save_path, f"gaze_acq-{dv}_{SA_score}.png"), dpi=300)
    plt.close()

    if wave == 1:
        plot_gaze_click(df_gaze, dv="Gaze Proportion", SA_score="SPAI")
        plt.savefig(os.path.join(save_path, f"gaze_click-{dv}_{SA_score}.png"), dpi=300)
        plt.close()

    plot_gaze_test(df_gaze, dv="Gaze Proportion", SA_score="SPAI")
    plt.savefig(os.path.join(save_path, f"gaze_test-{dv}.png"), dpi=300)
    plt.close()

    plot_gaze_test_roi(df_gaze, dv="Gaze Proportion", SA_score="SPAI")
    plt.savefig(os.path.join(save_path, f"gaze_test-{dv}_{SA_score}.png"), dpi=300)
    plt.close()

    df_diff_gaze = plot_diff_gaze(df_gaze, SA_score="SPAI")
    plt.savefig(os.path.join(save_path, f"gaze_test-{dv}-diff_{SA_score}.png"), dpi=300)
    plt.close()
