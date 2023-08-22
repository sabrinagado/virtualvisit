# =============================================================================
# Eye_tracking and Gaze: Proportion of Gaze on Social vs. Non-Social Stimuli
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
save_path = os.path.join(dir_path, 'Plots', 'Gaze')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)


dir_path = os.getcwd()
start = 1
end = 64
vps = np.arange(start, end + 1)

problematic_subjects = [1, 3, 12, 15, 19, 20, 23, 24, 31, 33, 41, 42, 45, 46, 47, 53]
vps = [vp for vp in vps if not vp in problematic_subjects]

# Visualize ET Validation
points_start = pd.DataFrame(columns=["x", "y"])
points_end = pd.DataFrame(columns=["x", "y"])
for vp in vps:
    # vp = vps[1]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    try:
        files = [item for item in os.listdir(os.path.join(dir_path, 'Data', 'VP_' + vp)) if (item.endswith(".csv"))]
        file = [file for file in files if "etcalibration" in file][0]
        df_cal = pd.read_csv(os.path.join(dir_path, 'Data', 'VP_' + vp, file), sep=';', decimal='.')
    except:
        print("no gaze file")
        continue

    for idx_row, row in df_cal.loc[df_cal["time"] == "Start"].iterrows():
        # idx_row = 0
        # row = df_cal.iloc[idx_row, :]
        position = row["position"]
        x = float(position.split("=")[1].split(",")[0]) + row["x_divergence"]
        y = float(position.split("=")[2]) + row["y_divergence"]
        points_start = pd.concat([points_start, pd.DataFrame({"x": [x], "y": [y]})])

    for idx_row, row in df_cal.loc[df_cal["time"] == "End"].iterrows():
        # idx_row = 0
        # row = df_cal.iloc[idx_row, :]
        position = row["position"]
        x = float(position.split("=")[1].split(",")[0]) + row["x_divergence"]
        y = float(position.split("=")[2]) + row["y_divergence"]
        points_end = pd.concat([points_end, pd.DataFrame({"x": [x], "y": [y]})])

points_cal = pd.DataFrame(columns=["x", "y"])
for idx_row, row in df_cal.loc[df_cal["time"] == "Start"].iterrows():
    # idx_row = 0
    # row = df_cal.iloc[idx_row, :]
    position = row["position"]
    x = float(position.split("=")[1].split(",")[0])
    y = float(position.split("=")[2])
    points_cal = pd.concat([points_cal, pd.DataFrame({"x": [x], "y": [y]})])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for idx_points, (points, title) in enumerate(zip([points_start, points_end], ["Start", "End"])):
    # idx_points = 0
    # points = points_start
    axes[idx_points].scatter(points["x"], points["y"], marker='+', s=20, c="k", linewidths=0.8)
    axes[idx_points].scatter(points_cal["x"], points_cal["y"], marker='+', s=100, c="red", linewidths=0.8)
    axes[idx_points].set_title(title)
    axes[idx_points].set_ylim(points_cal["y"].min()-20, points_cal["y"].max()+20)
    axes[idx_points].set_xlim(points_cal["x"].min() - 20, points_cal["x"].max() + 20)

plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"et_calibration{end}"), dpi=300)
plt.close()

# Gaze on ROIs of Virtual Humans
dv = "Gaze Proportion"
y_label = "Gaze Proportion on Person"

red = '#E2001A'
green = '#B1C800'
blue = '#1F82C0'
colors = [green, red, blue]

df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';')

max = round(df_gaze.loc[df_gaze["ROI"] != "other", dv].max(), 2) + 0.1

meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

# Acquisition: Interactions
phases = ["FriendlyInteraction", "UnfriendlyInteraction", "NeutralInteraction"]
df_acq = df_gaze.loc[df_gaze["Phase"].isin(phases)]
df_acq["Phase_corr"] = [string[0].lower() for string in df_acq["Phase"].str.split("Interaction")]
df_acq = df_acq.loc[df_acq["Phase_corr"] == df_acq["Condition"]]
df_acq = df_acq.drop(columns="Phase_corr")
# fig, axes = plt.subplots(nrows=1, ncols=len(phases), figsize=(3*len(phases), 6))
# titles = ["Friendly Interaction", "Unfriendly Interaction", "Neutral Interaction"]
# for idx_phase, phase in enumerate(phases):
#     # idx_phase = 1
#     # phase = "UnfriendlyInteraction"
#     rois = ["body", "head"]
#     labels = ["Body", "Head"]
#     y_label = y_labels[idx_dv]
#     df_phase = df_acq.loc[df_gaze['Phase'] == phase]
#     df_phase = df_phase.loc[df_phase['ROI'] != "other"].reset_index(drop=True)
#     data_phase = df_phase[dv].to_list()
#
#     boxWidth = 1 / (len(rois) + 1)
#     pos = [0 + x * boxWidth for x in np.arange(1, len(rois) + 1)]
#
#     colors = ['#183DB2', '#7FCEBC']
#
#     for idx_roi, roi in enumerate(rois):
#         # idx_roi = 0
#         # roi = rois[idx_roi]
#
#         # Plot raw data points
#         df_roi = df_phase.loc[df_phase['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)
#         for i in range(len(df_roi)):
#             # i = 0
#             x = random.uniform(pos[idx_roi] - (0.25 * boxWidth), pos[idx_roi] + (0.25 * boxWidth))
#             y = df_roi.loc[i, dv].item()
#             axes[idx_phase].plot(x, y, marker='o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.3)
#
#         # Plot boxplots
#         whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
#         capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
#         boxprops = dict(color=colors[idx_roi])
#
#         fwr_correction = False
#         alpha = (1 - (0.05 / 2)) if fwr_correction else (1 - (0.05))
#         bootstrapping_dict = bootstrapping(df_roi.loc[:, dv].values,
#                                            numb_iterations=5000,
#                                            alpha=alpha,
#                                            as_dict=True,
#                                            func='mean')
#
#         axes[idx_phase].boxplot([df_roi.loc[:, dv].values],
#                                 notch=True,  # bootstrap=5000,
#                                 medianprops=medianlineprops,
#                                 meanline=True,
#                                 showmeans=True,
#                                 meanprops=meanlineprops,
#                                 showfliers=False, flierprops=fliermarkerprops,
#                                 whiskerprops=whiskerprops,
#                                 capprops=capprops,
#                                 boxprops=boxprops,
#                                 conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
#                                 whis=[2.5, 97.5],
#                                 positions=[pos[idx_roi]],
#                                 widths=0.8 * boxWidth)
#
#     axes[idx_phase].set_xticklabels(labels)
#     axes[idx_phase].set_title(f"{titles[idx_phase]}", fontweight='bold')
#     axes[idx_phase].set_ylim([0, max])
#     axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
# axes[0].set_ylabel(y_label)
#
# plt.tight_layout()
# for end in (['.png']):  # '.pdf',
#     plt.savefig(os.path.join(save_path, f"gaze_acq-{dv}{end}"), dpi=300)
# plt.close()

# Acquisition: Interactions, Relationship SPAI
phases = ["FriendlyInteraction", "UnfriendlyInteraction", "NeutralInteraction"]
df_acq = df_gaze.loc[df_gaze["Phase"].isin(phases)]
df_acq["Phase_corr"] = [string[0].lower() for string in df_acq["Phase"].str.split("Interaction")]
df_acq = df_acq.loc[df_acq["Phase_corr"] == df_acq["Condition"]]
df_acq = df_acq.drop(columns="Phase_corr")
max = round(df_acq[dv].max(), 2) * 1.1

fig, axes = plt.subplots(nrows=1, ncols=len(phases), figsize=(3 * len(phases), 6))
titles = ["Friendly Interaction", "Unfriendly Interaction", "Neutral Interaction"]
df_acq = df_acq.sort_values(by="SPAI")
for idx_phase, phase in enumerate(phases):
    # idx_phase = 0
    # phase = "FriendlyInteraction"
    rois = ["body", "head"]
    labels = ["Body", "Head"]
    df_phase = df_acq.loc[df_gaze['Phase'] == phase]
    df_phase = df_phase.loc[df_phase['ROI'] != "other"].reset_index(drop=True)

    colors = ['#183DB2', '#7FCEBC']

    for idx_roi, roi in enumerate(rois):
        # idx_roi = 0
        # roi = rois[idx_roi]

        df_roi = df_phase.loc[df_phase['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)

        x = df_roi["SPAI"].to_numpy()
        y = df_roi[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_acq["SPAI"].to_numpy()
        all_y = df_acq[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        axes[idx_phase].plot(all_x, all_y_est, '-', color=colors[idx_roi])
        axes[idx_phase].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_roi])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_roi == 0:
            axes[idx_phase].text(df_acq["SPAI"].min() + 0.01 * np.max(x), 0.95 * max,
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_roi])
        else:
            axes[idx_phase].text(df_acq["SPAI"].min() + 0.01 * np.max(x), 0.91 * max,
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_roi])

        # Plot raw data points
        axes[idx_phase].plot(x, y, 'o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.6,
                label=roi.capitalize())

    axes[idx_phase].legend()
    axes[idx_phase].set_title(f"{titles[idx_phase]} (N = {len(df_phase['VP'].unique())})", fontweight='bold')
    axes[idx_phase].set_ylim([0, max])
    axes[idx_phase].set_xlabel("SPAI")
    axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel(y_label)

plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"gaze_acq-{dv}_SPAI{end}"), dpi=300)
plt.close()

df_acq = df_acq.rename(columns={dv: "gaze"})
df_acq["SPAI"] = (df_acq["SPAI"] - df_acq["SPAI"].mean()) / df_acq["SPAI"].std()

# df_acq = df_acq.loc[~(df_acq["Condition"].str.contains("neutral"))]
formula = f"gaze ~ Condition + SPAI + ROI +" \
          f"Condition:SPAI + Condition:ROI + SPAI:ROI +" \
          f"Condition:SPAI:ROI + (1 | VP)"

lm = smf.ols(formula, data=df_acq).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

contrasts = sp.posthoc_ttest(df_acq, val_col='gaze', group_col='Condition', p_adjust='holm')
df_acq.groupby("Condition")["gaze"].mean()

contrasts = sp.posthoc_ttest(df_acq, val_col='gaze', group_col='ROI', p_adjust='holm')
df_acq.groupby("ROI")["gaze"].mean()

# Clicks
# phases = ["Test_FriendlyWasClicked", "Test_NeutralWasClicked", "Test_UnfriendlyWasClicked"]
# df_click = df_gaze.loc[df_gaze["Phase"].isin(phases)]
# df_click["Phase_corr"] = [string[0].split("Test_")[1].lower() for string in df_click["Phase"].str.split("WasClicked")]
# df_click = df_click.loc[df_click["Phase_corr"] == df_click["Condition"]]
# df_click = df_click.drop(columns="Phase_corr")
# fig, axes = plt.subplots(nrows=1, ncols=len(phases), figsize=(3*len(phases), 6))
# titles = ["Clicked Friendly", "Clicked Neutral", "Clicked Unfriendly"]
# for idx_phase, phase in enumerate(phases):
#     # idx_phase = 1
#     # phase = "UnfriendlyInteraction"
#     rois = ["body", "head"]
#     labels = ["Body", "Head"]
#     y_label = y_labels[idx_dv]
#     df_phase = df_click.loc[df_gaze['Phase'] == phase]
#     df_phase = df_phase.loc[df_phase['ROI'] != "other"].reset_index(drop=True)
#
#     boxWidth = 1 / (len(rois) + 1)
#     pos = [0 + x * boxWidth for x in np.arange(1, len(rois) + 1)]
#
#     colors = ['#183DB2', '#7FCEBC']
#
#     for idx_roi, roi in enumerate(rois):
#         # idx_roi = 0
#         # roi = rois[idx_roi]
#
#         # Plot raw data points
#         df_roi = df_phase.loc[df_phase['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)
#         for i in range(len(df_roi)):
#             # i = 0
#             x = random.uniform(pos[idx_roi] - (0.25 * boxWidth), pos[idx_roi] + (0.25 * boxWidth))
#             y = df_roi.loc[i, dv].item()
#             axes[idx_phase].plot(x, y, marker='o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.3)
#
#         # Plot boxplots
#         whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
#         capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
#         boxprops = dict(color=colors[idx_roi])
#
#         fwr_correction = False
#         alpha = (1 - (0.05 / 2)) if fwr_correction else (1 - (0.05))
#         bootstrapping_dict = bootstrapping(df_roi.loc[:, dv].values,
#                                            numb_iterations=5000,
#                                            alpha=alpha,
#                                            as_dict=True,
#                                            func='mean')
#
#         axes[idx_phase].boxplot([df_roi.loc[:, dv].values],
#                                 notch=True,  # bootstrap=5000,
#                                 medianprops=medianlineprops,
#                                 meanline=True,
#                                 showmeans=True,
#                                 meanprops=meanlineprops,
#                                 showfliers=False, flierprops=fliermarkerprops,
#                                 whiskerprops=whiskerprops,
#                                 capprops=capprops,
#                                 boxprops=boxprops,
#                                 conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
#                                 whis=[2.5, 97.5],
#                                 positions=[pos[idx_roi]],
#                                 widths=0.8 * boxWidth)
#
#     axes[idx_phase].set_xticklabels(labels)
#     axes[idx_phase].set_title(f"{titles[idx_phase]}", fontweight='bold')
#     axes[idx_phase].set_ylim([0, max])
#     axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
# axes[0].set_ylabel(y_label)
#
# plt.tight_layout()
# for end in (['.png']):  # '.pdf',
#     plt.savefig(os.path.join(save_path, f"gaze_clicks-{dv}{end}"), dpi=300)
# plt.close()

# Clicks, Relationship SPAI
phases = ["Test_FriendlyWasClicked", "Test_UnfriendlyWasClicked"]
df_click = df_gaze.loc[df_gaze["Phase"].isin(phases)]
df_click["Phase_corr"] = [string[0].split("Test_")[1].lower() for string in df_click["Phase"].str.split("WasClicked")]
df_click = df_click.loc[df_click["Phase_corr"] == df_click["Condition"]]
df_click = df_click.drop(columns="Phase_corr")
df_spai = df_click[["VP", "SPAI"]].drop_duplicates(subset="VP")
df_grouped = df_click.groupby(["VP", "Phase", "Person", "Condition", "ROI"]).mean().reset_index()
max = round(df_grouped[dv].max(), 2) * 1.1

fig, axes = plt.subplots(nrows=1, ncols=len(phases), figsize=(3 * len(phases), 6))
titles = ["Clicked Friendly", "Clicked Unfriendly"]
df_grouped = df_grouped.sort_values(by="SPAI")
for idx_phase, phase in enumerate(phases):
    # idx_phase = 0
    # phase = "Test_FriendlyWasClicked"
    rois = ["body", "head"]
    labels = ["Body", "Head"]
    df_phase = df_grouped.loc[df_grouped['Phase'] == phase]
    df_phase = df_phase.loc[df_phase['ROI'] != "other"].reset_index(drop=True)

    colors = ['#183DB2', '#7FCEBC']

    for idx_roi, roi in enumerate(rois):
        # idx_roi = 0
        # roi = rois[idx_roi]

        df_roi = df_phase.loc[df_phase['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)

        x = df_roi["SPAI"].to_numpy()
        y = df_roi[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_grouped["SPAI"].to_numpy()
        all_y = df_grouped[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        axes[idx_phase].plot(all_x, all_y_est, '-', color=colors[idx_roi])
        axes[idx_phase].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2,
                                     color=colors[idx_roi])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_roi == 0:
            axes[idx_phase].text(df_click["SPAI"].min() + 0.01 * np.max(x), 0.95 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])
        else:
            axes[idx_phase].text(df_click["SPAI"].min() + 0.01 * np.max(x), 0.91 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])

        # Plot raw data points
        axes[idx_phase].plot(x, y, 'o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.6,
                             label=roi.capitalize())

    axes[idx_phase].legend(loc="upper right")
    axes[idx_phase].set_title(f"{titles[idx_phase]} (N = {len(df_phase['VP'].unique())})", fontweight='bold')
    axes[idx_phase].set_ylim([0, max])
    axes[idx_phase].set_xlabel("SPAI")
    axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel(y_label)

plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"gaze_click-{dv}_SPAI{end}"), dpi=300)
plt.close()

df_click = df_click.rename(columns={dv: "gaze"})
df_click["SPAI"] = (df_click["SPAI"] - df_click["SPAI"].mean()) / df_click["SPAI"].std()

formula = f"gaze ~ Condition + SPAI + ROI +" \
          f"Condition:SPAI + Condition:ROI + SPAI:ROI +" \
          f"Condition:SPAI:ROI + (1 | VP)"

lm = smf.ols(formula, data=df_click).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

# # Test: Rooms
# df_test = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
# conditions = ["friendly", "unfriendly"]
# fig, axes = plt.subplots(nrows=1, ncols=len(conditions), figsize=(3*len(conditions), 6))
# titles = ["Friendly Person", "Unfriendly Person"]
# for idx_condition, condition in enumerate(conditions):
#     # idx_condition = 0
#     # condition = conditions[idx_condition]
#     rois = ["body", "head"]
#     labels = ["Body", "Head"]
#     y_label = y_labels[idx_dv]
#     df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)
#     df_cond = df_cond.loc[df_cond['ROI'] != "other"].reset_index(drop=True)
#     data_phase = df_cond[dv].to_list()
#
#     boxWidth = 1 / (len(rois) + 1)
#     pos = [0 + x * boxWidth for x in np.arange(1, len(rois) + 1)]
#
#     colors = ['#183DB2', '#7FCEBC']
#
#     for idx_roi, roi in enumerate(rois):
#         # idx_roi = 0
#         # roi = rois[idx_roi]
#
#         # Plot raw data points
#         df_roi = df_cond.loc[df_cond['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)
#         for i in range(len(df_roi)):
#             # i = 0
#             x = random.uniform(pos[idx_roi] - (0.25 * boxWidth), pos[idx_roi] + (0.25 * boxWidth))
#             y = df_roi.loc[i, dv].item()
#             axes[idx_condition].plot(x, y, marker='o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.3)
#
#         # Plot boxplots
#         whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
#         capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_roi])
#         boxprops = dict(color=colors[idx_roi])
#
#         fwr_correction = False
#         alpha = (1 - (0.05 / 2)) if fwr_correction else (1 - (0.05))
#         bootstrapping_dict = bootstrapping(df_roi.loc[:, dv].values,
#                                            numb_iterations=5000,
#                                            alpha=alpha,
#                                            as_dict=True,
#                                            func='mean')
#
#         axes[idx_condition].boxplot([df_roi.loc[:, dv].values],
#                                 notch=True,  # bootstrap=5000,
#                                 medianprops=medianlineprops,
#                                 meanline=True,
#                                 showmeans=True,
#                                 meanprops=meanlineprops,
#                                 showfliers=False, flierprops=fliermarkerprops,
#                                 whiskerprops=whiskerprops,
#                                 capprops=capprops,
#                                 boxprops=boxprops,
#                                 conf_intervals=[[bootstrapping_dict['lower'], bootstrapping_dict['upper']]],
#                                 whis=[2.5, 97.5],
#                                 positions=[pos[idx_roi]],
#                                 widths=0.8 * boxWidth)
#
#     axes[idx_condition].set_xticklabels(labels)
#     axes[idx_condition].set_title(f"{titles[idx_condition]}", fontweight='bold')
#     axes[idx_condition].set_ylim([0, max])
#     axes[idx_condition].grid(color='lightgrey', linestyle='-', linewidth=0.3)
# axes[0].set_ylabel(y_label)
#
# plt.tight_layout()
# for end in (['.png']):  # '.pdf',
#     plt.savefig(os.path.join(save_path, f"gaze_test-{dv}{end}"), dpi=300)
# plt.close()

# Test, Relationship SPAI
df_test = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
max = round(df_test[dv].max(), 2) * 1.1

conditions = ["friendly", "unfriendly"]
titles = ["Spontaneous Fixations on Friendly Person", "Spontaneous Fixations on Unfriendly Person"]
fig, axes = plt.subplots(nrows=1, ncols=len(conditions), figsize=(15, 6))
df_test = df_test.sort_values(by="SPAI")
for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = "FriendlyInteraction"
    rois = ["body", "head"]
    labels = ["Body", "Head"]
    y_label = y_label
    df_condition = df_test.loc[df_gaze['Condition'] == condition]
    df_condition = df_condition.loc[df_condition['ROI'] != "other"].reset_index(drop=True)

    colors = ['#183DB2', '#7FCEBC']

    for idx_roi, roi in enumerate(rois):
        # idx_roi = 0
        # roi = rois[idx_roi]

        df_roi = df_condition.loc[df_condition['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)

        x = df_roi["SPAI"].to_numpy()
        y = df_roi[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_test["SPAI"].to_numpy()
        all_y = df_test[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        axes[idx_condition].plot(all_x, all_y_est, '-', color=colors[idx_roi])
        axes[idx_condition].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2,
                                     color=colors[idx_roi])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_roi == 0:
            axes[idx_condition].text(df_test["SPAI"].min() + 0.01 * np.max(x), 0.95 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])
        else:
            axes[idx_condition].text(df_test["SPAI"].min() + 0.01 * np.max(x), 0.91 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])

        # Plot raw data points
        axes[idx_condition].plot(x, y, 'o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.6,
                             label=roi.capitalize())

    axes[idx_condition].legend(loc="upper right")
    axes[idx_condition].set_title(f"{titles[idx_condition]} (N = {len(df_condition['VP'].unique())})", fontweight='bold')
    axes[idx_condition].set_ylim([0, max])
    axes[idx_condition].set_xlabel("SPAI")
    axes[idx_condition].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel(y_label)

plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"gaze_test-{dv}_SPAI{end}"), dpi=300)
plt.close()

df_test = df_test.rename(columns={dv: "gaze"})
df_test = df_test.loc[df_test["Condition"].str.contains("friendly")]
df_test["SPAI"] = (df_test["SPAI"] - df_test["SPAI"].mean()) / df_test["SPAI"].std()

formula = f"gaze ~ Condition + SPAI + ROI +" \
          f"Condition:SPAI + Condition:ROI + SPAI:ROI +" \
          f"Condition:SPAI:ROI + (1 | VP)"

lm = smf.ols(formula, data=df_test).fit()
anova = sm.stats.anova_lm(lm, typ=3)
sum_sq_error = anova.loc["Residual", "sum_sq"]
anova["p_eta_2"] = anova["sum_sq"] / (anova["sum_sq"] + sum_sq_error)

contrasts = sp.posthoc_ttest(df_test, val_col='gaze', group_col='ROI', p_adjust='holm')
df_test.groupby("ROI")["gaze"].mean()

contrasts = sp.posthoc_ttest(df_test, val_col='gaze', group_col='Condition', p_adjust='holm')
df_test.groupby("Condition")["gaze"].mean()

# Difference
df_test = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
df_spai = df_test[["VP", "SPAI"]].drop_duplicates(subset="VP")
df_diff = df_test.groupby(["VP", "Person", "Condition"]).sum().reset_index()
df_diff = df_diff.pivot(index='VP', columns='Condition', values='Gaze Proportion').reset_index()
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
ax.set_ylabel("Difference Gaze Proportion: Unfriendly-Friendly")
ax.legend(
    [Line2D([0], [0], color="white", marker='o', markeredgecolor="gold", markeredgewidth=1, markerfacecolor="gold", alpha=.7),
     Line2D([0], [0], color="white", marker='o', markeredgecolor="teal", markeredgewidth=1, markerfacecolor="teal", alpha=.7)],
    ["Hypervigilance", "Avoidance"], loc="best")

plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"gaze_test-{dv}-diff_SPAI{end}"), dpi=300)
plt.close()

# # Gaze on ROIs of Virtual Humans: Relationship with Social Anxiety
# df = df_gaze.loc[df_gaze["Phase"].str.contains("Test")]
# conditions = ["friendly", "unfriendly"]  # , "neutral"
# titles = ["Friendly Person", "Unfriendly Person"]  # , "Neutral Person"
# colors = ['#B1C800', '#E2001A']
# rois = ["body", "head"]
# fig, axes = plt.subplots(nrows=1, ncols=len(rois), figsize=(3.5*len(rois), 6))
# for idx_roi, roi in enumerate(rois):
#     # idx_roi = 0
#     # roi = rois[idx_roi]
#     df_roi = df.loc[df['ROI'] == roi].dropna(subset="Gaze Proportion").reset_index(drop=True)
#
#     boxWidth = 1
#     pos = [1]
#
#     for idx_condition, condition in enumerate(conditions):
#         # idx_condition = 0
#         # condition = conditions[idx_condition]
#         df_cond = df_roi.loc[df_gaze["Phase"].str.contains("Test")]
#         df_cond = df_cond.loc[df['Condition'] == condition].reset_index(drop=True)
#         x = df_cond["SPAI"].to_numpy()
#         y = df_cond["Gaze Proportion"].to_numpy()
#         linreg = linregress(x, y)
#         all_x = df_roi.sort_values(by=["SPAI"])["SPAI"].to_numpy()
#         all_y = df_cond["Gaze Proportion"].to_numpy()
#         all_y_est = linreg.slope * all_x + linreg.intercept
#         all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
#             1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))
#
#         # Plot regression line
#         axes[idx_roi].plot(all_x, all_y_est, '-', color=colors[idx_condition])
#         axes[idx_roi].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_condition])
#
#         # Plot raw data points
#         axes[idx_roi].plot(x, y, 'o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.6, label=titles[idx_condition])
#
#     axes[idx_roi].set_xlabel("SPAI")
#     axes[idx_roi].grid(color='lightgrey', linestyle='-', linewidth=0.3)
#     axes[idx_roi].set_ylabel(f"% Fixations on Person ({roi.capitalize()})")
#     axes[idx_roi].set_title(f"{roi.capitalize()}", fontweight='bold')
#     axes[idx_roi].set_ylim([0, 1])
# axes[1].legend()
# plt.tight_layout()
# for end in (['.png']):  # '.pdf',
#     plt.savefig(os.path.join(save_path, f"gaze_test_SA_{end}"), dpi=300)
# plt.close()

