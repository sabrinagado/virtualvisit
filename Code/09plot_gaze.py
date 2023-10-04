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
from scipy.stats import linregress
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
import pymer4

from Code.toolbox import utils


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

df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';')
SA_score = "SPAI"
dvs = ["Gaze Proportion", "Switches"]
dv = dvs[0]
y_labels = ["Gaze Proportion on Person", "Fixation Switches Towards Virtual Human"]
y_label = y_labels[0]

red = '#E2001A'
green = '#B1C800'
blue = '#1F82C0'


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
plt.savefig(os.path.join(save_path, f"et_calibration.png"), dpi=300)
plt.close()


# Acquisition: Interactions, Relationship SPAI
phases = ["FriendlyInteraction", "UnfriendlyInteraction", "NeutralInteraction"]
df_acq = df_gaze.loc[df_gaze["Phase"].isin(phases)]
df_acq["Phase_corr"] = [string[0].lower() for string in df_acq["Phase"].str.split("Interaction")]
df_acq = df_acq.loc[df_acq["Phase_corr"] == df_acq["Condition"]]
df_acq = df_acq.drop(columns="Phase_corr")
max = round(df_acq[dv].max(), 2) * 1.1

fig, axes = plt.subplots(nrows=1, ncols=len(phases), figsize=(3 * len(phases), 6))
titles = ["Friendly Interaction", "Unfriendly Interaction", "Neutral Interaction"]
df_acq = df_acq.sort_values(by=SA_score)
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

        x = df_roi[SA_score].to_numpy()
        y = df_roi[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_acq[SA_score].to_numpy()
        all_y = df_acq[dv].to_numpy()
        all_y_est = linreg.slope * all_x + linreg.intercept
        all_y_err = np.sqrt(np.sum((all_y - np.mean(all_y)) ** 2) / (len(all_y) - 2)) * np.sqrt(
            1 / len(all_x) + (all_x - np.mean(all_x)) ** 2 / np.sum((all_x - np.mean(all_x)) ** 2))

        # Plot regression line
        axes[idx_phase].plot(all_x, all_y_est, '-', color=colors[idx_roi])
        axes[idx_phase].fill_between(all_x, all_y_est + all_y_err, all_y_est - all_y_err, alpha=0.2, color=colors[idx_roi])

        p_sign = "***" if linreg.pvalue < 0.001 else "**" if linreg.pvalue < 0.01 else "*" if linreg.pvalue < 0.05 else ""
        if idx_roi == 0:
            axes[idx_phase].text(df_acq[SA_score].min() + 0.01 * np.max(x), 0.95 * max,
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_roi])
        else:
            axes[idx_phase].text(df_acq[SA_score].min() + 0.01 * np.max(x), 0.91 * max,
                    r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                    color=colors[idx_roi])

        # Plot raw data points
        axes[idx_phase].plot(x, y, 'o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.6,
                label=roi.capitalize())

    axes[idx_phase].legend(loc="upper right")
    axes[idx_phase].set_title(f"{titles[idx_phase]} (N = {len(df_phase['VP'].unique())})", fontweight='bold')
    axes[idx_phase].set_ylim([0, max])
    axes[idx_phase].set_xlabel(SA_score)
    axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel(y_label)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"gaze_acq-{dv}_{SA_score}.png"), dpi=300)
plt.close()

df_acq = df_acq.rename(columns={dv: "gaze"})
df_acq[SA_score] = (df_acq[SA_score] - df_acq[SA_score].mean()) / df_acq[SA_score].std()

# df_acq = df_acq.loc[~(df_acq["Condition"].str.contains("neutral"))]
formula = f"gaze ~ Condition + {SA_score} + ROI +" \
          f"Condition:{SA_score} + Condition:ROI + {SA_score}:ROI +" \
          f"Condition:{SA_score}:ROI + (1 | VP)"

model = pymer4.models.Lmer(formula, data=df_acq)
model.fit(factors={"Condition": ["friendly", "unfriendly", "neutral"], "ROI": ["body", "head"]}, summarize=False)
anova = model.anova(force_orthogonal=True)
sum_sq_error = (sum(i * i for i in model.residuals))
anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="ROI", p_adjust="holm")


# Clicks, Relationship SPAI
phases = ["Test_FriendlyWasClicked", "Test_UnfriendlyWasClicked"]
df_click = df_gaze.loc[df_gaze["Phase"].isin(phases)]
df_click["Phase_corr"] = [string[0].split("Test_")[1].lower() for string in df_click["Phase"].str.split("WasClicked")]
df_click = df_click.loc[df_click["Phase_corr"] == df_click["Condition"]]
df_click = df_click.drop(columns="Phase_corr")
df_spai = df_click[["VP", SA_score]].drop_duplicates(subset="VP")
df_grouped = df_click.groupby(["VP", "Phase", "Person", "Condition", "ROI"]).mean(numeric_only=True).reset_index()
max = round(df_grouped[dv].max(), 2) * 1.1

fig, axes = plt.subplots(nrows=1, ncols=len(phases), figsize=(8, 6))
titles = ["Fixations after Click on\nFriendly Person", "Fixations after Click on\nUnfriendly Person"]
df_grouped = df_grouped.sort_values(by=SA_score)
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

        x = df_roi[SA_score].to_numpy()
        y = df_roi[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_grouped[SA_score].to_numpy()
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
            axes[idx_phase].text(df_click[SA_score].min() + 0.01 * np.max(x), 0.95 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])
        else:
            axes[idx_phase].text(df_click[SA_score].min() + 0.01 * np.max(x), 0.91 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])

        # Plot raw data points
        axes[idx_phase].plot(x, y, 'o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.6,
                             label=roi.capitalize())

    axes[idx_phase].legend(loc="upper right")
    axes[idx_phase].set_title(f"{titles[idx_phase]} (N = {len(df_phase['VP'].unique())})", fontweight='bold')
    axes[idx_phase].set_ylim([0, max])
    axes[idx_phase].set_xlabel(SA_score)
    axes[idx_phase].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel(y_label)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"gaze_click-{dv}_{SA_score}.png"), dpi=300)
plt.close()

df_click = df_click.rename(columns={dv: "gaze"})
df_click[SA_score] = (df_click[SA_score] - df_click[SA_score].mean()) / df_click[SA_score].std()

formula = f"gaze ~ Condition + {SA_score} + ROI +" \
          f"Condition:{SA_score} + Condition:ROI + {SA_score}:ROI +" \
          f"Condition:{SA_score}:ROI + (1 | VP)"

model = pymer4.models.Lmer(formula, data=df_click)
model.fit(factors={"Condition": ["friendly", "unfriendly"], "ROI": ["body", "head"]}, summarize=False)
anova = model.anova(force_orthogonal=True)
sum_sq_error = (sum(i * i for i in model.residuals))
anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="ROI", p_adjust="holm")

# Test: Rooms
meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

df_test = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
max = round(df_test[dv].max(), 2) * 1.1
conditions = ["friendly", "unfriendly"]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
boxWidth = 1 / (len(conditions) + 1)
pos = [0 + x * boxWidth for x in np.arange(1, len(conditions) + 1)]

for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = conditions[idx_condition]
    labels = ["Friendly", "Unfriendly"]
    df_cond = df_test.loc[df_test['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.loc[df_cond['ROI'] != "other"].reset_index(drop=True)

    colors = [green, red]

    # Plot raw data points
    for i in range(len(df_cond)):
        # i = 0
        x = random.uniform(pos[idx_condition] - (0.25 * boxWidth), pos[idx_condition] + (0.25 * boxWidth))
        y = df_cond.loc[i, dv].item()
        ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_condition], mec=colors[idx_condition], alpha=0.3)

    # Plot boxplots
    whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
    capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_condition])
    boxprops = dict(color=colors[idx_condition])

    fwr_correction = False
    alpha = (1 - (0.05 / 2)) if fwr_correction else (1 - (0.05))
    bootstrapping_dict = utils.bootstrapping(df_cond.loc[:, dv].values,
                                       numb_iterations=5000,
                                       alpha=alpha,
                                       as_dict=True,
                                       func='mean')

    ax.boxplot([df_cond.loc[:, dv].values],
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

ax.set_xticklabels(labels)
ax.set_title(f"Spontaneous Fixations (N = {len(df_cond['VP'].unique())})", fontweight='bold')
ax.set_ylim([0, max])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_ylabel(y_label)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"gaze_test-{dv}.png"), dpi=300)
plt.close()

# Test, Relationship SPAI
df_test = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
max = round(df_test[dv].max(), 2) * 1.1

conditions = ["friendly", "unfriendly"]
titles = ["Spontaneous Fixations on\nFriendly Person", "Spontaneous Fixations on\nUnfriendly Person"]
fig, axes = plt.subplots(nrows=1, ncols=len(conditions), figsize=(8, 6))
df_test = df_test.sort_values(by=SA_score)
for idx_condition, condition in enumerate(conditions):
    # idx_condition = 0
    # condition = "FriendlyInteraction"
    rois = ["body", "head"]
    labels = ["Body", "Head"]
    df_condition = df_test.loc[df_gaze['Condition'] == condition]
    df_condition = df_condition.loc[df_condition['ROI'] != "other"].reset_index(drop=True)

    colors = ['#183DB2', '#7FCEBC']

    for idx_roi, roi in enumerate(rois):
        # idx_roi = 0
        # roi = rois[idx_roi]

        df_roi = df_condition.loc[df_condition['ROI'] == roi].dropna(subset=dv).reset_index(drop=True)

        x = df_roi[SA_score].to_numpy()
        y = df_roi[dv].to_numpy()
        linreg = linregress(x, y)
        all_x = df_test[SA_score].to_numpy()
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
            axes[idx_condition].text(df_test[SA_score].min() + 0.01 * np.max(x), 0.95 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])
        else:
            axes[idx_condition].text(df_test[SA_score].min() + 0.01 * np.max(x), 0.91 * max,
                                 r"$\it{r}$ = " + f"{round(linreg.rvalue, 2)}{p_sign}",
                                 color=colors[idx_roi])

        # Plot raw data points
        axes[idx_condition].plot(x, y, 'o', ms=5, mfc=colors[idx_roi], mec=colors[idx_roi], alpha=0.6,
                             label=roi.capitalize())

    axes[idx_condition].legend(loc="upper right")
    axes[idx_condition].set_title(f"{titles[idx_condition]} (N = {len(df_condition['VP'].unique())})", fontweight='bold')
    axes[idx_condition].set_ylim([0, max])
    axes[idx_condition].set_xlabel(SA_score)
    axes[idx_condition].grid(color='lightgrey', linestyle='-', linewidth=0.3)
axes[0].set_ylabel(y_label)

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"gaze_test-{dv}_{SA_score}.png"), dpi=300)
plt.close()

df_test = df_test.rename(columns={dv: "gaze"})
df_test = df_test.loc[df_test["Condition"].str.contains("friendly")]
df_test[SA_score] = (df_test[SA_score] - df_test[SA_score].mean()) / df_test[SA_score].std()

formula = f"gaze ~ Condition + {SA_score} + ROI +" \
          f"Condition:{SA_score} + Condition:ROI + {SA_score}:ROI +" \
          f"Condition:{SA_score}:ROI + (1 | VP)"

model = pymer4.models.Lmer(formula, data=df_test)
model.fit(factors={"Condition": ["friendly", "unfriendly"], "ROI": ["body", "head"]}, summarize=False)
anova = model.anova(force_orthogonal=True)
sum_sq_error = (sum(i * i for i in model.residuals))
anova["p_eta_2"] = anova["SS"] / (anova["SS"] + sum_sq_error)
estimates, contrasts = model.post_hoc(marginal_vars="Condition", grouping_vars="ROI", p_adjust="holm")


# Difference
df_test = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
df_spai = df_test[["VP", SA_score]].drop_duplicates(subset="VP")
df_diff = df_test.groupby(["VP", "Person", "Condition"]).sum(numeric_only=True).reset_index()
df_diff = df_diff.pivot(index='VP', columns='Condition', values='Gaze Proportion').reset_index()
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
plt.savefig(os.path.join(save_path, f"gaze_test-{dv}-diff_{SA_score}.png"), dpi=300)
plt.close()

df_diff = df_diff[["VP", "difference"]]
df_diff = df_diff.rename(columns={"difference": "gaze_diff"})
df_diff = df_diff.sort_values(by="VP").reset_index(drop=True)
try:
    df_aa = pd.read_csv(os.path.join(dir_path, 'Data', 'aa_tendency.csv'), decimal='.', sep=';')
    if "gaze_diff" in df_aa.columns:
        df_aa.update(df_diff)
    else:
        df_aa = df_aa.merge(df_diff, on="VP")
    df_aa.to_csv(os.path.join(dir_path, 'Data', 'aa_tendency.csv'), decimal='.', sep=';', index=False)
except:
    df_diff.to_csv(os.path.join(dir_path, 'Data', 'aa_tendency.csv'), decimal='.', sep=';', index=False)

x = df_aa["distance_diff"].to_numpy()
y = df_aa["time_diff"].to_numpy()
linreg = linregress(x, y)
print(f"Interpersonal Difference x Time: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

x = df_aa["distance_diff"].to_numpy()
y = df_aa["gaze_diff"].to_numpy()
linreg = linregress(x, y)
print(f"Interpersonal Difference x Gaze: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")

x = df_aa["time_diff"].to_numpy()
y = df_aa["gaze_diff"].to_numpy()
linreg = linregress(x, y)
print(f"Gaze x Time: r = {round(linreg.rvalue, 2)}, p = {round(linreg.pvalue, 3)}")
