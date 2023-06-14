# =============================================================================
# Scores
# source: SosciSurvey
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.power as pwr
import math

"""
Hypotheses:
- Larger interpersonal distance
- Reduced time spent in the same room
- Reduced visual attention towards the body or face of the negative virtual human
"""

colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']
dir_path = os.getcwd()
save_path = os.path.join(dir_path, 'Plots', 'Power Analysis')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)


# Factors for power analysis
alpha = 0.05
power = 0.8
pwr_analysis = pwr.TTestPower()

# =============================================================================
# Dwell Time
# =============================================================================
df_dwell = pd.read_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';')
df_dwell = df_dwell.loc[df_dwell["event"].str.contains("Habituation") | df_dwell["event"].str.contains("Test") & ~(df_dwell["event"].str.contains("Clicked"))]
conditions = ["friendly", "unfriendly"]
for condition in conditions:
    df_cond = df_dwell.loc[df_dwell['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.dropna(subset="duration")
    df_cond = df_cond.groupby(["VP", "event"]).sum().reset_index()
    df_hab = df_cond.loc[df_cond['event'].str.contains("Habituation")]
    df_hab = df_hab[["VP", "duration"]]
    df_hab = df_hab.rename(columns={"duration": "Habituation"})
    df_test = df_cond.loc[df_cond['event'].str.contains("Test")]
    df_test = df_test.merge(df_hab, on="VP")
    df_test["duration"] = df_test["duration"] - df_test["Habituation"]
    if condition == "friendly":
        dwelltime_friendly = df_test["duration"].to_list()
    elif condition == "unfriendly":
        dwelltime_unfriendly = df_test["duration"].to_list()

# parameters for power analysis
values = dwelltime_friendly + dwelltime_unfriendly
std = np.std(values)
mean_diff = abs(np.mean(dwelltime_unfriendly) - np.mean(dwelltime_friendly))
differences = np.arange(6, 31, 1)
effect_sizes = differences/std
sample_sizes = []

# calculate sample sizes based on effect sizes
for effect_size in effect_sizes:
    # effect_size = effect_sizes[1]
    sample_sizes.append(pwr_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"))

# plot
dv = "Dwell Time"
measure = "s"
meaningful_diff = 20
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax.plot(differences, sample_sizes, color=colors[0])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_title(f"Power Analysis (Within): {dv}", fontweight="bold", fontsize="x-large")
ax.set_xlabel(f"Difference [{measure}]")
ax.set_ylabel(f"Required Sample Size")
ax.set_ylim([np.min(sample_sizes), np.max(sample_sizes)])
ax.set_xlim([np.min(differences), np.max(differences)])

# Piloting Study:
ax.axvline(mean_diff, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), 0.8 * np.max(sample_sizes), f"Mean Difference\nfrom Pilot Study: \n{round(mean_diff, 2)}{measure}", color=colors[1])
n = pwr_analysis.solve_power(effect_size=mean_diff/std, alpha=alpha, power=power, alternative="two-sided")
ax.axhline(n, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}", color=colors[1])

# Meaningful Effect:
ax.axvline(meaningful_diff, color=colors[3], linestyle="--")
ax.text(meaningful_diff + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Meaningful\nEffect: \n{round(meaningful_diff, 2)}{measure}", color=colors[3])
n = pwr_analysis.solve_power(effect_size=meaningful_diff/std, alpha=alpha, power=power, alternative="two-sided")
ax.axhline(n, color=colors[3], linestyle="--")
ax.text(meaningful_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}", color=colors[3])

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"pwr_{dv.lower().replace(' ', '_')}.png"), dpi=300)
plt.close()

# =============================================================================
# Interpersonal Distance
# =============================================================================
df_dist = pd.read_csv(os.path.join(dir_path, 'Data', 'distance.csv'), decimal='.', sep=';')
df_dist = df_dist.loc[df_dist["distance"] <= 500]
df_dist = df_dist.loc[df_dist["event"].str.contains("Test") & ~(df_dist["event"].str.contains("Clicked"))]
df_dist = df_dist.groupby(["VP", "Condition"]).mean().reset_index()
df_dist = df_dist.loc[~(df_dist["Condition"].str.contains("unknown"))]
conditions = ["friendly", "neutral", "unfriendly"]

for condition in conditions:
    df_cond = df_dist.loc[df_dist['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.dropna(subset="distance")
    df_cond = df_cond.groupby(["VP"]).mean().reset_index()
    if condition == "friendly":
        distance_friendly = df_cond["distance"].to_list()
    elif condition == "unfriendly":
        distance_unfriendly = df_cond["distance"].to_list()
    elif condition == "neutral":
        distance_neutral = df_cond["distance"].to_list()

# parameters for power analysis
values = distance_friendly + distance_unfriendly + distance_neutral
std = np.std(values)
mean_diff = abs(np.mean(distance_friendly) - np.mean(distance_unfriendly))
differences = np.arange(5, 41, 1)
effect_sizes = differences/std
sample_sizes = []

# calculate sample sizes based on effect sizes
for effect_size in effect_sizes:
    # effect_size = effect_sizes[1]
    sample_sizes.append(pwr_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="larger"))

# plot
dv = "Interpersonal Distance"
measure = "cm"
meaningful_diff = 12
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax.plot(differences, sample_sizes, color=colors[0])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_title(f"Power Analysis (Within): {dv}", fontweight="bold", fontsize="x-large")
ax.set_xlabel(f"Difference [{measure}]")
ax.set_ylabel(f"Required Sample Size")
ax.set_ylim([np.min(sample_sizes), np.max(sample_sizes)])
ax.set_xlim([np.min(differences), np.max(differences)])

# Piloting Study:
ax.axvline(mean_diff, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), 0.8 * np.max(sample_sizes), f"Mean Difference\nfrom Pilot Study: \n{round(mean_diff, 2)}{measure}", color=colors[1])
n = pwr_analysis.solve_power(effect_size=mean_diff/std, alpha=alpha, power=power, alternative="larger")
ax.axhline(n, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}", color=colors[1])

# Meaningful Effect:
ax.axvline(meaningful_diff, color=colors[3], linestyle="--")
ax.text(meaningful_diff + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Meaningful\nEffect: \n{round(meaningful_diff, 2)}{measure}", color=colors[3])
n = pwr_analysis.solve_power(effect_size=meaningful_diff/std, alpha=alpha, power=power, alternative="larger")
ax.axhline(n, color=colors[3], linestyle="--")
ax.text(meaningful_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}", color=colors[3])

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"pwr_{dv.lower().replace(' ', '_')}.png"), dpi=300)
plt.close()

# =============================================================================
# Gaze
# =============================================================================
df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';')
df_gaze = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
conditions = ["friendly", "neutral", "unfriendly"]

for roi in ["body", "head"]:
    df_roi = df_gaze.loc[df_gaze['ROI'] == roi].reset_index(drop=True)
    for condition in conditions:
        df_cond = df_roi.loc[df_roi['Condition'] == condition].reset_index(drop=True)
        if condition == "friendly":
            gaze_friendly = df_cond["Gaze Proportion"].to_list()
            gaze_friendly = [item * 100 for item in gaze_friendly]
        elif condition == "unfriendly":
            gaze_unfriendly = df_cond["Gaze Proportion"].to_list()
            gaze_unfriendly = [item * 100 for item in gaze_unfriendly]
        elif condition == "neutral":
            gaze_neutral = df_cond["Gaze Proportion"].to_list()
            gaze_neutral = [item * 100 for item in gaze_neutral]

    # parameters for power analysis
    values = gaze_friendly + gaze_unfriendly + gaze_neutral
    std = np.std(values)
    mean_diff = abs(np.mean(gaze_friendly) - np.mean(gaze_unfriendly))

    if roi == "head":
        differences = np.arange(2, 7, 0.5)
        meaningful_diff = 5
    elif roi == "body":
        differences = np.arange(4, 15, 0.5)
        meaningful_diff = 10
    effect_sizes = differences/std
    sample_sizes = []

    # calculate sample sizes based on effect sizes
    for effect_size in effect_sizes:
        # effect_size = effect_sizes[1]
        sample_sizes.append(pwr_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"))

    # plot
    dv = f"Gaze Proportion ({roi.capitalize()})"
    measure = "%"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    ax.plot(differences, sample_sizes, color=colors[0])
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
    ax.set_title(f"Power Analysis (Within): {dv}", fontweight="bold", fontsize="x-large")
    ax.set_xlabel(f"Difference [{measure}]")
    ax.set_ylabel(f"Required Sample Size")
    ax.set_ylim([np.min(sample_sizes), np.max(sample_sizes)])
    ax.set_xlim([np.min(differences), np.max(differences)])

    # Piloting Study:
    ax.axvline(mean_diff, color=colors[1], linestyle="--")
    ax.text(mean_diff + 0.01 * np.max(differences), 0.8 * np.max(sample_sizes), f"Mean Difference\nfrom Pilot Study: \n{round(mean_diff, 2)}{measure}", color=colors[1])
    n = pwr_analysis.solve_power(effect_size=mean_diff / std, alpha=alpha, power=power, alternative="two-sided")
    ax.axhline(n, color=colors[1], linestyle="--")
    ax.text(mean_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}",  color=colors[1])

    # Meaningful Effect:
    ax.axvline(meaningful_diff, color=colors[3], linestyle="--")
    ax.text(meaningful_diff + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Meaningful\nEffect: \n{round(meaningful_diff, 2)}{measure}", color=colors[3])
    n = pwr_analysis.solve_power(effect_size=meaningful_diff / std, alpha=alpha, power=power, alternative="two-sided")
    ax.axhline(n, color=colors[3], linestyle="--")
    ax.text(meaningful_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}", color=colors[3])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"pwr_{dv.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()