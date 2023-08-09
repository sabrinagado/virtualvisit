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
N1 = 48
N2 = 96
pwr_analysis = pwr.TTestPower()

# =============================================================================
# Dwell Time
# =============================================================================
df_dwell = pd.read_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';')
df_dwell = df_dwell.loc[df_dwell["event"].str.contains("Habituation") | df_dwell["event"].str.contains("Test") & ~(df_dwell["event"].str.contains("Clicked"))]
conditions = ["friendly", "unfriendly"]
dwelltimes = pd.DataFrame()
vps_enterRoom = []
for condition in conditions:
    # condition = "unfriendly"
    df_cond = df_dwell.loc[df_dwell['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.dropna(subset="duration")
    df_cond = df_cond.groupby(["VP", "event"]).sum().reset_index()
    df_hab = df_cond.loc[df_cond['event'].str.contains("Habituation")]
    df_hab = df_hab[["VP", "duration"]]
    df_hab = df_hab.rename(columns={"duration": "Habituation"})
    df_test = df_cond.loc[df_cond['event'].str.contains("Test")]
    df_test = df_test.merge(df_hab, on="VP")
    df_test["duration"] = df_test["duration"] - df_test["Habituation"]
    df_test = df_test.rename(columns={"duration": condition})
    if condition == "friendly":
        dwelltimes = pd.concat([dwelltimes, df_test[["VP", condition]]])
        vps_enterRoom = vps_enterRoom + df_test["VP"].to_list()
    elif condition == "unfriendly":
        dwelltimes = dwelltimes.merge(df_test[["VP", condition]], on="VP")
        vps_enterRoom = vps_enterRoom + df_test["VP"].to_list()

dwelltime_friendly = dwelltimes["friendly"].to_list()
dwelltime_unfriendly = dwelltimes["unfriendly"].to_list()

vps_enterRoom = np.unique(vps_enterRoom)

# parameters for power analysis
values = dwelltime_friendly + dwelltime_unfriendly
std = np.std(values)
mean_diff = abs(np.mean(dwelltime_unfriendly) - np.mean(dwelltime_friendly))
differences = np.arange(10, 31, 1)
effect_sizes = differences/std
sample_sizes = []

# calculate sample sizes based on effect sizes
for effect_size in effect_sizes:
    # effect_size = effect_sizes[1]
    sample_sizes.append(pwr_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"))

# plot
dv = "Dwell Time"
measure = "s"
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5.5))
ax.plot(differences, sample_sizes, color=colors[0])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_title(f"Power Analysis: {dv}", fontweight="bold", fontsize="x-large")
ax.set_xlabel(f"Difference [{measure}]")
ax.set_ylabel(f"Required Sample Size")
ax.set_ylim([np.min(sample_sizes), np.max(sample_sizes)])
ax.set_xlim([np.min(differences), np.max(differences)])

# Piloting Study:
ax.axvline(mean_diff, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), 0.8 * np.max(sample_sizes), f"Mean Difference\nfrom Pilot Study:\n{round(mean_diff, 2)} {measure}", color=colors[1])
n = pwr_analysis.solve_power(effect_size=mean_diff/std, alpha=alpha, power=power, alternative="two-sided")
ax.axhline(n, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}", color=colors[1])

# N = 48:
effect = pwr_analysis.solve_power(nobs=N1, alpha=alpha, power=power, alternative="two-sided") * std
ax.axhline(N1, color=colors[3], linestyle="--")
ax.text(effect + 0.01 * np.max(differences), N1 + 0.01 * np.max(sample_sizes), f"N: {math.ceil(N1)}", color=colors[3])
ax.axvline(effect, color=colors[3], linestyle="--")
pwr = pwr_analysis.solve_power(effect_size=mean_diff/std, nobs=N1, alpha=alpha, alternative="two-sided")
ax.text(effect + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Effect for N = {N1}: {round(effect, 2)} {measure},\nPower: {round(pwr, 2)}", color=colors[3])

# # N = 96:
# effect = pwr_analysis.solve_power(nobs=N2, alpha=alpha, power=power, alternative="two-sided") * std
# ax.axhline(N2, color=colors[4], linestyle="--")
# ax.text(effect + 0.01 * np.max(differences), N2 + 0.01 * np.max(sample_sizes), f"N: {math.ceil(N2)}", color=colors[4])
# ax.axvline(effect, color=colors[4], linestyle="--")
# ax.text(effect + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Effect for\nN = {N2}: \n{round(effect, 2)} {measure}", color=colors[4])

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"pwr_{dv.lower().replace(' ', '_')}.png"), dpi=300)
plt.close()

# =============================================================================
# Interpersonal Distance
# =============================================================================
df_dist = pd.read_csv(os.path.join(dir_path, 'Data', 'distance.csv'), decimal='.', sep=';')
df_dist = df_dist.loc[df_dist["distance"] <= 1000]
df_dist = df_dist.loc[df_dist["distance"] >= 1]
df_dist = df_dist[df_dist["VP"].isin(vps_enterRoom)]
df_dist = df_dist.loc[df_dist["event"].str.contains("Test") & ~(df_dist["event"].str.contains("Clicked"))]
df_dist = df_dist.groupby(["VP", "Condition"]).mean().reset_index()
df_dist = df_dist.loc[~(df_dist["Condition"].str.contains("unknown"))]
conditions = ["friendly", "neutral", "unfriendly"]
distances = pd.DataFrame()

for condition in conditions:
    # condition = "friendly"
    df_cond = df_dist.loc[df_dist['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.dropna(subset="distance")
    df_cond = df_cond.groupby(["VP"]).mean().reset_index()
    df_cond = df_cond.rename(columns={"distance": condition})
    if condition == "friendly":
        distances = pd.concat([distances, df_cond[["VP", condition]]])
    elif condition == "neutral":
        distances = distances.merge(df_cond[["VP", condition]], on="VP")
    elif condition == "unfriendly":
        distances = distances.merge(df_cond[["VP", condition]], on="VP")

distance_friendly = distances["friendly"].to_list()
distance_unfriendly = distances["unfriendly"].to_list()
distance_neutral = distances["neutral"].to_list()

# parameters for power analysis
values = distance_friendly + distance_unfriendly # + distance_neutral
std = np.std(values)
mean_diff = abs(np.mean(distance_friendly) - np.mean(distance_unfriendly))
differences = np.arange(10, 41, 1)
effect_sizes = differences/std
sample_sizes = []

# calculate sample sizes based on effect sizes
for effect_size in effect_sizes:
    # effect_size = effect_sizes[1]
    sample_sizes.append(pwr_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="larger"))

# plot
dv = "Interpersonal Distance"
measure = "cm"
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5.5))
ax.plot(differences, sample_sizes, color=colors[0])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_title(f"Power Analysis: {dv}", fontweight="bold", fontsize="x-large")
ax.set_xlabel(f"Difference [{measure}]")
ax.set_ylabel(f"Required Sample Size")
ax.set_ylim([np.min(sample_sizes), np.max(sample_sizes)])
ax.set_xlim([np.min(differences), np.max(differences)])

# Piloting Study:
ax.axvline(mean_diff, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), 0.8 * np.max(sample_sizes), f"Mean Difference\nfrom Pilot Study:\n{round(mean_diff, 2)} {measure}", color=colors[1])
n = pwr_analysis.solve_power(effect_size=mean_diff/std, alpha=alpha, power=power, alternative="larger")
ax.axhline(n, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}", color=colors[1])

# N = 48:
effect = pwr_analysis.solve_power(nobs=N1, alpha=alpha, power=power, alternative="larger") * std
ax.axhline(N1, color=colors[3], linestyle="--")
ax.text(effect + 0.01 * np.max(differences), N1 + 0.01 * np.max(sample_sizes), f"N: {math.ceil(N1)}", color=colors[3])
ax.axvline(effect, color=colors[3], linestyle="--")
pwr = pwr_analysis.solve_power(effect_size=mean_diff/std, nobs=N1, alpha=alpha, alternative="larger")
ax.text(effect + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Effect for N = {N1}: {round(effect, 2)} {measure},\nPower: {round(pwr, 2)}", color=colors[3])

# # N = 96:
# effect = pwr_analysis.solve_power(nobs=N2, alpha=alpha, power=power, alternative="larger") * std
# ax.axhline(N2, color=colors[4], linestyle="--")
# ax.text(effect + 0.01 * np.max(differences), N2 + 0.01 * np.max(sample_sizes), f"N: {math.ceil(N2)}", color=colors[4])
# ax.axvline(effect, color=colors[4], linestyle="--")
# ax.text(effect + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Effect for\nN = {N2}: \n{round(effect, 2)} {measure}", color=colors[4])

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"pwr_{dv.lower().replace(' ', '_')}.png"), dpi=300)
plt.close()

# =============================================================================
# Gaze
# =============================================================================
df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';')
df_gaze = df_gaze.loc[df_gaze["Phase"].str.contains("Test") & ~(df_gaze["Phase"].str.contains("Clicked"))]
df_gaze = df_gaze.loc[~(df_gaze["Phase"].str.contains("Office"))]
df_gaze = df_gaze[df_gaze["VP"].isin(vps_enterRoom)]
conditions = ["friendly", "unfriendly"]

roi = "head"
df_roi = df_gaze.loc[df_gaze['ROI'] == roi].reset_index(drop=True)
df_roi = df_roi.groupby(["VP", "Condition"]).mean().reset_index()
proportions = pd.DataFrame()
for condition in conditions:
    # condition = "unfriendly"
    df_cond = df_roi.loc[df_roi['Condition'] == condition].reset_index(drop=True)
    df_cond = df_cond.rename(columns={"Gaze Proportion": condition})
    if condition == "friendly":
        proportions = pd.concat([proportions, df_cond[["VP", condition]]])
    elif condition == "unfriendly":
        proportions = proportions.merge(df_cond[["VP", condition]], on="VP")

gaze_friendly = proportions["friendly"].to_list()
gaze_friendly = [item * 100 for item in gaze_friendly]
gaze_unfriendly = proportions["unfriendly"].to_list()
gaze_unfriendly = [item * 100 for item in gaze_unfriendly]

# parameters for power analysis
values = [item * 100 for item in df_roi["Gaze Proportion"].to_list()]
std = np.std(values)
mean_diff = abs(np.mean(gaze_friendly) - np.mean(gaze_unfriendly))

differences = np.arange(2, 7, 0.2)
effect_sizes = differences/std
sample_sizes = []

# calculate sample sizes based on effect sizes
for effect_size in effect_sizes:
    # effect_size = effect_sizes[1]
    sample_sizes.append(pwr_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"))

# plot
dv = f"Gaze Proportion ({roi.capitalize()})"
measure = "%"
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5.5))
ax.plot(differences, sample_sizes, color=colors[0])
ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
ax.set_title(f"Power Analysis: {dv}", fontweight="bold", fontsize="x-large")
ax.set_xlabel(f"Difference [{measure}]")
ax.set_ylabel(f"Required Sample Size")
ax.set_ylim([np.min(sample_sizes), np.max(sample_sizes)])
ax.set_xlim([np.min(differences), np.max(differences)])

# Piloting Study:
ax.axvline(mean_diff, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), 0.8 * np.max(sample_sizes), f"Mean Difference\nfrom Pilot Study:\n{round(mean_diff, 2)} {measure}", color=colors[1])
n = pwr_analysis.solve_power(effect_size=mean_diff / std, alpha=alpha, power=power, alternative="two-sided")
ax.axhline(n, color=colors[1], linestyle="--")
ax.text(mean_diff + 0.01 * np.max(differences), n + 0.01 * np.max(sample_sizes), f"N: {math.ceil(n)}",  color=colors[1])

# N = 48:
effect = pwr_analysis.solve_power(nobs=N1, alpha=alpha, power=power, alternative="two-sided") * std
ax.axhline(N1, color=colors[3], linestyle="--")
ax.text(effect + 0.01 * np.max(differences), N1 + 0.01 * np.max(sample_sizes), f"N: {math.ceil(N1)}", color=colors[3])
ax.axvline(effect, color=colors[3], linestyle="--")
pwr = pwr_analysis.solve_power(effect_size=mean_diff/std, nobs=N1, alpha=alpha, alternative="two-sided")
ax.text(effect + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Effect for N = {N1}: {round(effect, 2)} {measure},\nPower: {round(pwr, 2)}", color=colors[3])

# # N = 96:
# effect = pwr_analysis.solve_power(nobs=N2, alpha=alpha, power=power, alternative="two-sided") * std
# ax.axhline(N2, color=colors[4], linestyle="--")
# ax.text(effect + 0.01 * np.max(differences), N2 + 0.01 * np.max(sample_sizes), f"N: {math.ceil(N2)}", color=colors[4])
# ax.axvline(effect, color=colors[4], linestyle="--")
# ax.text(effect + 0.01 * np.max(differences), 0.65 * np.max(sample_sizes), f"Effect for\nN = {N2}: \n{round(effect, 2)} {measure}", color=colors[4])

plt.tight_layout()
plt.savefig(os.path.join(save_path, f"pwr_{dv.lower().replace(' ', '_')}.png"), dpi=300)
plt.close()