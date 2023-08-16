# =============================================================================
# Scores
# source: SosciSurvey
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
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
file_path = os.path.join(dir_path, 'Data')
file_name = [item for item in os.listdir(file_path) if (item.endswith(".xlsx") and "raw" in item)][0]
df_scores = pd.read_excel(os.path.join(file_path, file_name))

# Recode variables from numeric to string
df_scores['gender'] = df_scores['gender'].replace({1: "male", 2: "female", 3: "diverse"})
df_scores['tiredness'] = df_scores['tiredness'].replace({5: 1})
df_scores['motivation'] = df_scores['motivation'].replace({5: 1, 6: 5})
df_scores['handedness'] = df_scores['handedness'].replace({1: "right", 2: "left"})
df_scores['smoking'] = df_scores['smoking'].replace({1: "smoker", 2: "no smoker"})

# ToDo: Adapt problematic subject list
problematic_subjects = [1, 3, 12, 19, 33, 45, 46]

# % ===========================================================================
# SPAI
# =============================================================================
df_spai = df_scores.filter(like='spai')
# Adapt scaling (from 1-7 to 0-6)
df_spai = df_spai - 1

# Calculate means of nested items
columns_spai_multi = [col for col in df_spai.columns if '_' in col]
df_spai_multi = df_spai[columns_spai_multi]
columns_spai = [col for col in df_spai.columns if not '_' in col]
df_spai = df_spai[columns_spai]
items = list(dict.fromkeys([int(column.split('spai')[1].split('_')[0]) for column in df_spai_multi.columns]))
for item in items:
    # item = items[0]
    df_spai_multi_subset = df_spai_multi.filter(like=f'{item}_')
    df_spai[f'spai_{item}'] = df_spai_multi_subset.mean(axis=1)

# Calculate mean of scale
df_spai['SPAI'] = df_spai.mean(axis=1)
df_spai = df_spai[['SPAI']]

# % ===========================================================================
# SIAS
# =============================================================================
df_sias = df_scores.filter(like='sias')
# Adapt scaling (from 1-5 to 0-4)
df_sias = df_sias - 1

# Calculate sum of items
df_sias['SIAS'] = df_sias.sum(axis=1)
df_sias = df_sias[['SIAS']]

# % ===========================================================================
# AQ-K
# =============================================================================
df_aqk = df_scores.filter(like='AQ_')
# Dichotomize scale
df_aqk = df_aqk.replace([1, 2, 3, 4], [0, 0, 1, 1])

# Build subscales and calculate sum of respective items
df_aqk_si = df_aqk.filter(like='SI')  # Soziale Interaktion und Spontanitaet
df_aqk['AQ-K_SI'] = df_aqk_si.sum(axis=1)
df_aqk_fv = df_aqk.filter(like='FV')  # Fantasie und Vorstellungsvermoegen
df_aqk['AQ-K_FV'] = df_aqk_fv.sum(axis=1)
df_aqk_kr = df_aqk.filter(like='KR')  # Kommunikation und Reziprozitaet
df_aqk['AQ-K_KR'] = df_aqk_kr.sum(axis=1)

# Sum of subscales
df_aqk['AQ-K'] = df_aqk[['AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV']].sum(axis=1)
df_aqk = df_aqk[['AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV']]

# % ===========================================================================
# ISK-K
# =============================================================================
df_isk = df_scores.filter(like='ISK_')

# Build subscales and calculate sum of respective items
df_isk_so = df_isk.filter(like='SO')  # Soziale Orientierung
df_isk['ISK-K_SO'] = df_isk_so.sum(axis=1)
df_isk_of = df_isk.filter(like='OF')  # Offensivitaet
df_isk['ISK-K_OF'] = df_isk_of.sum(axis=1)
df_isk_sst = df_isk.filter(like='SSt')  # Selbststeuerung
df_isk['ISK-K_SSt'] = df_isk_sst.sum(axis=1)
df_isk_re = df_isk.filter(like='RE')  # Reflexibilitaet
df_isk['ISK-K_RE'] = df_isk_re.sum(axis=1)
df_isk = df_isk[['ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']]

# % ===========================================================================
# ASI
# =============================================================================
df_asi = df_scores.filter(like='ASI_')
df_asi = df_asi - 1

df_asi_pc = df_asi.filter(like='PC')  # Physical Concerns
df_asi['ASI3-PC'] = df_asi_pc.sum(axis=1)
df_asi_cc = df_asi.filter(like='CC')  # Cognitive Concerns
df_asi['ASI3-CC'] = df_asi_cc.sum(axis=1)
df_asi_sc = df_asi.filter(like='SC')  # Social Concerns
df_asi['ASI3-SC'] = df_asi_sc.sum(axis=1)
df_asi['ASI3'] = df_asi.sum(axis=1)

df_asi = df_asi[['ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC']]

# % ===========================================================================
# SSQ
# =============================================================================
df_ssq = df_scores.filter(like='SSQ')
df_ssq = df_ssq - 1

df_ssq_pre = df_ssq.filter(like='pre')
df_ssq_post = df_ssq.filter(like='post')
# means = df_ssq_pre.mean()
# means = means.sort_values()

# Nausea
df_ssq_n_pre = df_ssq_pre.filter(like='N')
df_ssq_n_post = df_ssq_post.filter(like='N')
df_ssq['SSQ-pre-N'] = df_ssq_n_pre.sum(axis=1) * 9.54
df_ssq['SSQ-post-N'] = df_ssq_n_post.sum(axis=1) * 9.54
df_ssq['SSQ-diff-N'] = df_ssq['SSQ-post-N'] - df_ssq['SSQ-pre-N']

# Oculomotor Disturbance
df_ssq_o_pre = df_ssq_pre.filter(like='O')
df_ssq_o_post = df_ssq_post.filter(like='O')
df_ssq['SSQ-pre-O'] = df_ssq_o_pre.sum(axis=1) * 7.58
df_ssq['SSQ-post-O'] = df_ssq_o_post.sum(axis=1) * 7.58
df_ssq['SSQ-diff-O'] = df_ssq['SSQ-post-O'] - df_ssq['SSQ-pre-O']

# Disorientation
df_ssq_d_pre = df_ssq_pre.filter(like='D')
df_ssq_d_post = df_ssq_post.filter(like='D')
df_ssq['SSQ-pre-D'] = df_ssq_d_pre.sum(axis=1) * 13.92
df_ssq['SSQ-post-D'] = df_ssq_d_post.sum(axis=1) * 13.92
df_ssq['SSQ-diff-D'] = df_ssq['SSQ-post-D'] - df_ssq['SSQ-pre-D']

df_ssq['SSQ-pre'] = (df_ssq_n_pre.sum(axis=1) + df_ssq_o_pre.sum(axis=1) + df_ssq_d_pre.sum(axis=1)) * 3.74
df_ssq['SSQ-post'] = (df_ssq_n_post.sum(axis=1) + df_ssq_o_post.sum(axis=1) + df_ssq_d_post.sum(axis=1)) * 3.74
df_ssq['SSQ-diff'] = df_ssq['SSQ-post'] - df_ssq['SSQ-pre']

df_subjects = df_scores[['ID', 'age', 'gender']]
df_problematic_subjects = pd.concat([df_subjects, df_ssq], axis=1)
cutoff = df_problematic_subjects["SSQ-diff"].mean() + df_problematic_subjects["SSQ-diff"].std()  # .quantile(0.75)
df_problematic_subjects = df_problematic_subjects.loc[df_problematic_subjects["SSQ-diff"] > cutoff]
problematic_subjects = list(np.unique(problematic_subjects + df_problematic_subjects["ID"].to_list()))

# old_columns = df_problematic_subjects.columns
# replacements = {
#     "SSQ_N_O_01": "Allgemeines_Unwohlsein",
#     "SSQ_O_02": "Ermüdung",
#     "SSQ_O_03": "Kopfschmerzen",
#     "SSQ_O_04": "Angestrengte_Augen",
#     "SSQ_O_D_05": "Schwierigkeiten_ScharfesSehen",
#     "SSQ_N_06": "Erhöhter_Speichelfluss",
#     "SSQ_N_07": "Schwitzen",
#     "SSQ_N_D_08": "Übelkeit",
#     "SSQ_N_O_09": "Konzentrationsschwierigkeiten",
#     "SSQ_D_10": "Kopfdruck",
#     "SSQ_O_D_11": "Verschwommenes_Sehen",
#     "SSQ_D_12": "Schwindel_OffeneAugen",
#     "SSQ_D_13": "Schwindel_GeschlosseneAugen",
#     "SSQ_D_14": "Gleichgewichtsstörungen",
#     "SSQ_N_15": "Magenprobleme",
#     "SSQ_N_16": "Aufstoßen"}
# new_columns = old_columns
# for key in replacements.keys():
#     new_columns = [item.replace(key, replacements[key]) for item in list(new_columns)]

df_ssq = df_ssq[['SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff', 'SSQ-diff-N', 'SSQ-diff-O', 'SSQ-diff-D']]

# % ===========================================================================
# MPS (Multimodal Presence Scale)
# =============================================================================
df_mps = df_scores.filter(like='MPS')
# Physical Presence
df_mps_pp = df_scores.filter(like='PP')
df_mps['MPS-PP'] = df_mps_pp.mean(axis=1)

# Social Presence
df_mps_socp = df_scores.filter(like='SocP')
df_mps['MPS-SocP'] = df_mps_socp.mean(axis=1)

# Self Presence
df_mps_selfp = df_scores.filter(like='SelfP')
df_mps['MPS-SelfP'] = df_mps_selfp.mean(axis=1)

df_mps = df_mps[['MPS-PP', 'MPS-SocP', 'MPS-SelfP']]

# % ===========================================================================
# IPQ (Igroup Presence Questionnaire)
# =============================================================================
df_ipq = df_scores.filter(like='IPQ')
df_ipq = df_ipq - 1

df_ipq_sp = df_ipq.filter(like='SP')  # Spatial Presence
df_ipq['IPQ-SP'] = df_ipq_sp.mean(axis=1)
df_ipq_er = df_ipq.filter(like='ER')  # Involvement
df_ipq['IPQ-ER'] = df_ipq_er.mean(axis=1)
df_ipq_inv = df_ipq.filter(like='INV')  # Experienced Realism
df_ipq['IPQ-INV'] = df_ipq_inv.mean(axis=1)
df_ipq['IPQ'] = df_ipq.mean(axis=1)

df_ipq = df_ipq[['IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV']]

# % ===========================================================================
# VAS: State Anxiety, Nervousness, Distress, Stress
# =============================================================================
df_vas = df_scores.filter(like='VAS')

# % ===========================================================================
# Create Summary
# =============================================================================
df = df_scores[['ID', 'age', 'gender', 'handedness', 'motivation', 'tiredness', 'purpose', 'variables']]
df = pd.concat([df, df_ssq, df_ipq, df_mps, df_vas, df_asi, df_spai, df_sias, df_aqk, df_isk], axis=1)
df['purpose'] = df['purpose'].str.replace('-', '')
# df['purpose'] = df['purpose'].astype("string")
df['variables'] = df['variables'].str.replace('-', '')
# df['variables'] = df['variables'].astype("string")
df['ID'] = df['ID'].astype('string')
df = df.loc[~(df['ID'].str.contains('test'))]
df['ID'] = df['ID'].astype('int32')

df.to_csv(os.path.join(file_path, 'scores_summary.csv'), index=False, decimal=',', sep=';', encoding='utf-8-sig')

df = pd.read_csv(os.path.join(file_path, 'scores_summary.csv'), decimal=',', sep=';')

problematic_subjects = [1, 3, 12, 15, 19, 20, 23, 24, 31, 33, 41, 42, 45, 46, 47, 53]
df = df.loc[~(df["ID"].isin(problematic_subjects))]
print(f"N = {len(df)}")
print(f"Mean Age = {df['age'].mean()}, SD = {df['age'].std()}, Range = {df['age'].min()}-{df['age'].max()}")
print(df['gender'].value_counts(normalize=True))


df = pd.read_csv(os.path.join(file_path, 'scores_summary.csv'), decimal=',', sep=';')
save_path = os.path.join(dir_path, 'Plots', 'Scores')
if not os.path.exists(save_path):
    print('creating path for saving')
    os.makedirs(save_path)

scales = ["SSQ-post", "SSQ-diff", "IPQ", "MPS", "ASI", "SPAI", "SIAS", "AQ", "ISK"]
colors = ['#1F82C0', '#F29400', '#E2001A', '#B1C800', '#179C7D']

for idx_scale, scale in enumerate(scales):
    # idx_scale = 1
    # scale = scales[idx_scale]
    df_scale = df.filter(like=scale)
    min = np.min(df_scale.min()) - 0.02 * np.max(df_scale.max())
    max = np.max(df_scale.max()) + 0.02 * np.max(df_scale.max())
    n_subscales = len(df_scale.columns)
    if n_subscales > 1:
        fig, axes = plt.subplots(nrows=1, ncols=n_subscales, figsize=(n_subscales * 2, 6))
        boxWidth = 1
        pos = [1]

        for idx_subscale, subscale in enumerate(df_scale.columns):
            # idx_subscale = 0
            # subscale = df_scale.columns[idx_subscale]

            # Plot raw data points
            for i in range(len(df)):
                # i = 0
                x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
                y = df_scale.reset_index().loc[i, subscale].item()
                if df.reset_index().loc[i, "ID"].item() in problematic_subjects:
                    axes[idx_subscale].plot(x, y, marker='o', ms=5, mfc="grey", mec="grey", alpha=0.3)
                else:
                    axes[idx_subscale].plot(x, y, marker='o', ms=5, mfc=colors[idx_subscale], mec=colors[idx_subscale], alpha=0.3)

            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
            fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            boxprops = dict(color=colors[idx_subscale])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = bootstrapping(df_scale.loc[:, subscale].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            axes[idx_subscale].boxplot([df_scale.loc[:, subscale].values],
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
                       positions=[pos[0]],
                       widths=0.8 * boxWidth)

            axes[idx_subscale].set_xticklabels([subscale])
            axes[idx_subscale].set_ylim(min, max)
            axes[idx_subscale].grid(color='lightgrey', linestyle='-', linewidth=0.3)
            if subscale == "SSQ-diff":
                axes[idx_subscale].axhline(cutoff, color="lightgrey", linewidth=0.8, linestyle="dashed")

        fig.suptitle(scale)
        plt.tight_layout()
        for end in (['.png']):  # '.pdf',
            plt.savefig(os.path.join(save_path, f"{scale}_vr{end}"), dpi=300)
        plt.close()
    elif n_subscales == 1:
        fig, ax = plt.subplots(nrows=1, ncols=n_subscales, figsize=(n_subscales * 2, 6))
        boxWidth = 1
        pos = [1]
        for idx_subscale, subscale in enumerate(df_scale.columns):
            # idx_subscale = 0
            # subscale = df_scale.columns[idx_subscale]

            # Plot raw data points
            for i in range(len(df)):
                # i = 0
                x = random.uniform(pos[0] - (0.25 * boxWidth), pos[0] + (0.25 * boxWidth))
                y = df_scale.reset_index().loc[i, subscale].item()
                if df.reset_index().loc[i, "ID"].item() in problematic_subjects:
                    ax.plot(x, y, marker='o', ms=5, mfc="grey", mec="grey", alpha=0.3)
                else:
                    ax.plot(x, y, marker='o', ms=5, mfc=colors[idx_subscale], mec=colors[idx_subscale], alpha=0.3)


            # Plot boxplots
            meanlineprops = dict(linestyle='solid', linewidth=1, color='black')
            medianlineprops = dict(linestyle='dashed', linewidth=1, color='grey')
            fliermarkerprops = dict(marker='o', markersize=1, color='lightgrey')

            whiskerprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            capprops = dict(linestyle='solid', linewidth=1, color=colors[idx_subscale])
            boxprops = dict(color=colors[idx_subscale])

            fwr_correction = True
            alpha = (1 - (0.05))
            bootstrapping_dict = bootstrapping(df_scale.loc[:, subscale].values,
                                               numb_iterations=5000,
                                               alpha=alpha,
                                               as_dict=True,
                                               func='mean')

            ax.boxplot([df_scale.loc[:, subscale].values],
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
                                       positions=[pos[0]],
                                       widths=0.8 * boxWidth)

            ax.set_xticklabels([subscale])
            # ax.set_ylim(min, max)
            ax.grid(color='lightgrey', linestyle='-', linewidth=0.3)
        fig.suptitle(scale)
        plt.tight_layout()
        for end in (['.png']):  # '.pdf',
            plt.savefig(os.path.join(save_path, f"{scale}_vr{end}"), dpi=300)
        plt.close()

    df_plot = df.copy()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 1.6))
colors = ['#FF5733', '#FFC300', '#183DB2']
df_plot = df_plot.loc[~(df_plot["ID"].isin(problematic_subjects))]
sns.histplot(df_plot["SPAI"], color="#1B4C87", ax=ax, binwidth=0.2, binrange=(0, 5),
             kde=True, line_kws={"linewidth": 1, "color": "#173d6a"}, edgecolor='#f8f8f8',)
ax.set_xlabel("SPAI (Social Anxiety)")
ax.set_xlim([0, 5])
# ax.set_ylim([0, 11])
ax.set_yticks(range(0, 5, 2))
ax.axvline(x=df_plot["SPAI"].median(), color="#FFC300")
ax.axvline(x=2.79, color="#FF5733")
ax.legend(
        [Line2D([0], [0], color='#FFC300'), Line2D([0], [0], color='#FF5733')],
        ['Median', 'Remission Cut-Off'], fontsize='xx-small', loc="best", frameon=False)
ax.set_facecolor('#f8f8f8')
plt.tight_layout()
for end in (['.png']):  # '.pdf',
    plt.savefig(os.path.join(save_path, f"Distribution_SPAI{end}"), dpi=300)
plt.close()

"""
from rpy2.situation import (get_r_home)
os.environ["R_HOME"] = get_r_home()
# Execute one block after another
%load_ext rpy2.ipython

df_corr = df[['ASI3', 'SPAI', 'SIAS', 'AQ-K', 'VAS_start_anxiety', 'SSQ-diff', 'IPQ', 'MPS-SocP']]

%%R -i df_corr
library(apaTables)
library(tidyverse)
library(readr)
apa.cor.table(df_corr, filename = "test.doc", show.sig.stars=TRUE)
"""