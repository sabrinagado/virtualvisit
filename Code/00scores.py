# =============================================================================
# Scores
# source: SosciSurvey
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np

wave = 2
dir_path = os.getcwd()
file_path = os.path.join(dir_path, f'Data-Wave{wave}')
file_name = [item for item in os.listdir(file_path) if (item.endswith(".xlsx") and "raw" in item)][0]
df_scores = pd.read_excel(os.path.join(file_path, file_name))

# Recode variables from numeric to string
df_scores['gender'] = df_scores['gender'].replace({1: "male", 2: "female", 3: "diverse"})
df_scores['tiredness'] = df_scores['tiredness'].replace({5: 1})
df_scores['motivation'] = df_scores['motivation'].replace({5: 1, 6: 5})
df_scores['handedness'] = df_scores['handedness'].replace({1: "right", 2: "left"})
df_scores['smoking'] = df_scores['smoking'].replace({1: "smoker", 2: "no smoker"})

# ToDo: Adapt problematic subject list
if wave == 1:
    problematic_subjects = [1, 3, 12, 19, 33, 45, 46]
elif wave == 2:
    problematic_subjects = [1, 2, 3, 4]

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

df_asi['ASI3'] = df_asi.sum(axis=1)
df_asi_pc = df_asi.filter(like='PC')  # Physical Concerns
df_asi['ASI3-PC'] = df_asi_pc.sum(axis=1)
df_asi_cc = df_asi.filter(like='CC')  # Cognitive Concerns
df_asi['ASI3-CC'] = df_asi_cc.sum(axis=1)
df_asi_sc = df_asi.filter(like='SC')  # Social Concerns
df_asi['ASI3-SC'] = df_asi_sc.sum(axis=1)

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
cutoff_ssq = df_problematic_subjects["SSQ-diff"].mean() + df_problematic_subjects["SSQ-diff"].std()  # .quantile(0.75)
df_problematic_subjects = df_problematic_subjects.loc[df_problematic_subjects["SSQ-diff"] > cutoff_ssq]
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
