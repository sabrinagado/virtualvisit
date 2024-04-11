# =============================================================================
# Scores
# source: SosciSurvey
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
from functools import reduce

pd.options.mode.chained_assignment = None


# % ===========================================================================
# SPAI
# =============================================================================
def calculate_spai(df_spai):
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
    return df_spai


# % ===========================================================================
# SIAS
# =============================================================================
def calculate_sias(df_sias):
    # Adapt scaling (from 1-5 to 0-4)
    df_sias = df_sias - 1

    # Calculate sum of items
    df_sias['SIAS'] = df_sias.sum(axis=1)
    df_sias = df_sias[['SIAS']]
    return df_sias


# % ===========================================================================
# AQ-K
# =============================================================================
def calculate_aqk(df_aqk):
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
    return df_aqk


# % ===========================================================================
# ISK-K
# =============================================================================
def calculate_isk(df_isk):
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
    return df_isk


# % ===========================================================================
# ASI
# =============================================================================
def calculate_asi(df_asi):
    # Adapt scaling
    df_asi = df_asi - 1

    df_asi['ASI3'] = df_asi.sum(axis=1)
    df_asi_pc = df_asi.filter(like='PC')  # Physical Concerns
    df_asi['ASI3-PC'] = df_asi_pc.sum(axis=1)
    df_asi_cc = df_asi.filter(like='CC')  # Cognitive Concerns
    df_asi['ASI3-CC'] = df_asi_cc.sum(axis=1)
    df_asi_sc = df_asi.filter(like='SC')  # Social Concerns
    df_asi['ASI3-SC'] = df_asi_sc.sum(axis=1)

    df_asi = df_asi[['ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC']]
    return df_asi


# % ===========================================================================
# SSQ
# =============================================================================
def calculate_ssq(df_ssq):
    # Adapt scaling
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
    df_ssq = df_ssq[['SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff', 'SSQ-diff-N', 'SSQ-diff-O', 'SSQ-diff-D']]

    return df_ssq


# % ===========================================================================
# MPS (Multimodal Presence Scale)
# =============================================================================
def calculate_mps(df_mps):
    # Physical Presence
    df_mps_pp = df_mps.filter(like='PP')
    df_mps['MPS-PP'] = df_mps_pp.mean(axis=1)

    # Social Presence
    df_mps_socp = df_mps.filter(like='SocP')
    df_mps['MPS-SocP'] = df_mps_socp.mean(axis=1)

    # Self Presence
    df_mps_selfp = df_mps.filter(like='SelfP')
    df_mps['MPS-SelfP'] = df_mps_selfp.mean(axis=1)

    df_mps = df_mps[['MPS-PP', 'MPS-SocP', 'MPS-SelfP']]
    return df_mps


# % ===========================================================================
# IPQ (Igroup Presence Questionnaire)
# =============================================================================
def calculate_ipq(df_ipq):
    # Adapt scaling
    df_ipq = df_ipq - 1

    df_ipq_sp = df_ipq.filter(like='SP')  # Spatial Presence
    df_ipq['IPQ-SP'] = df_ipq_sp.mean(axis=1)
    df_ipq_er = df_ipq.filter(like='ER')  # Involvement
    df_ipq['IPQ-ER'] = df_ipq_er.mean(axis=1)
    df_ipq_inv = df_ipq.filter(like='INV')  # Experienced Realism
    df_ipq['IPQ-INV'] = df_ipq_inv.mean(axis=1)
    df_ipq['IPQ'] = df_ipq.mean(axis=1)

    df_ipq = df_ipq[['IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV']]
    return df_ipq


# % ===========================================================================
# Create Summary
# =============================================================================
def create_scores(df, df_labbook, problematic_subjects=None):
    # df = df_scores
    df_demo = df[['ID', 'age', 'gender', 'handedness', 'motivation', 'tiredness', 'purpose', 'variables']]
    # Recode variables from numeric to string
    df_demo['gender'].replace({1: "male", 2: "female", 3: "diverse"}, inplace=True)
    df_demo['handedness'].replace({1: "right", 2: "left"}, inplace=True)

    df_spai = calculate_spai(df.filter(like='spai')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_sias = calculate_sias(df.filter(like='sias')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_aqk = calculate_aqk(df.filter(like='AQ_')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_isk = calculate_isk(df.filter(like='ISK_')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_asi = calculate_asi(df.filter(like='ASI_')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_ssq = calculate_ssq(df.filter(like='SSQ')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_mps = calculate_mps(df.filter(like='MPS')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_ipq = calculate_ipq(df.filter(like='IPQ')).merge(df.loc[:, "ID"], left_index=True, right_index=True)
    df_vas = df.filter(like='VAS').merge(df.loc[:, "ID"], left_index=True, right_index=True)

    df_summary = reduce(lambda left, right: pd.merge(left, right, on=['ID'], how='outer'), [df_demo, df_ssq, df_ipq, df_mps, df_vas, df_asi, df_spai, df_sias, df_aqk, df_isk])

    cutoff_ssq = df_summary["SSQ-diff"].mean() + df_summary["SSQ-diff"].std()  # .quantile(0.75)
    df_problematic_subjects = df_summary.loc[df_summary["SSQ-diff"] > cutoff_ssq]
    problematic_subjects = list(np.unique(problematic_subjects + df_problematic_subjects["ID"].to_list()))

    df_summary['purpose'] = df_summary['purpose'].str.replace('-', ', ')
    df_summary['variables'] = df_summary['variables'].str.replace('-', ', ')
    df_summary['purpose'] = df_summary['purpose'].str.replace('\n', ', ')
    df_summary['variables'] = df_summary['variables'].str.replace('\n', ', ')
    df_summary['ID'] = df_summary['ID'].astype('string')
    df_summary = df_summary.loc[~(df_summary['ID'].str.contains('test'))]
    df_summary['ID'] = df_summary['ID'].astype('int32')

    df_labbook = df_labbook[["VP", "Raumtemperatur", "Luftfeuchtigkeit"]]
    df_labbook.columns = ["VP", "temperature", "humidity"]

    df_summary = df_summary.merge(df_labbook, left_on="ID", right_on="VP")

    df_summary = df_summary.drop(columns="VP")

    if problematic_subjects:
        return df_summary, problematic_subjects
    else:
        return df_summary


if __name__ == '__main__':
    wave = 1
    dir_path = os.getcwd()
    file_path = os.path.join(dir_path, f'Data-Wave{wave}')
    file_name = [item for item in os.listdir(file_path) if (item.endswith(".xlsx") and "raw" in item)][0]
    df = pd.read_excel(os.path.join(file_path, file_name))
    df = df.loc[df["FINISHED"] == 1]

    file_name_labbook = [item for item in os.listdir(file_path) if (item.endswith(".xlsx") and "Labbook" in item)][0]
    df_labbook = pd.read_excel(os.path.join(file_path, file_name_labbook), sheet_name=f"Wave{wave}")

    # ToDo: Adapt problematic subject list
    if wave == 1:
        problematic_subjects = [1, 3, 12, 19, 33, 45, 46]
    elif wave == 2:
        problematic_subjects = [1, 2, 3, 4, 20, 29, 64]

    df_summary, problematic_subjects = create_scores(df, df_labbook, problematic_subjects)

    df_summary.to_csv(os.path.join(file_path, 'scores_summary.csv'), index=False, decimal=',', sep=';', encoding='utf-8-sig')

    print(f"Problematic Subject: {problematic_subjects}")
