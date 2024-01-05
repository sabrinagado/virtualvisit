# =============================================================================
# Scores
# source: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np

from Code import preproc_scores

pd.options.mode.chained_assignment = None


def create_ratings(vps, filepath, problematic_subjects, df_scores):
    df_ratings = pd.DataFrame()
    for vp in vps:
        # vp = vps[0]
        vp = f"0{vp}" if vp < 10 else f"{vp}"

        # Get rating data
        try:
            files = [item for item in os.listdir(os.path.join(filepath, 'VP_' + vp)) if (item.endswith(".csv"))]
            file = [file for file in files if "rating" in file][0]
            df_ratings_vp = pd.read_csv(os.path.join(filepath, 'VP_' + vp, file), sep=';', decimal='.')
        except:
            print("no ratings file")
            continue

        # Get conditions
        try:
            df_cond = pd.read_excel(os.path.join(filepath, 'Conditions.xlsx'), sheet_name="Conditions3", usecols=[0, 1, 2])
            df_cond.columns = ["VP", "Roles", "Rooms"]
            df_cond = df_cond.loc[df_cond["VP"] == int(vp)]

            df_roles = pd.read_excel(os.path.join(filepath, 'Conditions.xlsx'), sheet_name="Roles")
            df_roles = df_roles[["Character", int(df_cond["Roles"].item())]]
            df_roles = df_roles.rename(columns={int(df_cond["Roles"].item()): "Role"})

            df_rooms = pd.read_excel(os.path.join(filepath, 'Conditions.xlsx'), sheet_name="Rooms3")
            df_rooms = df_rooms[["Role", int(df_cond["Rooms"].item())]]
            df_rooms = df_rooms.rename(columns={int(df_cond["Rooms"].item()): "Rooms"})

            df_roles = df_roles.merge(df_rooms, on="Role")
        except:
            print("no conditions file")
            continue

        df_ratings_vp["criterion"] = df_ratings_vp["phase"]
        df_ratings_vp["phase"] = df_ratings_vp["phase"].str.replace("RoomRating ", "")

        df_ratings_vp.loc[df_ratings_vp["phase"].str.contains("Rating"), "phase"] = "Test"
        df_ratings_vp["criterion"] = df_ratings_vp["criterion"].str.replace("Rating", "")
        df_ratings_vp["criterion"] = df_ratings_vp["criterion"].str.replace("Sympathy", "Likeability")
        df_ratings_vp.loc[(df_ratings_vp["criterion"].str.contains("Orientation")) | (df_ratings_vp["criterion"].str.contains("Test")) | (df_ratings_vp["criterion"].str.contains("Habituation")), "criterion"] = ""
        df_ratings_vp = df_ratings_vp.merge(df_roles, left_on="rating", right_on="Rooms", how="left")
        df_ratings_vp["Character"] = pd.concat([df_ratings_vp.loc[df_ratings_vp["criterion"] == "", "Character"], df_ratings_vp.loc[~(df_ratings_vp["criterion"] == ""), "rating"]])
        df_ratings_vp = df_ratings_vp.drop(columns=["Role", "Rooms"])
        df_ratings_vp = df_ratings_vp.merge(df_roles, on="Character", how="left")
        df_ratings_vp = df_ratings_vp.drop(columns=["Rooms"])
        df_ratings_vp = df_ratings_vp.rename(columns={"Role": "condition"})
        df_ratings_vp.loc[df_ratings_vp['criterion'] == "", "criterion"] = "wellbeing"

        behav_friendly = df_ratings_vp.loc[(df_ratings_vp["condition"] == "friendly") & (df_ratings_vp["criterion"].str.contains("Behavior")), "value"].item()
        behav_unfriendly = df_ratings_vp.loc[(df_ratings_vp["condition"] == "unfriendly") & (df_ratings_vp["criterion"].str.contains("Behavior")), "value"].item()

        if behav_friendly-behav_unfriendly < 10:
            problematic_subjects.append(int(vp))
            print(f"VP {vp}: Rating Friendly: {behav_friendly}; Rating Unfriendly: {behav_unfriendly}")

        df_ratings_vp["VP"] = int(vp)
        df_ratings_vp = df_ratings_vp.rename(columns={"phase": "Phase", "rating": "Object", "condition": "Condition", "criterion": "Criterion", "value": "Value",})
        df_ratings_vp = df_ratings_vp[["VP", "Phase", "Object", "Condition", "Criterion", "Value"]]

        df_ratings = pd.concat([df_ratings, df_ratings_vp])

    # Add participant scores
    df_ratings = df_ratings.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                           'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                           'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                           'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                           'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
    df_ratings = df_ratings.drop(columns=['ID'])

    return df_ratings, problematic_subjects


if __name__ == '__main__':
    wave = 2
    dir_path = os.getcwd()
    filepath = os.path.join(dir_path, f'Data-Wave{wave}')

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

    df_ratings, problematic_subjects = create_ratings(vps, filepath, problematic_subjects, df_scores)

    df_ratings.to_csv(os.path.join(filepath, 'ratings.csv'), index=False, decimal='.', sep=';', encoding='utf-8-sig')

    print(f"Problematic Subject: {problematic_subjects}")
