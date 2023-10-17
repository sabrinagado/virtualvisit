# =============================================================================
# Scores
# source: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np

wave = 2
dir_path = os.getcwd()
start = 1
end = 64
vps = np.arange(start, end + 1)

if wave == 1:
    problematic_subjects = [1, 3, 12, 15, 19, 20, 23, 24, 31, 33, 41, 45, 46, 47]
elif wave == 2:
    problematic_subjects = []

vps = [vp for vp in vps if not vp in problematic_subjects]

for vp in vps:
    # vp = vps[0]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    # Get rating data
    try:
        files = [item for item in os.listdir(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp)) if (item.endswith(".csv"))]
        file = [file for file in files if "rating" in file][0]
        df_ratings = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp, file), sep=';', decimal='.')
    except:
        print("no ratings file")
        continue

    # Get conditions
    try:
        df_cond = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Conditions3")
        df_cond = df_cond[["VP", "Roles", "Rooms"]]
        df_cond = df_cond.loc[df_cond["VP"] == int(vp)]

        df_roles = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Roles")
        df_roles = df_roles[["Character", int(df_cond["Roles"].item())]]
        df_roles = df_roles.rename(columns={int(df_cond["Roles"].item()): "Role"})

        df_rooms = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Rooms3")
        df_rooms = df_rooms[["Role", int(df_cond["Rooms"].item())]]
        df_rooms = df_rooms.rename(columns={int(df_cond["Rooms"].item()): "Rooms"})

        df_roles = df_roles.merge(df_rooms, on="Role")
    except:
        print("no conditions file")
        continue

    df_ratings["criterion"] = df_ratings["phase"]
    df_ratings["phase"] = df_ratings["phase"].str.replace("RoomRating ", "")

    df_ratings.loc[df_ratings["phase"].str.contains("Rating"), "phase"] = "Test"
    df_ratings["criterion"] = df_ratings["criterion"].str.replace("Rating", "")
    df_ratings.loc[(df_ratings["criterion"].str.contains("Orientation")) | (df_ratings["criterion"].str.contains("Test")) | (df_ratings["criterion"].str.contains("Habituation")), "criterion"] = ""
    df_ratings = df_ratings.merge(df_roles, left_on="rating", right_on="Rooms", how="left")
    df_ratings["Character"] = pd.concat([df_ratings.loc[df_ratings["criterion"] == "", "Character"], df_ratings.loc[~(df_ratings["criterion"] == ""), "rating"]])
    df_ratings = df_ratings.drop(columns=["Role", "Rooms"])
    df_ratings = df_ratings.merge(df_roles, on="Character", how="left")
    df_ratings = df_ratings.drop(columns=["Rooms"])
    df_ratings = df_ratings.rename(columns={"Role": "condition"})
    df_ratings.loc[df_ratings['criterion'] == "", "criterion"] = "wellbeing"

    behav_friendly = df_ratings.loc[(df_ratings["condition"] == "friendly") & (df_ratings["criterion"].str.contains("Behavior")), "value"].item()
    behav_unfriendly = df_ratings.loc[(df_ratings["condition"] == "unfriendly") & (df_ratings["criterion"].str.contains("Behavior")), "value"].item()

    if behav_friendly-behav_unfriendly < 10:
        problematic_subjects.append(int(vp))
        print(f"Rating Friendly: {behav_friendly}; Rating Unfriendly: {behav_unfriendly}")
        continue

    for idx_row, row in df_ratings.iterrows():
        # idx_row = 0
        # row = df_ratings.iloc[idx_row]

        # Save as dataframe
        df_rating_temp = pd.DataFrame({'VP': [int(vp)],
                                       'Phase': [row['phase']],
                                       'Object': [row['rating']],
                                       'Condition': [row['condition']],
                                       'Criterion': [row['criterion']],
                                       'Value': [row['value']]})
        df_rating_temp.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'ratings.csv'), decimal='.', sep=';', index=False, mode='a',
                              header=not (os.path.exists(os.path.join(dir_path, f'Data-Wave{wave}', 'ratings.csv'))))

# Add Subject Data
df_rating = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'ratings.csv'), decimal='.', sep=';')
df_rating = df_rating.iloc[:, 0:6]

df_scores = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'scores_summary.csv'), decimal=',', sep=';')
df_rating = df_rating.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                       'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                       'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                       'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                       'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_rating = df_rating.drop(columns=['ID'])
df_rating.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'ratings.csv'), decimal='.', sep=';', index=False)

problematic_subjects = list(np.unique(problematic_subjects))
