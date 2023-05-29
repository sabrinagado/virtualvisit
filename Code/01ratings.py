# =============================================================================
# Scores
# source: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np

dir_path = os.getcwd()
start = 1
end = 11
vps = np.arange(start, end + 1)

for vp in vps:
    # vp = vps[1]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    # Get rating data
    try:
        files = [item for item in os.listdir(os.path.join(dir_path, 'Data', 'VP_' + vp)) if (item.endswith(".csv"))]
        file = [file for file in files if "rating" in file][0]
        df_ratings = pd.read_csv(os.path.join(dir_path, 'Data', 'VP_' + vp, file), sep=';', decimal='.')
    except:
        print("no ratings file")
        continue

    df_ratings["criterion"] = df_ratings["phase"]
    df_ratings["phase"] = df_ratings["phase"].str.replace("RoomRating ", "")

    df_ratings.loc[df_ratings["phase"].str.contains("Rating"), "phase"] = "Test"
    df_ratings["criterion"] = df_ratings["criterion"].str.replace("Rating", "")
    df_ratings.loc[(df_ratings["criterion"].str.contains("Orientation")) | (df_ratings["criterion"].str.contains("Test")) | (df_ratings["criterion"].str.contains("Habituation")), "criterion"] = ""

    for idx_row, row in df_ratings.iterrows():
        # idx_row = 0
        # row = df_ratings.iloc[idx_row]

        # Save as dataframe
        df_rating_temp = pd.DataFrame({'VP': [int(vp)],
                                       'Phase': [row['phase']],
                                       'Object': [row['rating']],
                                       'Condition': [""],
                                       'Criterion': [row['criterion']],
                                       'Value': [row['value']]})
        df_rating_temp.to_csv(os.path.join(dir_path, 'Data', 'ratings.csv'), decimal='.', sep=';', index=False, mode='a',
                              header=not (os.path.exists(os.path.join(dir_path, 'Data', 'ratings.csv'))))

# Add Subject Data
df_rating = pd.read_csv(os.path.join(dir_path, 'Data', 'ratings.csv'), decimal='.', sep=';')
df_rating = df_rating.iloc[:, 0:6]

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_rating = df_rating.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                               'SSQ', 'SSQ-N', 'SSQ-O', 'SSQ-D', 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS',
                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_rating = df_rating.drop(columns=['ID'])
df_rating.to_csv(os.path.join(dir_path, 'Data', 'ratings.csv'), decimal='.', sep=';', index=False)

