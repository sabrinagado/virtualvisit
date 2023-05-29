# =============================================================================
# Eye-Tracking / Gaze
# source: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

dir_path = os.getcwd()
start = 1
end = 11
vps = np.arange(start, end + 1)


def drop_consecutive_duplicates(df, subset, keep="first", times="timestamp", tolerance=0.1):
    if keep == "first":
        df = df.loc[(df[subset].shift(1) != df[subset]) | ((df[times] - df[times].shift(1)).dt.total_seconds() >= tolerance)]
    elif keep == "last":
        df = df.loc[(df[subset].shift(-1) != df[subset]) | ((df[times].shift(-1) - df[times]).dt.total_seconds() >= tolerance)]
    return df


for vp in vps:
    # vp = vps[4]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    try:
        files = [item for item in os.listdir(os.path.join(dir_path, 'Data', 'VP_' + vp)) if (item.endswith(".csv"))]
        file = [file for file in files if "gaze" in file][0]
        df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'VP_' + vp, file), sep=';', decimal='.')
    except:
        print("no gaze file")
        continue

    # Adapt timestamps
    if pd.to_datetime(df_gaze.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
        df_gaze["timestamp"] = pd.to_datetime(df_gaze["timestamp"]) + timedelta(hours=2)
    else:
        df_gaze["timestamp"] = pd.to_datetime(df_gaze["timestamp"]) + timedelta(hours=1)
    df_gaze["timestamp"] = df_gaze["timestamp"].apply(lambda t: t.replace(tzinfo=None))

    # Get Events
    files = [item for item in os.listdir(os.path.join(dir_path, 'Data', 'VP_' + vp)) if (item.endswith(".csv"))]
    event_file = [file for file in files if "event" in file][0]
    df_event = pd.read_csv(os.path.join(dir_path, 'Data', 'VP_' + vp, event_file), sep=';', decimal='.')

    if pd.to_datetime(df_event.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
        df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=2)
    else:
        df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=1)
    df_event["timestamp"] = df_event["timestamp"].apply(lambda t: t.replace(tzinfo=None))

    df_event = drop_consecutive_duplicates(df_event, subset="event", keep="first", times="timestamp", tolerance=0.1)
    df_event = df_event.reset_index(drop=True)

    start_roomtour = df_event.loc[df_event["event"] == "StartRoomTour", "timestamp"].item()
    start_habituation = df_event.loc[df_event["event"] == "StartExploringRooms", "timestamp"].item()
    start_roomrating1 = df_event.loc[df_event["event"] == "EndExploringRooms", "timestamp"].item()
    start_conditioning = df_event.loc[df_event["event"] == "EnterTerrace", "timestamp"].reset_index(drop=True)[0]
    start_test = df_event.loc[df_event["event"] == "AllInteractionsFinished", "timestamp"].reset_index(drop=True)[0]
    start_roomrating2 = df_event.loc[df_event["event"] == "EndExploringRooms2", "timestamp"].item()
    start_personrating = df_event.loc[df_event["event"] == "TeleportToStartingRoom", "timestamp"].item()
    end = df_event.loc[df_event["event"] == "End", "timestamp"].item()

    dfs = []
    df_hab = df_event.loc[(start_habituation <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating1)]
    df_hab["duration"] = (df_hab["timestamp"].shift(-1) - df_hab["timestamp"]).dt.total_seconds()
    df_hab = df_hab.loc[df_hab["event"].str.contains("Enter")]
    df_hab["event"] = ["Habituation_" + name[1] for name in df_hab["event"].str.split("Enter")]
    dfs.append(df_hab)

    df_acq = df_event.loc[(start_conditioning <= df_event["timestamp"]) & (df_event["timestamp"] < start_test)]
    df_acq = df_acq.loc[df_acq["event"].str.contains("Interaction")]
    df_acq["duration"] = 5
    df_acq["event"] = [name[1] for name in df_acq["event"].str.split("Start")]
    df_acq.loc[df_acq["event"].str.contains("Unfiendly"), "event"] = "UnfriendlyInteraction"
    dfs.append(df_acq)

    df_test = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
    df_test = df_test.loc[~(df_test["event"].str.contains("Teleport"))]
    df_test["duration"] = (df_test["timestamp"].shift(-1) - df_test["timestamp"]).dt.total_seconds()
    df_test = df_test.loc[df_test["event"].str.contains("Enter")]
    df_test["event"] = ["Test_" + name[1] for name in df_test["event"].str.split("Enter")]
    dfs.append(df_test)

    df_test_person = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
    df_test_person = df_test_person.loc[(df_test_person["event"].str.contains("Clicked"))]
    if len(df_test_person) > 0:
        df_test_person["duration"] = 2
        df_test_person["event"] = ["Test_" + name for name in df_test_person["event"]]
        dfs.append(df_test_person)

    df_event = pd.concat(dfs)
    df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)

    # Merge "event"-column to df_gaze
    df_gaze = pd.merge_asof(df_gaze, df_event[["timestamp", "event"]], on="timestamp", direction="backward").reset_index(drop=True)

    # Iterate through experimental phases and check ECG data
    for idx_row, row in df_event.iterrows():
        # idx_row = 4
        # row = df_event.iloc[idx_row]
        phase = row['event']
        print(f"Phase: {phase}")

        # Get start and end point of phase
        start_phase = row['timestamp']
        end_phase = row['timestamp'] + pd.to_timedelta(row['duration'], unit="S")

        # Cut gaze dataset
        df_gaze_subset = df_gaze.loc[(df_gaze["timestamp"] >= start_phase) & (df_gaze["timestamp"] < end_phase + timedelta(seconds=1))]
        df_gaze_subset = df_gaze_subset.loc[df_gaze_subset['event'] == phase].reset_index(drop=True)

        df_gaze_subset = df_gaze_subset.loc[df_gaze_subset["eye_openness"] == 1]

        # Pupil: Get Mean and Std
        # Save as dataframe
        df_pupil_temp = pd.DataFrame({'VP': [int(vp)],
                                      'Phase': [phase],
                                      'Pupil Dilation Right (Mean)': [df_gaze_subset['pupil_right'].mean()],
                                      'Pupil Dilation Right (Std)': [df_gaze_subset['pupil_right'].std()],
                                      'Pupil Dilation Left (Mean)': [df_gaze_subset['pupil_left'].mean()],
                                      'Pupil Dilation Left (Std)': [df_gaze_subset['pupil_left'].std()],
                                      'HR (Mean)': [df_gaze_subset['hr'].mean()],
                                      'HR (Std)': [df_gaze_subset['hr'].std()],
                                      'Duration': [(end_phase - start_phase).total_seconds()]})
        df_pupil_temp.to_csv(os.path.join(dir_path, 'Data', 'pupil.csv'), decimal='.', sep=';', index=False, mode='a',
                             header=not (os.path.exists(os.path.join(dir_path, 'Data', 'pupil.csv'))))
        
        if not "Habituation" in phase:
            for character in ["Bryan", "Emanuel", "Ettore", "Oskar"]:
                # character = "Ettore"
                for roi, searchstring in zip(["head", "body"], ["Head", "_Char"]):
                    # roi = "body"
                    # searchstring = "Head"
                    number = len(df_gaze_subset.loc[df_gaze_subset['actor'].str.contains(f"{character}{searchstring}")])
                    proportion = number / len(df_gaze_subset)

                    # Save as dataframe
                    df_gaze_temp = pd.DataFrame({'VP': [int(vp)],
                                                 'Phase': [phase],
                                                 'Person': [character],
                                                 'Condition': [""],
                                                 'ROI': [roi],
                                                 'Gaze Proportion': [proportion],
                                                 'Number': [number]})
                    df_gaze_temp.to_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';', index=False, mode='a',
                                         header=not (os.path.exists(os.path.join(dir_path, 'Data', 'gaze.csv'))))

# Add Subject Data
df_gaze = pd.read_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';')
df_gaze = df_gaze.iloc[:, 0:7]
df_gaze = df_gaze.loc[df_gaze['Number'] > 0]

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_gaze = df_gaze.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                               'SSQ', 'SSQ-N', 'SSQ-O', 'SSQ-D', 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS',
                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_gaze = df_gaze.drop(columns=['ID'])
df_gaze.to_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';', index=False)


# Add Subject Data
df_pupil = pd.read_csv(os.path.join(dir_path, 'Data', 'pupil.csv'), decimal='.', sep=';')
df_pupil = df_pupil.iloc[:, 0:7]
df_pupil.loc[df_pupil["HR (Mean)"] == 0, "HR (Mean)"] = np.nan

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_pupil = df_pupil.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                     'SSQ', 'SSQ-N', 'SSQ-O', 'SSQ-D', 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS',
                                     'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                     'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_pupil = df_pupil.drop(columns=['ID'])
df_pupil.to_csv(os.path.join(dir_path, 'Data', 'pupil.csv'), decimal='.', sep=';', index=False)
