# =============================================================================
# Eye-Tracking / Gaze
# source: HMD & Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import signal
from Code.toolbox import utils

dir_path = os.getcwd()
start = 1
end = 64
vps = np.arange(start, end + 1)

problematic_subjects = [1, 3, 12, 15, 19, 20, 23, 24, 31, 33, 41, 42, 45, 46, 47, 53]
vps = [vp for vp in vps if not vp in problematic_subjects]


def drop_consecutive_duplicates(df, subset, keep="first", times="timestamp", tolerance=0.1):
    if keep == "first":
        df = df.loc[(df[subset].shift(1) != df[subset]) | ((df[times] - df[times].shift(1)).dt.total_seconds() >= tolerance)]
    elif keep == "last":
        df = df.loc[(df[subset].shift(-1) != df[subset]) | ((df[times].shift(-1) - df[times]).dt.total_seconds() >= tolerance)]
    return df


for vp in vps:
    # vp = vps[1]
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

    try:
        start_roomtour = df_event.loc[df_event["event"] == "StartRoomTour", "timestamp"].item()
        start_habituation = df_event.loc[df_event["event"] == "StartExploringRooms", "timestamp"].item()
        start_roomrating1 = df_event.loc[df_event["event"] == "EndExploringRooms", "timestamp"].item()
        start_conditioning = df_event.loc[df_event["event"] == "EnterTerrace", "timestamp"].reset_index(drop=True)[0]
        start_test = df_event.loc[df_event["event"] == "AllInteractionsFinished", "timestamp"].reset_index(drop=True)[0]
        start_roomrating2 = df_event.loc[df_event["event"] == "EndExploringRooms2", "timestamp"].item()
        start_personrating = df_event.loc[df_event["event"] == "TeleportToStartingRoom", "timestamp"].item()
        end = df_event.loc[df_event["event"] == "End", "timestamp"].item()
    except:
        print("not enough events")
        continue

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
    df_test = df_test.loc[df_test["event"].str.contains("Enter")]
    df_test["duration"] = (df_test["timestamp"].shift(-1) - df_test["timestamp"]).dt.total_seconds()
    df_test = df_test.reset_index(drop=True)
    df_test.loc[len(df_test) - 1, "duration"] = (
                start_roomrating2 - df_test.loc[len(df_test) - 1, "timestamp"]).total_seconds()
    df_test["event"] = ["Test_" + name[1] for name in df_test["event"].str.split("Enter")]
    dfs.append(df_test)

    df_test_person = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
    df_test_person = df_test_person.loc[(df_test_person["event"].str.contains("Clicked"))]
    if len(df_test_person) > 0:
        df_test_person["duration"] = 2
        df_test_person["event"] = ["Test_" + name for name in df_test_person["event"]]
        for person in list(df_test_person["event"].unique()):
            # person = list(df_test_person["event"].unique())[1]
            df_test_person_unique = df_test_person.loc[df_test_person["event"] == person].reset_index(drop=True)
            df_test_person_unique = drop_consecutive_duplicates(df_test_person_unique, subset="event", tolerance=2.1)
            dfs.append(df_test_person_unique)

    df_event = pd.concat(dfs)
    df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)

    # Merge "event"-column to df_gaze
    df_gaze = pd.merge_asof(df_gaze, df_event[["timestamp", "event"]], on="timestamp", direction="backward").reset_index(drop=True)

    # Iterate through experimental phases
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
        df_gaze_subset = df_gaze_subset.loc[df_gaze_subset['event'] == phase]

        df_gaze_subset = df_gaze_subset.loc[df_gaze_subset["eye_openness"] == 1]
        df_gaze_subset = df_gaze_subset.reset_index(drop=True)

        df_gaze_subset.loc[df_gaze_subset["pupil_left"] == -1, "pupil_left"] = np.nan
        df_gaze_subset.loc[df_gaze_subset["pupil_right"] == -1, "pupil_right"] = np.nan
        df_gaze_subset["pupil_mean"] = df_gaze_subset[["pupil_left", "pupil_right"]].mean(axis=1)

        if "Interaction" in phase:
            df_gaze_subset_save = df_gaze_subset.copy()
            df_gaze_subset_save[f"Impact_X"] = [float(element[1].split(" Y=")[0]) for element in df_gaze_subset_save['impact_point'].str.split(f"X=")]
            df_gaze_subset_save[f"Impact_Y"] = [float(element[1].split(" Z=")[0]) for element in df_gaze_subset_save['impact_point'].str.split(f"Y=")]
            df_gaze_subset_save[f"Impact_Z"] = [float(element[1]) for element in df_gaze_subset_save['impact_point'].str.split(f"Z=")]

            df_gaze_subset_save[f"Gaze_X"] = [float(element[1].split(" Y=")[0]) for element in df_gaze_subset_save['gaze_combined'].str.split(f"X=")]
            df_gaze_subset_save[f"Gaze_Y"] = [float(element[1].split(" Z=")[0]) for element in df_gaze_subset_save['gaze_combined'].str.split(f"Y=")]
            df_gaze_subset_save[f"Gaze_Z"] = [float(element[1]) for element in df_gaze_subset_save['gaze_combined'].str.split(f"Z=")]

            df_gaze_subset_save = drop_consecutive_duplicates(df_gaze_subset_save, subset="timestamp", keep="first")
            start = df_gaze_subset_save.loc[0, "timestamp"]
            df_gaze_subset_save["time"] = pd.to_timedelta(df_gaze_subset_save["timestamp"] - start)

            # 2 Hz low-pass butterworth filter
            timestamps = np.array(df_gaze_subset_save["time"].dt.total_seconds()*1000)
            sr, fs = utils.get_sampling_rate(timestamps)
            pupil_signal = df_gaze_subset_save["pupil_mean"].to_numpy()
            rolloff = 12
            lpfreq = 2
            pupil_filtered = np.concatenate((np.repeat(pupil_signal[0], 100), pupil_signal, np.repeat(pupil_signal[-1], 100)))  # zero padding
            pupil_filtered[np.isnan(pupil_filtered)] = np.nanmean(pupil_filtered)
            b, a = signal.butter(int(rolloff / 6), lpfreq * (1 / (sr / 2)))  # low-pass filter
            pupil_filtered = signal.filtfilt(b, a, pupil_filtered)  # apply filter
            pupil_filtered = pupil_filtered[100:-100]
            df_gaze_subset_save["pupil"] = pupil_filtered

            start_pupil = df_gaze_subset_save.loc[0, "pupil"]
            df_gaze_subset_save["pupil"] = df_gaze_subset_save["pupil"] - start_pupil

            df_gaze_subset_save = df_gaze_subset_save.set_index("time")
            df_gaze_subset_save = df_gaze_subset_save.resample("0.1S").mean()
            df_gaze_subset_save = df_gaze_subset_save.reset_index()
            df_gaze_subset_save["time"] = df_gaze_subset_save["time"].dt.total_seconds()
            df_gaze_subset_save["VP"] = int(vp)
            df_gaze_subset_save["event"] = phase
            df_gaze_subset_save = df_gaze_subset_save[["VP", "event", "time", "Impact_X", "Impact_Y", "Impact_Z",
                                                       "Gaze_X", "Gaze_Y", "Gaze_Z", "pupil"]]
            df_gaze_subset_save.to_csv(os.path.join(dir_path, 'Data', 'pupil_interaction.csv'), decimal='.', sep=';', index=False,
                                 mode='a', header=not (os.path.exists(os.path.join(dir_path, 'Data', 'pupil_interaction.csv'))))

        # Pupil: Get Mean and Std
        # Save as dataframe
        df_pupil_temp = pd.DataFrame({'VP': [int(vp)],
                                      'Phase': [phase],
                                      'Pupil Dilation (Mean)': [df_gaze_subset['pupil_mean'].mean()],
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
                # character = "Emanuel"
                for roi, searchstring in zip(["head", "body"], ["Head", "_Char"]):
                    # roi = "head"
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

# Get conditions
dfs_gaze = []
for vp in vps:
    # vp = vps[10]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    df_gaze_vp = df_gaze.loc[df_gaze["VP"] == int(vp)]

    try:
        df_cond = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Conditions3")
        df_cond = df_cond[["VP", "Roles", "Rooms"]]
        df_cond = df_cond.loc[df_cond["VP"] == int(vp)]

        df_roles = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Roles")
        df_roles = df_roles[["Character", int(df_cond["Roles"].item())]]
        df_roles = df_roles.rename(columns={int(df_cond["Roles"].item()): "Role"})

        df_rooms = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Rooms3")
        df_rooms = df_rooms[["Role", int(df_cond["Rooms"].item())]]
        df_rooms = df_rooms.rename(columns={int(df_cond["Rooms"].item()): "Rooms"})

        df_roles = df_roles.merge(df_rooms, on="Role")
    except:
        print("no conditions file")

    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        character = row["Character"]
        role = row["Role"]
        df_gaze_vp["Phase"] = [event.replace(character, role.capitalize()) for event in df_gaze_vp.loc[:, "Phase"]]
    df_gaze_vp = df_gaze_vp.merge(df_roles, left_on="Person", right_on="Character", how="left")
    df_gaze_vp["Condition"] = df_gaze_vp["Role"]
    df_gaze_vp = df_gaze_vp.drop(columns=["Character", "Rooms", "Role"])

    dfs_gaze.append(df_gaze_vp)

df_gaze = pd.concat(dfs_gaze)

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_gaze = df_gaze.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                   'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                   'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                   'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                   'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_gaze = df_gaze.drop(columns=['ID'])
df_gaze.to_csv(os.path.join(dir_path, 'Data', 'gaze.csv'), decimal='.', sep=';', index=False)


# Add Subject Data
df_pupil = pd.read_csv(os.path.join(dir_path, 'Data', 'pupil.csv'), decimal='.', sep=';')
df_pupil = df_pupil.iloc[:, 0:7]

# Get conditions
dfs_pupil = []
for vp in vps:
    # vp = vps[10]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    df_pupil_vp = df_pupil.loc[df_pupil["VP"] == int(vp)]

    try:
        df_cond = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Conditions3")
        df_cond = df_cond[["VP", "Roles", "Rooms"]]
        df_cond = df_cond.loc[df_cond["VP"] == int(vp)]

        df_roles = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Roles")
        df_roles = df_roles[["Character", int(df_cond["Roles"].item())]]
        df_roles = df_roles.rename(columns={int(df_cond["Roles"].item()): "Role"})

        df_rooms = pd.read_excel(os.path.join(dir_path, 'Data', 'Conditions.xlsx'), sheet_name="Rooms3")
        df_rooms = df_rooms[["Role", int(df_cond["Rooms"].item())]]
        df_rooms = df_rooms.rename(columns={int(df_cond["Rooms"].item()): "Rooms"})

        df_roles = df_roles.merge(df_rooms, on="Role")
    except:
        print("no conditions file")

    df_pupil_vp["Condition"] = ""
    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        room = row["Rooms"]
        role = row["Role"]
        df_pupil_vp.loc[df_pupil_vp["Phase"].str.contains(room), "Condition"] = role
    dfs_pupil.append(df_pupil_vp)

df_pupil = pd.concat(dfs_pupil)

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_pupil = df_pupil.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                     'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                     'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                     'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                     'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_pupil = df_pupil.drop(columns=['ID'])
df_pupil.to_csv(os.path.join(dir_path, 'Data', 'pupil.csv'), decimal='.', sep=';', index=False)
