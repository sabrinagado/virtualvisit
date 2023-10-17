# =============================================================================
# Scores
# source: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math


def drop_consecutive_duplicates(df, subset, keep="first", times="timestamp", tolerance=0.1):
    if keep == "first":
        df = df.loc[(df[subset].shift(1) != df[subset]) | ((df[times] - df[times].shift(1)).dt.total_seconds() >= tolerance)]
    elif keep == "last":
        df = df.loc[(df[subset].shift(-1) != df[subset]) | ((df[times].shift(-1) - df[times]).dt.total_seconds() >= tolerance)]
    return df


def distance(X, Y):
    N = len(X)
    T = 0
    oldx, oldy = X[-1], Y[-1]
    for x, y in zip(X, Y):
        T += np.linalg.norm((np.array([x, y])-np.array([oldx, oldy])))
        oldx = x
        oldy = y
    return T


wave = 2
dir_path = os.getcwd()
start = 1
end = 64
vps = np.arange(start, end + 1)

if wave == 1:
    problematic_subjects = [1, 3, 12, 15, 19, 20, 23, 24, 31, 33, 41, 42, 45, 46, 47, 53]
elif wave == 2:
    problematic_subjects = []

dir_path = os.getcwd()
start = 1
end = 64
vps = np.arange(start, end + 1)

vps = [vp for vp in vps if not vp in problematic_subjects]

# Durations and Clicks
for vp in vps:
    # vp = vps[26]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    # Get Events
    try:
        files = [item for item in os.listdir(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp)) if (item.endswith(".csv"))]
        event_file = [file for file in files if "event" in file][0]
        df_event = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp, event_file), sep=';', decimal='.')

        if pd.to_datetime(df_event.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
            df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=2)
        else:
            df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=1)
        df_event["timestamp"] = df_event["timestamp"].apply(lambda t: t.replace(tzinfo=None))

        df_event = drop_consecutive_duplicates(df_event, subset="event", keep="first", times="timestamp", tolerance=0.1)
        df_event = df_event.reset_index(drop=True)
    except:
        print("no events file")
        continue

    try:
        start_roomtour = df_event.loc[df_event["event"] == "StartRoomTour", "timestamp"].item()
        start_habituation = df_event.loc[df_event["event"] == "StartExploringRooms", "timestamp"].item()
        start_roomrating1 = df_event.loc[df_event["event"] == "EndExploringRooms", "timestamp"].item()
        start_conditioning = df_event.loc[df_event["event"] == "EnterTerrace", "timestamp"].reset_index(drop=True)[0]
        start_roomrating2 = df_event.loc[df_event["event"] == "EndExploringRooms2", "timestamp"].item()
        start_test = start_roomrating2 - timedelta(seconds=180)
        end_acq = df_event.loc[(df_event["event"] == "EnterOffice") & (df_event["timestamp"] > start_conditioning) & (
                    df_event["timestamp"] < start_test), "timestamp"].reset_index(drop=True)[0]
        start_personrating = df_event.loc[df_event["event"] == "TeleportToStartingRoom", "timestamp"].item()
        end = df_event.loc[df_event["event"] == "End", "timestamp"].item()
    except:
        print("not enough events")
        continue

    dfs = []
    df_hab = df_event.loc[(start_habituation <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating1)]
    df_hab["duration"] = (df_hab["timestamp"].shift(-1) - df_hab["timestamp"]).dt.total_seconds()
    df_hab["event"] = df_hab["event"].replace("StartExploringRooms", "EnterOffice")
    df_hab = df_hab.loc[df_hab["event"].str.contains("Enter")]
    df_hab["event"] = ["Habituation_" + name[1] for name in df_hab["event"].str.split("Enter")]
    dfs.append(df_hab)

    df_acq = df_event.loc[(start_conditioning <= df_event["timestamp"]) & (df_event["timestamp"] < end_acq)]
    df_acq = df_acq.loc[(df_acq["event"].str.contains("Interaction")) & ~(df_acq["event"].str.contains("Finished"))]
    df_acq["duration"] = 5
    df_acq["event"] = [name[1] for name in df_acq["event"].str.split("Start")]
    df_acq = df_acq.drop_duplicates(subset="event")
    dfs.append(df_acq)

    df_test = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
    df_test = pd.concat([df_test, pd.DataFrame({"timestamp": [start_test], "event": "EnterOffice"})])
    df_test = df_test.sort_values(by="timestamp")
    df_test = df_test.loc[
        (df_test["event"].str.contains("Enter")) | (df_test["event"].str.contains("Clicked"))].reset_index(drop=True)
    room = ""
    for idx_row, row in df_test.iterrows():
        # idx_row = 0
        # row = df_test.iloc[idx_row, :]
        if "Enter" in row["event"]:
            room = row["event"]
        elif "Clicked" in row["event"]:
            df_test = pd.concat(
                [df_test, pd.DataFrame({"timestamp": [row["timestamp"] + timedelta(seconds=3)], "event": [room]})])
    df_test = df_test.sort_values(by="timestamp").reset_index(drop=True)
    df_test = drop_consecutive_duplicates(df_test, subset="event", keep="first", times="timestamp", tolerance=0.1)
    df_test = df_test.reset_index(drop=True)
    df_test["duration"] = (df_test["timestamp"].shift(-1) - df_test["timestamp"]).dt.total_seconds()
    df_test.loc[len(df_test) - 1, "duration"] = (
                start_roomrating2 - df_test.loc[len(df_test) - 1, "timestamp"]).total_seconds()
    df_test = df_test.loc[df_test["event"].str.contains("Enter")]
    df_test["event"] = ["Test_" + name[1] for name in df_test["event"].str.split("Enter")]
    dfs.append(df_test)

    df_test_person = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
    df_test_person = pd.concat([df_test_person, pd.DataFrame({"timestamp": [start_test], "event": "EnterOffice"})])
    df_test_person = df_test_person.sort_values(by="timestamp").reset_index(drop=True)
    df_test_person = df_test_person.loc[(df_test_person["event"].str.contains("Clicked"))]
    if len(df_test_person) > 0:
        df_test_person["duration"] = 3
        df_test_person["event"] = ["Test_" + name for name in df_test_person["event"]]
        for person in list(df_test_person["event"].unique()):
            # person = list(df_test_person["event"].unique())[1]
            df_test_person_unique = df_test_person.loc[df_test_person["event"] == person].reset_index(drop=True)
            df_test_person_unique = drop_consecutive_duplicates(df_test_person_unique, subset="event", tolerance=2.1)
            dfs.append(df_test_person_unique)

    df_event = pd.concat(dfs)
    df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)

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

    df_event["Condition"] = ""
    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        room = row["Rooms"]
        role = row["Role"]
        character = row["Character"]
        df_event.loc[df_event["event"].str.contains(room), "Condition"] = role
        df_event.loc[df_event["event"].str.contains(character), "Condition"] = role
    df_event["VP"] = int(vp)
    df_event = df_event[["VP", "timestamp", "event", "Condition", "duration"]]

    df_event.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';', index=False, mode='a',
                    header=not (os.path.exists(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'))))

# Add Subject Data
df_events = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';')
df_events = df_events.iloc[:, 0:5]

df_scores = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'scores_summary.csv'), decimal=',', sep=';')
df_events = df_events.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                       'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                       'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                       'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                       'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_events = df_events.drop(columns=['ID'])
df_events.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';', index=False)

# Movement
df_distance = pd.DataFrame()
for vp in vps:
    # vp = vps[0]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    # Get Movement File
    try:
        files = [item for item in os.listdir(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp)) if (item.endswith(".csv"))]
        file = [file for file in files if "movement" in file][0]
        df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp, file), sep=';', decimal='.')
        if pd.to_datetime(df.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
            df["timestamp"] = pd.to_datetime(df["timestamp"]) + timedelta(hours=2)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"]) + timedelta(hours=1)
        df["timestamp"] = df["timestamp"].apply(lambda t: t.replace(tzinfo=None))
    except:
        print("no movement file")
        continue

    # Get Events
    try:
        files = [item for item in os.listdir(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp)) if (item.endswith(".csv"))]
        event_file = [file for file in files if "event" in file][0]
        df_event = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp, event_file), sep=';', decimal='.')

        if pd.to_datetime(df_event.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
            df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=2)
        else:
            df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=1)
        df_event["timestamp"] = df_event["timestamp"].apply(lambda t: t.replace(tzinfo=None))
    except:
        print("no events file")
        continue

    try:
        start_roomtour = df_event.loc[df_event["event"] == "StartRoomTour", "timestamp"].item()
        start_habituation = df_event.loc[df_event["event"] == "StartExploringRooms", "timestamp"].item()
        start_roomrating1 = df_event.loc[df_event["event"] == "EndExploringRooms", "timestamp"].item()
        start_conditioning = df_event.loc[df_event["event"] == "EnterTerrace", "timestamp"].reset_index(drop=True)[0]
        start_roomrating2 = df_event.loc[df_event["event"] == "EndExploringRooms2", "timestamp"].item()
        start_test = start_roomrating2 - timedelta(seconds=180)
        end_acq = df_event.loc[(df_event["event"] == "EnterOffice") & (df_event["timestamp"] > start_conditioning) & (
                    df_event["timestamp"] < start_test), "timestamp"].reset_index(drop=True)[0]
        start_personrating = df_event.loc[df_event["event"] == "TeleportToStartingRoom", "timestamp"].item()
        end = df_event.loc[df_event["event"] == "End", "timestamp"].item()
    except:
        print("not enough events")
        continue

    dfs = []
    df_hab = df_event.loc[(start_habituation <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating1)]
    df_hab["duration"] = (df_hab["timestamp"].shift(-1) - df_hab["timestamp"]).dt.total_seconds()
    df_hab["event"] = df_hab["event"].replace("StartExploringRooms", "EnterOffice")
    df_hab = df_hab.loc[df_hab["event"].str.contains("Enter")]
    df_hab["event"] = ["Habituation_" + name[1] for name in df_hab["event"].str.split("Enter")]
    dfs.append(df_hab)

    df_acq = df_event.loc[(start_conditioning <= df_event["timestamp"]) & (df_event["timestamp"] < end_acq)]
    df_acq = df_acq.loc[(df_acq["event"].str.contains("Interaction")) & ~(df_acq["event"].str.contains("Finished"))]
    df_acq["duration"] = 5
    df_acq["event"] = [name[1] for name in df_acq["event"].str.split("Start")]
    df_acq = df_acq.drop_duplicates(subset="event")
    dfs.append(df_acq)

    df_test = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
    df_test = pd.concat([df_test, pd.DataFrame({"timestamp": [start_test], "event": "EnterOffice"})])
    df_test = df_test.sort_values(by="timestamp")
    df_test = df_test.loc[
        (df_test["event"].str.contains("Enter")) | (df_test["event"].str.contains("Clicked"))].reset_index(drop=True)
    room = ""
    for idx_row, row in df_test.iterrows():
        # idx_row = 0
        # row = df_test.iloc[idx_row, :]
        if "Enter" in row["event"]:
            room = row["event"]
        elif "Clicked" in row["event"]:
            df_test = pd.concat(
                [df_test, pd.DataFrame({"timestamp": [row["timestamp"] + timedelta(seconds=3)], "event": [room]})])
    df_test = df_test.sort_values(by="timestamp").reset_index(drop=True)
    df_test = drop_consecutive_duplicates(df_test, subset="event", keep="first", times="timestamp", tolerance=0.1)
    df_test = df_test.reset_index(drop=True)
    df_test["duration"] = (df_test["timestamp"].shift(-1) - df_test["timestamp"]).dt.total_seconds()
    df_test.loc[len(df_test) - 1, "duration"] = (
                start_roomrating2 - df_test.loc[len(df_test) - 1, "timestamp"]).total_seconds()
    df_test = df_test.loc[df_test["event"].str.contains("Enter")]
    df_test["event"] = ["Test_" + name[1] for name in df_test["event"].str.split("Enter")]
    dfs.append(df_test)

    df_test_person = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
    df_test_person = pd.concat([df_test_person, pd.DataFrame({"timestamp": [start_test], "event": "EnterOffice"})])
    df_test_person = df_test_person.sort_values(by="timestamp").reset_index(drop=True)
    df_test_person = df_test_person.loc[(df_test_person["event"].str.contains("Clicked"))]
    df_test_person = drop_consecutive_duplicates(df_test_person, subset="event", keep="first", times="timestamp", tolerance=0.1)
    df_test_person = df_test_person.reset_index(drop=True)
    if len(df_test_person) > 0:
        df_test_person["duration"] = 3
        df_test_person["event"] = ["Test_" + name for name in df_test_person["event"]]
        for person in list(df_test_person["event"].unique()):
            # person = list(df_test_person["event"].unique())[1]
            df_test_person_unique = df_test_person.loc[df_test_person["event"] == person].reset_index(drop=True)
            df_test_person_unique = drop_consecutive_duplicates(df_test_person_unique, subset="event", tolerance=2.1)
            dfs.append(df_test_person_unique)

    df_event = pd.concat(dfs)
    df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)
    df_event = df_event.loc[df_event["duration"] > 0]

    # Add "event" columns (for MNE)
    df_event["name"] = df_event["event"]
    df_event.loc[df_event['name'].str.contains("resting state"), 'event'] = 1
    df_event.loc[df_event['name'].str.contains("Habituation_Living"), 'event'] = 2
    df_event.loc[df_event['name'].str.contains("Habituation_Dining"), 'event'] = 3
    df_event.loc[df_event['name'].str.contains("Habituation_Office"), 'event'] = 4
    df_event.loc[df_event['name'].str.contains("Test_Living"), 'event'] = 12
    df_event.loc[df_event['name'].str.contains("Test_Dining"), 'event'] = 13
    df_event.loc[df_event['name'].str.contains("Test_Office"), 'event'] = 14
    df_event.loc[df_event['name'].str.contains("Test_Terrace"), 'event'] = 15
    df_event.loc[df_event['name'].str.contains("Unfriendly"), 'event'] = 21
    df_event.loc[df_event['name'].str.contains("Friendly"), 'event'] = 22
    df_event.loc[df_event['name'].str.contains("Neutral"), 'event'] = 23
    df_event.loc[df_event['name'].str.contains("Test_BryanWasClicked"), 'event'] = 31
    df_event.loc[df_event['name'].str.contains("Test_EmanuelWasClicked"), 'event'] = 32
    df_event.loc[df_event['name'].str.contains("Test_EttoreWasClicked"), 'event'] = 33
    df_event.loc[df_event['name'].str.contains("Test_OskarWasClicked"), 'event'] = 34

    # Merge "name" and "event"-column to df
    df = pd.merge_asof(df, df_event[["timestamp", "name"]], on="timestamp", direction="backward").reset_index(drop=True)

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

    df = df.rename(columns={"name": "event"})
    df = df.dropna(subset="event")
    df["Condition"] = ""
    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        room = row["Rooms"]
        role = row["Role"]
        character = row["Character"]
        df.loc[df["event"].str.contains(room), "Condition"] = role
        df.loc[df["event"].str.contains(character), "Condition"] = role
    df["VP"] = int(vp)

    df.loc[(start_habituation <= df["timestamp"]) & (df["timestamp"] < start_roomrating1), "phase"] = "Habituation"
    df.loc[(start_test <= df["timestamp"]) & (df["timestamp"] < start_roomrating2), "phase"] = "Test"

    df["x"] = [float(position.split("=")[1].split(" ")[0]) for position in df["Position"]]
    df["y"] = [float(position.split("=")[2].split(" ")[0]) for position in df["Position"]]

    df = df[["VP", "timestamp", "event", "Condition", "phase", "x", "y"]].reset_index(drop=True)
    df = df.sort_values(by="timestamp")
    df = df.dropna(subset="phase")

    for phase in df["phase"].unique():
        # phase = "Habituation"
        df_phase = df.loc[df["phase"] == phase].reset_index(drop=True)

        start_position = df_phase.drop_duplicates("VP", keep="first")[["x", "y"]].to_numpy().flatten()
        df_phase["distance_from_start"] = df_phase.apply(lambda x: math.dist(x[["x", "y"]].to_numpy().flatten(), start_position), axis=1)
        df_distance_temp = pd.DataFrame({"VP": [vp],
                                         "phase": [phase],
                                         "walking_distance": [distance(df_phase["x"].to_numpy(), df_phase["y"].to_numpy()) / 100],
                                         "average_distance_to_start": [df_phase["distance_from_start"].mean() / 100],
                                         "maximum_distance_to_start": [df_phase["distance_from_start"].max() / 100]})
        df_distance = pd.concat([df_distance, df_distance_temp])

        start = df_phase.loc[0, "timestamp"]
        df_phase["time"] = pd.to_timedelta(df_phase["timestamp"] - start)
        df_phase = df_phase.set_index("time")
        df_phase = df_phase.resample("0.2S").mean(numeric_only=True)
        df_phase = df_phase.reset_index()

        movement_index = []
        for idx_row, row in df_phase.iterrows():
            # idx_row = 600
            # row = df.iloc[idx_row, :]
            if idx_row == 0:
                movement_index.append(0.)
                continue
            distance_to_previous = []
            point = df_phase.loc[idx_row, ["x", "y"]].to_numpy()

            if idx_row == 1:
                previous_point = df_phase.loc[idx_row - 1, ["x", "y"]].to_numpy()
                distance_to_previous.append(math.dist(point, previous_point))

            for num_previous_point in np.arange(1, min(idx_row, 50)):
                previous_point = df_phase.loc[idx_row - num_previous_point, ["x", "y"]].to_numpy()
                distance_to_previous.append(math.dist(point, previous_point))

            movement_index.append(np.mean(distance_to_previous))

        # movement_index_scaled = list((movement_index - np.max(movement_index)) / (np.min(movement_index) - np.max(movement_index)))
        df_phase["distance_to_previous"] = movement_index
        df_phase["phase"] = phase
        df_phase["VP"] = df_phase["VP"].astype("int")

        df_phase.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';', index=False, mode='a',
                        header=not (os.path.exists(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'))))

        df_phase = df.loc[df["phase"] == phase].reset_index(drop=True)
        df_phase = df_phase.loc[~(df_phase["event"].str.contains("Office"))]

        df_phase["section"] = 0
        section = 0
        df_phase["delta_timestamp"] = (df_phase["timestamp"].shift(-1) - df_phase["timestamp"]).dt.total_seconds()
        for idx_row, row in df_phase.iterrows():
            # idx_row = 0
            # row = df_phase.iloc[idx_row, :]
            df_phase.loc[idx_row, "section"] = section
            if row["delta_timestamp"] > 1:
                section += 1
        df_phase = df_phase.drop(columns="delta_timestamp")

        for section in df_phase["section"].unique():
            # section = df_phase["section"].unique()[0]
            df_section = df_phase.loc[df_phase["section"] == section].reset_index(drop=True)
            condition = df_section.loc[0, "Condition"]
            room = df_rooms.loc[df_rooms["Role"] == condition, "Rooms"].item()
            vh_position = np.array([-870, 262]) if room == "Living" else np.array([-490, -1034])

            df_section["time"] = pd.to_timedelta(df_section["timestamp"] - start)
            df_section = df_section.set_index("time")
            df_section = df_section.resample("0.2S").mean(numeric_only=True)
            df_section = df_section.reset_index()

            df_section = df_section.dropna(subset="VP")

            distance_to_vh = []
            for idx_row, row in df_section.iterrows():
                # idx_row = 600
                # row = df_section.iloc[idx_row, :]
                point = df_section.loc[idx_row, ["x", "y"]].to_numpy()
                distance_to_vh.append(math.dist(point, vh_position))
            df_section["distance"] = [distance / 100 for distance in distance_to_vh]
            df_section["phase"] = phase
            df_section["Condition"] = condition
            df_section["VP"] = df_section["VP"].astype("int")

            df_section = df_section[['VP', 'time', 'phase', 'Condition', 'distance']]

            df_section.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance_vh.csv'), decimal='.', sep=';', index=False, mode='a',
                            header=not (os.path.exists(os.path.join(dir_path, f'Data-Wave{wave}', 'distance_vh.csv'))))

# Add Subject Data
df_move = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';')
df_move = df_move.iloc[:, 0:8]
df_move["distance_to_previous_scaled"] = (df_move["distance_to_previous"] - np.max(df_move["distance_to_previous"])) / (np.min(df_move["distance_to_previous"]) - np.max(df_move["distance_to_previous"]))

df_scores = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'scores_summary.csv'), decimal=',', sep=';')
df_move = df_move.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                   'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                   'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                   'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                   'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_move = df_move.drop(columns=['ID'])
df_move.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';', index=False)

# df_distance = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'walking_distance.csv'), decimal='.', sep=';')
# df_distance = df_distance.iloc[:, 0:5]
# # df_distance = df_distance.reset_index(drop=True)
# # df_distance["VP"] = df_distance["VP"].astype("int")
# df_distance = df_distance.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
#                                    'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
#                                    'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
#                                    'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
#                                    'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
# df_distance = df_distance.drop(columns=['ID'])
# df_distance.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'walking_distance.csv'), decimal='.', sep=';', index=False)

df_dist_vh = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance_vh.csv'), decimal='.', sep=';')
df_dist_vh = df_dist_vh.iloc[:, 0:5]

df_scores = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'scores_summary.csv'), decimal=',', sep=';')
df_dist_vh = df_dist_vh.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                   'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                   'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                   'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                   'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_dist_vh = df_dist_vh.drop(columns=['ID'])
df_dist_vh.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance_vh.csv'), decimal='.', sep=';', index=False)

df_event = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';')
df_event = df_event.iloc[:, 0:5]

df_scores = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'scores_summary.csv'), decimal=',', sep=';')
df_event = df_event.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                   'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                   'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                   'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                   'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_event = df_event.drop(columns=['ID'])
df_event.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';', index=False)


# # Interpersonal Distance
# for vp in vps:
#     # vp = vps[1]
#     vp = f"0{vp}" if vp < 10 else f"{vp}"
#     print(f"VP: {vp}")
#
#     # Get Distances
#     try:
#         files = [item for item in os.listdir(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp)) if (item.endswith(".csv"))]
#         file = [file for file in files if "distance" in file][0]
#         df = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp, file), sep=';', decimal='.')
#         if pd.to_datetime(df.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
#             df["timestamp"] = pd.to_datetime(df["timestamp"]) + timedelta(hours=2)
#         else:
#             df["timestamp"] = pd.to_datetime(df["timestamp"]) + timedelta(hours=1)
#         df["timestamp"] = df["timestamp"].apply(lambda t: t.replace(tzinfo=None))
#     except:
#         print("no distance file")
#         continue
#
#     # Get Events
#     try:
#         files = [item for item in os.listdir(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp)) if (item.endswith(".csv"))]
#         event_file = [file for file in files if "event" in file][0]
#         df_event = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'VP_' + vp, event_file), sep=';', decimal='.')
#
#         if pd.to_datetime(df_event.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
#             df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=2)
#         else:
#             df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=1)
#         df_event["timestamp"] = df_event["timestamp"].apply(lambda t: t.replace(tzinfo=None))
#     except:
#         print("no events file")
#         continue
#
#     try:
#         start_roomtour = df_event.loc[df_event["event"] == "StartRoomTour", "timestamp"].item()
#         start_habituation = df_event.loc[df_event["event"] == "StartExploringRooms", "timestamp"].item()
#         start_roomrating1 = df_event.loc[df_event["event"] == "EndExploringRooms", "timestamp"].item()
#         start_conditioning = df_event.loc[df_event["event"] == "EnterTerrace", "timestamp"].reset_index(drop=True)[0]
#         start_roomrating2 = df_event.loc[df_event["event"] == "EndExploringRooms2", "timestamp"].item()
#         start_test = start_roomrating2 - timedelta(seconds=180)
#         end_acq = df_event.loc[(df_event["event"] == "EnterOffice") & (df_event["timestamp"] > start_conditioning) & (df_event["timestamp"] < start_test), "timestamp"].reset_index(drop=True)[0]
#         start_personrating = df_event.loc[df_event["event"] == "TeleportToStartingRoom", "timestamp"].item()
#         end = df_event.loc[df_event["event"] == "End", "timestamp"].item()
#     except:
#         print("not enough events")
#         continue
#
#     dfs = []
#     df_hab = df_event.loc[(start_habituation <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating1)]
#     df_hab["duration"] = (df_hab["timestamp"].shift(-1) - df_hab["timestamp"]).dt.total_seconds()
#     df_hab["event"] = df_hab["event"].replace("StartExploringRooms", "EnterOffice")
#     df_hab = df_hab.loc[df_hab["event"].str.contains("Enter")]
#     df_hab["event"] = ["Habituation_" + name[1] for name in df_hab["event"].str.split("Enter")]
#     dfs.append(df_hab)
#
#     df_acq = df_event.loc[(start_conditioning <= df_event["timestamp"]) & (df_event["timestamp"] < end_acq)]
#     df_acq = df_acq.loc[(df_acq["event"].str.contains("Interaction")) & ~(df_acq["event"].str.contains("Finished"))]
#     df_acq["duration"] = 5
#     df_acq["event"] = [name[1] for name in df_acq["event"].str.split("Start")]
#     df_acq = df_acq.drop_duplicates(subset="event")
#     dfs.append(df_acq)
#
#     df_test = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
#     df_test = pd.concat([df_test, pd.DataFrame({"timestamp": [start_test], "event": "EnterOffice"})])
#     df_test = df_test.sort_values(by="timestamp")
#     df_test = df_test.loc[(df_test["event"].str.contains("Enter")) | (df_test["event"].str.contains("Clicked"))].reset_index(drop=True)
#     room = ""
#     for idx_row, row in df_test.iterrows():
#         # idx_row = 0
#         # row = df_test.iloc[idx_row, :]
#         if "Enter" in row["event"]:
#             room = row["event"]
#         elif "Clicked" in row["event"]:
#             df_test = pd.concat(
#                 [df_test, pd.DataFrame({"timestamp": [row["timestamp"] + timedelta(seconds=3)], "event": [room]})])
#     df_test = df_test.sort_values(by="timestamp").reset_index(drop=True)
#     df_test = drop_consecutive_duplicates(df_test, subset="event", keep="first", times="timestamp", tolerance=0.1)
#     df_test = df_test.reset_index(drop=True)
#     df_test["duration"] = (df_test["timestamp"].shift(-1) - df_test["timestamp"]).dt.total_seconds()
#     df_test.loc[len(df_test) - 1, "duration"] = (start_roomrating2 - df_test.loc[len(df_test) - 1, "timestamp"]).total_seconds()
#     df_test = df_test.loc[df_test["event"].str.contains("Enter")]
#     df_test["event"] = ["Test_" + name[1] for name in df_test["event"].str.split("Enter")]
#     dfs.append(df_test)
#
#     df_test_person = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
#     df_test_person = pd.concat([df_test_person, pd.DataFrame({"timestamp": [start_test], "event": "EnterOffice"})])
#     df_test_person = df_test_person.sort_values(by="timestamp").reset_index(drop=True)
#     df_test_person = df_test_person.loc[(df_test_person["event"].str.contains("Clicked"))]
#     if len(df_test_person) > 0:
#         df_test_person["duration"] = 3
#         df_test_person["event"] = ["Test_" + name for name in df_test_person["event"]]
#         for person in list(df_test_person["event"].unique()):
#             # person = list(df_test_person["event"].unique())[1]
#             df_test_person_unique = df_test_person.loc[df_test_person["event"] == person].reset_index(drop=True)
#             df_test_person_unique = drop_consecutive_duplicates(df_test_person_unique, subset="event", tolerance=2.1)
#             dfs.append(df_test_person_unique)
#
#     df_event = pd.concat(dfs)
#     df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)
#
#     # Add "event" columns (for MNE)
#     df_event["name"] = df_event["event"]
#     df_event.loc[df_event['name'].str.contains("resting state"), 'event'] = 1
#     df_event.loc[df_event['name'].str.contains("Habituation_Living"), 'event'] = 2
#     df_event.loc[df_event['name'].str.contains("Habituation_Dining"), 'event'] = 3
#     df_event.loc[df_event['name'].str.contains("Habituation_Office"), 'event'] = 4
#     df_event.loc[df_event['name'].str.contains("Test_Living"), 'event'] = 12
#     df_event.loc[df_event['name'].str.contains("Test_Dining"), 'event'] = 13
#     df_event.loc[df_event['name'].str.contains("Test_Office"), 'event'] = 14
#     df_event.loc[df_event['name'].str.contains("Test_Terrace"), 'event'] = 15
#     df_event.loc[df_event['name'].str.contains("Unfriendly"), 'event'] = 21
#     df_event.loc[df_event['name'].str.contains("Friendly"), 'event'] = 22
#     df_event.loc[df_event['name'].str.contains("Neutral"), 'event'] = 23
#     df_event.loc[df_event['name'].str.contains("Test_BryanWasClicked"), 'event'] = 31
#     df_event.loc[df_event['name'].str.contains("Test_EmanuelWasClicked"), 'event'] = 32
#     df_event.loc[df_event['name'].str.contains("Test_EttoreWasClicked"), 'event'] = 33
#     df_event.loc[df_event['name'].str.contains("Test_OskarWasClicked"), 'event'] = 34
#
#     # Merge "name" and "event"-column to df_ecg
#     df = pd.merge_asof(df, df_event[["timestamp", "name"]], on="timestamp", direction="backward").reset_index(drop=True)
#     df = drop_consecutive_duplicates(df, subset="timestamp", keep="first", times="timestamp", tolerance=0.1)
#     df = df.reset_index(drop=True)
#     df["VP"] = int(vp)
#     df["event"] = df["name"]
#     df = df.sort_values(by="timestamp")
#
#     try:
#         df_cond = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Conditions3")
#         df_cond = df_cond[["VP", "Roles", "Rooms"]]
#         df_cond = df_cond.loc[df_cond["VP"] == int(vp)]
#
#         df_roles = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Roles")
#         df_roles = df_roles[["Character", int(df_cond["Roles"].item())]]
#         df_roles = df_roles.rename(columns={int(df_cond["Roles"].item()): "Role"})
#
#         df_rooms = pd.read_excel(os.path.join(dir_path, f'Data-Wave{wave}', 'Conditions.xlsx'), sheet_name="Rooms3")
#         df_rooms = df_rooms[["Role", int(df_cond["Rooms"].item())]]
#         df_rooms = df_rooms.rename(columns={int(df_cond["Rooms"].item()): "Rooms"})
#
#         df_roles = df_roles.merge(df_rooms, on="Role")
#     except:
#         print("no conditions file")
#
#     df["Condition"] = ""
#     df["Condition_Room"] = ""
#     df["Condition_Actor"] = ""
#     df = df.dropna(subset=["event"])
#     df["actor"] = df["actor"].str.replace(" ", "_")
#     df["actor"] = [actor[1].split("_Char")[0] for actor in df["actor"].str.split("BP_")]
#
#     for idx_row, row in df_roles.iterrows():
#         # idx_row = 0
#         # row = df_roles.iloc[idx_row, :]
#         room = row["Rooms"]
#         role = row["Role"]
#         character = row["Character"]
#         df.loc[df["event"].str.contains(character), "Condition_Room"] = role
#         df.loc[df["event"].str.contains(room), "Condition_Room"] = role
#         df.loc[df["actor"].str.contains(character), "Condition_Actor"] = role
#
#     df = df.loc[df["Condition_Room"] == df["Condition_Actor"]]
#     df.loc[:, "Condition"] = df["Condition_Room"]
#     df = df.drop(columns=["Condition_Room", "Condition_Actor"])
#
#     df = df[["VP", "timestamp", "event", "Condition", "actor", "distance", "phase"]]
#     df = df.loc[df["phase"] == "Test"].reset_index(drop=True)
#     df = df.loc[~(df["event"].str.contains("Office"))].reset_index(drop=True)
#
#     df["section"] = 0
#     section = 0
#     df["delta_timestamp"] = (df["timestamp"].shift(-1) - df["timestamp"]).dt.total_seconds()
#     for idx_row, row in df.iterrows():
#         # idx_row = 0
#         # row = df.iloc[idx_row, :]
#         df.loc[idx_row, "section"] = section
#         if row["delta_timestamp"] > 1:
#             section += 1
#     df = df.drop(columns="delta_timestamp")
#
#     for section in df["section"].unique():
#         # section = df["section"].unique()[0]
#         df_phase = df.loc[df["section"] == section].reset_index(drop=True)
#         condition = df_phase.loc[0, "Condition"]
#         actor = df_phase.loc[0, "actor"]
#
#         start = df_phase.loc[0, "timestamp"]
#         df_phase["time"] = pd.to_timedelta(df_phase["timestamp"] - start)
#         df_phase = df_phase.set_index("time")
#         df_phase = df_phase.resample("0.21S").mean(numeric_only=True)
#         df_phase = df_phase.reset_index()
#
#         df_phase["VP"] = int(vp)
#         df_phase["section"] = int(section)
#         df_phase["Condition"] = condition
#
#         df_phase.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance.csv'), decimal='.', sep=';', index=False, mode='a',
#                         header=not (os.path.exists(os.path.join(dir_path, f'Data-Wave{wave}', 'distance.csv'))))
#
# # Add Subject Data
# df_distance = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance.csv'), decimal='.', sep=';')
# df_distance = df_distance.iloc[:, 0:6]
#
# df_scores = pd.read_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'scores_summary.csv'), decimal=',', sep=';')
# df_distance = df_distance.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
#                                            'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
#                                            'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
#                                            'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
#                                            'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
# df_distance = df_distance.drop(columns=['ID'])
# df_distance.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance.csv'), decimal='.', sep=';', index=False)

