# =============================================================================
# Behavior
# source: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import xml.etree.ElementTree as ET
from tqdm import tqdm

from Code import preproc_scores, preproc_ratings


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


def get_conditions(vp, filepath):
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
    return df_roles


def get_events(vp, filepath, wave, df_roles):
    files = [item for item in os.listdir(os.path.join(filepath, 'VP_' + vp)) if (item.endswith(".csv"))]
    event_file = [file for file in files if "event" in file][0]
    df_event = pd.read_csv(os.path.join(filepath, 'VP_' + vp, event_file), sep=';', decimal='.')

    if (pd.to_datetime(df_event.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26")) & (pd.to_datetime(df_event.loc[0, "timestamp"][0:10]) < pd.Timestamp("2023-10-29")):
        df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=2)
    else:
        df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=1)

    df_event["timestamp"] = df_event["timestamp"].apply(lambda t: t.replace(tzinfo=None))

    if wave == 2:
        vis_file = [file for file in files if "visibility" in file][0]
        df_vis = pd.read_csv(os.path.join(filepath, 'VP_' + vp, vis_file), sep=';', decimal='.')

        if (pd.to_datetime(df_vis.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26")) & (pd.to_datetime(df_vis.loc[0, "timestamp"][0:10]) < pd.Timestamp("2023-10-29")):
            df_vis["timestamp"] = pd.to_datetime(df_vis["timestamp"]) + timedelta(hours=2)
        else:
            df_vis["timestamp"] = pd.to_datetime(df_vis["timestamp"]) + timedelta(hours=1)

        df_vis["timestamp"] = df_vis["timestamp"].apply(lambda t: t.replace(tzinfo=None))

    df_event = drop_consecutive_duplicates(df_event, subset="event", keep="first", times="timestamp", tolerance=0.1)
    df_event = df_event.reset_index(drop=True)

    dfs = []

    # Add ECG and EDA markers:
    for physio in ["ECG", "EDA"]:
        # physio = "EDA"
        try:
            file_path_physio = os.path.join(filepath, 'VP_' + vp, physio)
            folder = [item for item in os.listdir(file_path_physio)][0]
            file_path_physio = os.path.join(file_path_physio, folder)
            start_time = ET.parse(os.path.join(file_path_physio, 'unisens.xml')).getroot().attrib['timestampStart']
            start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%f')
        except:
            continue
        try:
            # Get marker file
            marker = pd.read_csv(os.path.join(file_path_physio, 'marker.csv'), sep=';', decimal='.', names=['sample'], header=None, usecols=[0])
            marker_samplingrate = 64
            marker_timepoints = [start_time + timedelta(seconds=marker["sample"][i] / marker_samplingrate) for i in np.arange(0, len(marker["sample"]))]
            marker_timepoints = marker_timepoints[0]
            df_marker = pd.DataFrame([marker_timepoints], columns=['timestamp'])
            df_marker['event'] = f"{physio}_marker"
            # Add markers to event file
            df_event = pd.concat([df_event, df_marker])
            df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)
        except:
            continue

    # Get timepoint for resting state measurement
    try:
        df_rs = df_event.loc[(df_event['timestamp'] < pd.to_datetime(df_event.loc[df_event["event"] == "WelcomeWidget", 'timestamp'].item() - timedelta(seconds=45), unit="ns")) | (df_event['timestamp'] > pd.to_datetime(df_event.loc[df_event["event"] == "End", 'timestamp'].item(), unit="ns"))]
        df_rs = df_rs.loc[df_rs['event'].str.contains("marker")].reset_index(drop=True)
        df_rs = df_rs.drop_duplicates(subset=["event"], keep="last").reset_index(drop=True)
        rs_start = df_rs.loc[len(df_rs) - 1, "timestamp"] + timedelta(seconds=15)
        # rs_end = rs_start + timedelta(seconds=30)

        df_event = pd.concat([df_event, pd.DataFrame([[rs_start, "resting state"]], columns=["timestamp", "event"])])
        df_event = df_event.loc[~(df_event['event'].str.contains("marker"))]
        df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)

        df_event = drop_consecutive_duplicates(df_event, subset="event", keep="first", times="timestamp", tolerance=0.1)
        df_event = df_event.reset_index(drop=True)

        df_rs = df_event.loc[df_event["event"].str.contains("resting")]
        df_rs["duration"] = 30
        dfs.append(df_rs)
    except:
        print(f"no resting state for VP {vp}")

    df_event = drop_consecutive_duplicates(df_event, subset="event", keep="first", times="timestamp", tolerance=0.1)
    df_event = df_event.reset_index(drop=True)

    try:
        end_orientation = df_event.loc[df_event["event"] == "StartWidget", "timestamp"].item()
        baseline_orientation = end_orientation - timedelta(seconds=30)
        start_roomtour = df_event.loc[df_event["event"] == "StartRoomTour", "timestamp"].item()
        start_habituation = df_event.loc[df_event["event"] == "StartExploringRooms", "timestamp"].item()
        start_roomrating1 = df_event.loc[df_event["event"] == "EndExploringRooms", "timestamp"].item()
        start_roomrating2 = df_event.loc[df_event["event"] == "EndExploringRooms2", "timestamp"].item()
        if wave == 1:
            start_conditioning = df_event.loc[df_event["event"] == "EnterTerrace", "timestamp"].reset_index(drop=True)[0]
            start_test = start_roomrating2 - timedelta(seconds=180)
            end_acq = df_event.loc[(df_event["event"] == "AllInteractionsFinished") & (df_event["timestamp"] > start_conditioning) & (df_event["timestamp"] < start_test), "timestamp"].reset_index(drop=True)[0]
        elif wave == 2:
            start_conditioning = df_event.loc[df_event["event"] == "Player_EnterTerrace", "timestamp"].reset_index(drop=True)[0]
            start_test = start_roomrating2 - timedelta(seconds=180)
            end_acq = df_event.loc[(df_event["event"] == "Player_EnterOffice") & (df_event["timestamp"] > start_conditioning) & (df_event["timestamp"] < start_test), "timestamp"].reset_index(drop=True)[0]
        start_personrating = df_event.loc[df_event["event"] == "TeleportToStartingRoom", "timestamp"].item()
        end = df_event.loc[df_event["event"] == "End", "timestamp"].item()
    except:
        print("not enough events")

    df_orient = pd.DataFrame({"timestamp": [baseline_orientation], "event": ["Orientation"], "duration": [30.]})
    dfs.append(df_orient)

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
    df_acq = pd.concat([df_acq, pd.DataFrame({"timestamp": [start_conditioning], "event": [f"Start_Acquisition"], "duration": [(end_acq-start_conditioning).total_seconds()]})])
    df_acq = pd.concat([df_acq, pd.DataFrame({"timestamp": [end_acq], "event": [f"End_Acquisition"]})])
    df_acq = df_acq.sort_values(by="timestamp").reset_index(drop=True)
    dfs.append(df_acq)

    if wave == 1:
        df_test = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
        df_test = pd.concat([df_test, pd.DataFrame({"timestamp": [start_test], "event": "EnterOffice"})])
        df_test = df_test.sort_values(by="timestamp")
        df_test = df_test.loc[(df_test["event"].str.contains("Enter")) | (df_test["event"].str.contains("Clicked"))].reset_index(drop=True)
        room = ""
        for idx_row, row in df_test.iterrows():
            # idx_row = 0
            # row = df_test.iloc[idx_row, :]
            if "Enter" in row["event"]:
                room = row["event"]
            elif "Clicked" in row["event"]:
                df_test = pd.concat([df_test, pd.DataFrame({"timestamp": [row["timestamp"] + timedelta(seconds=3)], "event": [room]})])
        df_test = df_test.sort_values(by="timestamp").reset_index(drop=True)
        df_test = drop_consecutive_duplicates(df_test, subset="event", keep="first", times="timestamp", tolerance=0.1)
        df_test = df_test.reset_index(drop=True)
        df_test["duration"] = (df_test["timestamp"].shift(-1) - df_test["timestamp"]).dt.total_seconds()
        df_test.loc[len(df_test) - 1, "duration"] = (start_roomrating2 - df_test.loc[len(df_test) - 1, "timestamp"]).total_seconds()
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

    elif wave == 2:
        df_test = df_event.loc[(start_test <= df_event["timestamp"]) & (df_event["timestamp"] <= start_roomrating2)]
        df_test = pd.concat([df_test, pd.DataFrame({"timestamp": [start_test], "event": "Player_EnterOffice"})])
        df_test = df_test.sort_values(by="timestamp")

        df_test_rooms = df_test.loc[df_test["event"].str.contains("Player")]
        df_test_rooms = df_test_rooms.reset_index(drop=True)
        df_test_rooms = df_test_rooms.loc[df_test_rooms["event"].str.contains("Enter")]
        df_test_rooms["event"] = ["Test_AloneIn" + name[1] for name in df_test_rooms["event"].str.split("Enter")]

        df_test = df_test.loc[(df_test["event"].str.contains("Enter"))].reset_index(drop=True)
        df_test_vis = df_vis.loc[(start_test <= df_vis["timestamp"]) & (df_vis["timestamp"] <= start_roomrating2)]

        df_test_agents = df_test.copy()
        df_test_agents["Player"] = "Office"
        df_test_agents["Emanuel"] = df_roles.loc[(df_roles["Character"] == "Emanuel"), "Rooms"].item()
        df_test_agents["Ettore"] = df_roles.loc[(df_roles["Character"] == "Ettore"), "Rooms"].item()
        df_test_agents["Bryan"] = df_roles.loc[(df_roles["Character"] == "Bryan"), "Rooms"].item()
        df_test_agents["Oskar"] = df_roles.loc[(df_roles["Character"] == "Oskar"), "Rooms"].item()

        for idx_row, row in df_test.iterrows():
            # idx_row = 9
            # row = df_test.iloc[idx_row, :]
            for actor in ["Player", "Emanuel", "Ettore", "Bryan", "Oskar"]:
                if actor in row["event"]:
                    df_test_agents.loc[idx_row:len(df_test_agents), actor] = row["event"].split("Enter")[1].split("Room")[0]

        for room in ["Office", "Dining", "Living"]:
            # room = "Office"
            for actor in ["Emanuel", "Ettore", "Bryan", "Oskar"]:
                # actor = "Emanuel"
                df_test_agents["tog"] = 0
                for idx_row, row in df_test_agents.iterrows():
                    # idx_row = 1
                    # row = df_test_agents.iloc[idx_row, :]
                    if (row["Player"] == room) & (row["Player"] == row[actor]):
                        df_test_agents.loc[idx_row, "tog"] = 1
                together = False
                start = None
                end = None
                duration = 0
                for idx_row, row in df_test_agents.iterrows():
                    # idx_row = 2
                    # row = df_test_agents.iloc[idx_row, :]
                    if row["tog"] and not together:
                        start = row["timestamp"]
                        together = True
                    elif together and not row["tog"]:
                        end = row["timestamp"]
                        duration = (end - start).total_seconds()
                        df_test_rooms = pd.concat([df_test_rooms, pd.DataFrame({"timestamp": [start], "event": [f"Test_With{actor}In{room}"], "duration": [duration]})])
                        together = False
        df_test_rooms = df_test_rooms.sort_values(by="timestamp").reset_index(drop=True)
        df_test_rooms["duration"] = (df_test_rooms["timestamp"].shift(-1) - df_test_rooms["timestamp"]).dt.total_seconds()
        df_test_rooms.loc[len(df_test_rooms) - 1, "duration"] = (start_roomrating2 - df_test_rooms.loc[len(df_test_rooms) - 1, "timestamp"]).total_seconds()
        df_test_rooms = df_test_rooms.loc[df_test_rooms["duration"] > 0]

        for actor in ["Emanuel", "Ettore", "Bryan", "Oskar"]:
            # actor = "Bryan"
            visible = False
            start = None
            end = None
            duration = 0
            df_test_vis_actor = df_test_vis.loc[df_test_vis["actor"].str.contains(actor)].reset_index(drop=True)
            for idx_row, row in df_test_vis_actor.iterrows():
                # idx_row = 0
                # row = df_test_vis_actor.iloc[idx_row, :]
                if row["sight"]:
                    start = row["timestamp"]
                elif not row["sight"]:
                    if idx_row == 0:
                        continue
                    end = row["timestamp"]
                    duration = (end - start).total_seconds()
                    df_test_rooms = pd.concat([df_test_rooms, pd.DataFrame({"timestamp": [start], "event": [f"Test_{actor}WasVisible"], "duration": [duration]})])
        df_test_rooms = df_test_rooms.sort_values(by="timestamp").reset_index(drop=True)

        for actor in ["Emanuel", "Ettore", "Bryan", "Oskar"]:
            # actor = "Emanuel"
            df_test_agent_visible = df_test_rooms.loc[(df_test_rooms["event"].str.contains("WasVisible")) & (
                df_test_rooms["event"].str.contains(actor))].reset_index(drop=True)
            for idx_row, row in df_test_agent_visible.iterrows():
                # idx_row = 0
                # row = df_test_agent_visible.loc[idx_row, :]
                start_notvisible = row["timestamp"] + timedelta(seconds=row["duration"])
                if idx_row == len(df_test_agent_visible) - 1:
                    start_next_event = start_roomrating2
                else:
                    start_next_event = df_test_agent_visible.loc[idx_row + 1, "timestamp"]
                duration = (start_next_event - start_notvisible).total_seconds()
                df_test_rooms = pd.concat([df_test_rooms, pd.DataFrame({"timestamp": [start_notvisible], "event": [f"Test_{actor}NotVisible"], "duration": [duration]})])

        df_test_rooms = df_test_rooms.sort_values(by="timestamp").reset_index(drop=True)
        df_test_vis = df_test_rooms.loc[(df_test_rooms["event"].str.contains("Test")) & (df_test_rooms["event"].str.contains("Visible"))].reset_index(drop=True)
        df_test_vis = pd.concat([df_test_vis, pd.DataFrame({"timestamp": [start_test], "event": [f"Test_NoActorVisible"], "duration": [(df_test_vis.iloc[0, 0] - start_test).total_seconds()]})])
        df_test_vis = df_test_vis.sort_values(by="timestamp").reset_index(drop=True)
        df_test_rooms = df_test_rooms.loc[(df_test_rooms["event"].str.contains("Test")) & ~(df_test_rooms["event"].str.contains("Visible"))].reset_index(drop=True)

        visible = []

        for idx_row, row in df_test_vis.iterrows():
            # idx_row = 1
            # row = df_test_rooms.loc[idx_row, :]
            if "WasVisible" in row["event"]:
                visible.append(row["event"].split("Test_")[1].split("WasVisible")[0])
            elif "NotVisible" in row["event"]:
                visible.remove(row["event"].split("Test_")[1].split("NotVisible")[0])

            if len(visible) == 0:
                df_test_vis = pd.concat([df_test_vis, pd.DataFrame({"timestamp": [row["timestamp"]], "event": [f"Test_NoActorVisible"], "duration": [np.nan]})])
            else:
                df_test_vis = pd.concat([df_test_vis, pd.DataFrame({"timestamp": [row["timestamp"]], "event": [f"Test_{''.join(visible)}Visible"], "duration": [np.nan]})])

        df_test_vis = df_test_vis.loc[~((df_test_vis["event"].str.contains("NotVisible")) | (df_test_vis["event"].str.contains("WasVisible")))]
        df_test_vis = df_test_vis.sort_values(by="timestamp").reset_index(drop=True)
        df_test_vis["duration"] = (df_test_vis["timestamp"].shift(-1) - df_test_vis["timestamp"]).dt.total_seconds()
        df_test_vis.loc[len(df_test_vis) - 1, "duration"] = (start_roomrating2 - df_test_vis.loc[len(df_test_vis) - 1, "timestamp"]).total_seconds()
        df_test_vis = df_test_vis.loc[df_test_vis["duration"] > 0]

        df_test = pd.concat([df_test_rooms, df_test_vis])
        df_test = df_test.sort_values(by="timestamp").reset_index(drop=True)
        dfs.append(df_test)

    df_event = pd.concat(dfs)
    df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)

    df_event = df_event.loc[df_event["duration"] > 0]
    events = {"start_orientation": baseline_orientation,
              "start_habituation": start_habituation,
              "start_roomrating1": start_roomrating1,
              "start_roomrating2": start_roomrating2,
              "start_acq": start_conditioning,
              "end_acq": end_acq,
              "start_test": start_test}

    return df_event, events


# Durations and Clicks
def get_phases(vps, filepath, wave, df_scores):
    df_events = pd.DataFrame()
    for vp in vps:
        # vp = vps[8]
        vp = f"0{vp}" if vp < 10 else f"{vp}"

        # Get Conditions
        try:
            df_roles = get_conditions(vp, filepath)
        except:
            print(f"no conditions file for VP {vp}")
            continue

        # Get Events
        try:
            df_events_vp, events = get_events(vp, filepath, wave, df_roles)
        except:
            print(f"no events file for VP {vp}")
            continue

        df_events_vp["Condition"] = ""
        for idx_row, row in df_roles.iterrows():
            # idx_row = 0
            # row = df_roles.iloc[idx_row, :]
            room = row["Rooms"]
            role = row["Role"]
            character = row["Character"]
            df_events_vp.loc[(df_events_vp["event"].str.contains(room)) & ~ (df_events_vp["event"].str.contains("With")), "Condition"] = role
            df_events_vp.loc[df_events_vp["event"].str.contains(character), "Condition"] = role
        df_events_vp["VP"] = int(vp)
        df_events_vp = df_events_vp[["VP", "timestamp", "event", "Condition", "duration"]]

        # Add Conditions
        for idx_row, row in df_roles.iterrows():
            # idx_row = 0
            # row = df_roles.iloc[idx_row, :]
            room = row["Rooms"]
            role = row["Role"]
            character = row["Character"]
            df_events_vp["event"] = df_events_vp["event"].str.replace(character, role.capitalize())

        df_events = pd.concat([df_events, df_events_vp])

    # Add Subject Data
    df_events = df_events.iloc[:, 0:5]

    df_events = df_events.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                           'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                           'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                           'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                           'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
    df_events = df_events.drop(columns=['ID'])
    return df_events


# Movement
def get_distances(vps, filepath, wave, df_scores):
    df_distance = pd.DataFrame()
    df_movement = pd.DataFrame()
    df_dist_vh = pd.DataFrame()
    for vp in tqdm(vps):
        # vp = vps[14]
        vp = f"0{vp}" if vp < 10 else f"{vp}"

        # Get Movement File
        try:
            files = [item for item in os.listdir(os.path.join(filepath, 'VP_' + vp)) if (item.endswith(".csv"))]
            file = [file for file in files if "movement" in file][0]
            df = pd.read_csv(os.path.join(filepath, 'VP_' + vp, file), sep=';', decimal='.')
            if (pd.to_datetime(df.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26")) & (pd.to_datetime(df.loc[0, "timestamp"][0:10]) < pd.Timestamp("2023-10-29")):
                df["timestamp"] = pd.to_datetime(df["timestamp"]) + timedelta(hours=2)
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"]) + timedelta(hours=1)
            df["timestamp"] = df["timestamp"].apply(lambda t: t.replace(tzinfo=None))
        except:
            print(f"no movement file for VP {vp}")
            continue

        # Get Conditions
        try:
            df_roles = get_conditions(vp, filepath)
        except:
            print(f"no conditions file for VP {vp}")
            continue

        # Get Events
        try:
            df_events_vp, events = get_events(vp, filepath, wave, df_roles)
        except:
            print(f"no events file for VP {vp}")
            continue

        # Merge "event"-column to df
        df_event_rooms = df_events_vp.loc[~df_events_vp["event"].str.contains("Visible")]
        df = pd.merge_asof(df, df_event_rooms[["timestamp", "event"]], on="timestamp", direction="backward").reset_index(drop=True)

        df_event_vis = df_events_vp.loc[df_events_vp["event"].str.contains("Visible")]
        df_event_vis = pd.concat([df_event_vis, df_event_rooms.loc[~df_event_rooms["event"].str.contains("Test")]])
        df_event_vis = df_event_vis.sort_values(by="timestamp").reset_index(drop=True)
        df = pd.merge_asof(df, df_event_vis[["timestamp", "event"]], on="timestamp", direction="backward", suffixes=["", "_vis"]).reset_index(drop=True)

        df = df.dropna(subset="event")
        df["Condition"] = ""
        for idx_row, row in df_roles.iterrows():
            # idx_row = 0
            # row = df_roles.iloc[idx_row, :]
            room = row["Rooms"]
            role = row["Role"]
            character = row["Character"]
            if wave == 1:
                df.loc[df["event"].str.contains(room), "Condition"] = role
                df.loc[df["event"].str.contains(character), "Condition"] = role
            if wave == 2:
                df.loc[df["Actor"].str.contains(character), "Condition"] = role
        df["VP"] = int(vp)

        df.loc[(events["start_habituation"] <= df["timestamp"]) & (df["timestamp"] < events["start_roomrating1"]), "phase"] = "Habituation"
        df.loc[(events["start_test"] <= df["timestamp"]) & (df["timestamp"] < events["start_roomrating2"]), "phase"] = "Test"

        if wave == 1:
            df["x"] = [float(position.split("=")[1].split(" ")[0]) for position in df["Position"]]
            df["y"] = [float(position.split("=")[2].split(" ")[0]) for position in df["Position"]]
            df = df[["VP", "timestamp", "event", "Condition", "phase", "x", "y"]].reset_index(drop=True)
            df = df.sort_values(by="timestamp")
            df = df.dropna(subset="phase")

            for phase in df["phase"].unique():
                # phase = "Habituation"
                df_phase = df.loc[df["phase"] == phase].reset_index(drop=True)

                # Movement
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
                df_phase["distance_to_previous_player"] = movement_index
                df_phase["phase"] = phase
                df_phase["VP"] = df_phase["VP"].astype("int")

                df_movement = pd.concat([df_movement, df_phase])

                # Interpersonal Distance
                df_event_rooms_phase = df_event_rooms.copy()
                df_event_rooms_phase["time"] = pd.to_timedelta(df_event_rooms_phase["timestamp"] - start)
                df_phase = pd.merge_asof(df_phase, df_event_rooms_phase[["time", "event"]], on="time", direction="backward").reset_index(drop=True)

                dict_room_cond = dict(df_roles[["Rooms", "Role"]].values)

                df_phase[f"distance_to_{dict_room_cond['Living']}"] = df_phase.apply(
                    lambda x: math.dist(x[["x", "y"]].to_numpy(), [-870, 262]), axis=1)
                df_phase[f"distance_to_{dict_room_cond['Dining']}"] = df_phase.apply(
                    lambda x: math.dist(x[["x", "y"]].to_numpy(), [-490, -1034]), axis=1)

                for condition in ["friendly", "unfriendly"]:
                    # condition = "unfriendly"
                    df_dist_vh_cond = df_phase.copy()
                    df_dist_vh_cond[f"distance"] = df_dist_vh_cond[f"distance_to_{condition}"] / 100
                    df_dist_vh_cond["Condition"] = condition
                    df_dist_vh_cond = df_dist_vh_cond[['VP', 'time', 'phase', 'event', 'Condition', 'distance']]

                    # Add Conditions
                    for idx_row, row in df_roles.iterrows():
                        # idx_row = 0
                        # row = df_roles.iloc[idx_row, :]
                        room = row["Rooms"]
                        role = row["Role"]
                        character = row["Character"]
                        df_dist_vh_cond["event"] = df_dist_vh_cond["event"].str.replace(room, role.capitalize())
                        df_dist_vh_cond["event"] = df_dist_vh_cond["event"].str.replace(character, role.capitalize())
                    df_dist_vh = pd.concat([df_dist_vh, df_dist_vh_cond])

        elif wave == 2:
            df["x"] = [float(position.split("=")[1].split(" ")[0]) for position in df["Position"]]
            df["y"] = [float(position.split("=")[2].split(" ")[0]) for position in df["Position"]]

            df.loc[df["Actor"] == "Player", "x_player"] = df["x"]
            df.loc[df["Actor"] == "Player", "y_player"] = df["y"]
            df.loc[df["Condition"] == "unfriendly", "x_unfriendly"] = df["x"]
            df.loc[df["Condition"] == "unfriendly", "y_unfriendly"] = df["y"]
            df.loc[df["Condition"] == "friendly", "x_friendly"] = df["x"]
            df.loc[df["Condition"] == "friendly", "y_friendly"] = df["y"]

            df = df[["VP", "timestamp", "event_vis", "event", "Condition", "phase", "x_player", "y_player", "x_friendly", "y_friendly", "x_unfriendly",  "y_unfriendly"]].reset_index(drop=True)
            df = df.sort_values(by=["timestamp", "Condition"])
            df[["x_player", "y_player", "x_friendly", "y_friendly", "x_unfriendly",  "y_unfriendly"]] = df[["x_player", "y_player", "x_friendly", "y_friendly", "x_unfriendly",  "y_unfriendly"]].fillna(method="bfill")
            df = df.drop_duplicates(subset="timestamp")
            df = df.dropna(subset="phase").reset_index(drop=True)

            for phase in df["phase"].unique():
                # phase = "Test"
                df_phase = df.loc[df["phase"] == phase].reset_index(drop=True)

                start = df_phase.loc[0, "timestamp"]
                df_phase["time"] = pd.to_timedelta(df_phase["timestamp"] - start)
                df_phase = df_phase.set_index("time")
                df_phase = df_phase.resample("0.2S").mean(numeric_only=True)
                df_phase = df_phase.reset_index()

                df_event_vis_phase = df_event_vis.copy()
                df_event_vis_phase["time"] = pd.to_timedelta(df_event_vis["timestamp"] - start)

                start_position = df_phase.drop_duplicates("VP", keep="first")[["x_player", "y_player"]].to_numpy().flatten()
                df_phase["distance_from_start"] = df_phase.apply(lambda x: math.dist(x[["x_player", "y_player"]].to_numpy().flatten(), start_position), axis=1)
                df_distance_temp = pd.DataFrame({"VP": [vp],
                                                 "phase": [phase],
                                                 "walking_distance": [distance(df_phase["x_player"].to_numpy(), df_phase["y_player"].to_numpy()) / 100],
                                                 "average_distance_to_start": [df_phase["distance_from_start"].mean() / 100],
                                                 "maximum_distance_to_start": [df_phase["distance_from_start"].max() / 100]})
                df_distance = pd.concat([df_distance, df_distance_temp])

                movement_index_player = []
                movement_index_friendly = []
                movement_index_unfriendly = []
                for idx_row, row in df_phase.iterrows():
                    # idx_row = 600
                    # row = df.iloc[idx_row, :]
                    if idx_row == 0:
                        movement_index_player.append(0.)
                        movement_index_friendly.append(0.)
                        movement_index_unfriendly.append(0.)
                        continue

                    distance_to_previous_player = []
                    point_player = df_phase.loc[idx_row, ["x_player", "y_player"]].to_numpy()
                    if idx_row == 1:
                        previous_point_player = df_phase.loc[idx_row - 1, ["x_player", "y_player"]].to_numpy()
                        distance_to_previous_player.append(math.dist(point_player, previous_point_player))
                    for num_previous_point in np.arange(1, min(idx_row, 50)):
                        previous_point = df_phase.loc[idx_row - num_previous_point, ["x_player", "y_player"]].to_numpy()
                        distance_to_previous_player.append(math.dist(point_player, previous_point_player))
                    movement_index_player.append(np.mean(distance_to_previous_player))

                    distance_to_previous_friendly = []
                    point_friendly = df_phase.loc[idx_row, ["x_friendly", "y_friendly"]].to_numpy()
                    if idx_row == 1:
                        previous_point_friendly = df_phase.loc[idx_row - 1, ["x_friendly", "y_friendly"]].to_numpy()
                        distance_to_previous_friendly.append(math.dist(point_friendly, previous_point_friendly))
                    for num_previous_point in np.arange(1, min(idx_row, 50)):
                        previous_point = df_phase.loc[idx_row - num_previous_point, ["x_friendly", "y_friendly"]].to_numpy()
                        distance_to_previous_friendly.append(math.dist(point_friendly, previous_point_friendly))
                    movement_index_friendly.append(np.mean(distance_to_previous_friendly))

                    distance_to_previous_unfriendly = []
                    point_unfriendly = df_phase.loc[idx_row, ["x_unfriendly", "y_unfriendly"]].to_numpy()
                    if idx_row == 1:
                        previous_point_unfriendly = df_phase.loc[idx_row - 1, ["x_unfriendly", "y_unfriendly"]].to_numpy()
                        distance_to_previous_unfriendly.append(math.dist(point_unfriendly, previous_point_unfriendly))
                    for num_previous_point in np.arange(1, min(idx_row, 50)):
                        previous_point = df_phase.loc[idx_row - num_previous_point, ["x_unfriendly", "y_unfriendly"]].to_numpy()
                        distance_to_previous_unfriendly.append(math.dist(point_unfriendly, previous_point_unfriendly))
                    movement_index_unfriendly.append(np.mean(distance_to_previous_unfriendly))

                # movement_index_scaled = list((movement_index - np.max(movement_index)) / (np.min(movement_index) - np.max(movement_index)))
                df_phase["distance_to_previous_player"] = movement_index_player
                df_phase["distance_to_previous_friendly"] = movement_index_friendly
                df_phase["distance_to_previous_unfriendly"] = movement_index_unfriendly
                df_phase["phase"] = phase
                df_phase["VP"] = df_phase["VP"].astype("int")

                df_phase = pd.merge_asof(df_phase, df_event_vis_phase[["time", "event"]], on="time", direction="backward").reset_index(drop=True)

                df_movement = pd.concat([df_movement, df_phase])

                if phase == "Test":
                    df_phase["distance_to_friendly"] = df_phase.apply(lambda x: math.dist(x[["x_player", "y_player"]].to_numpy().flatten(),
                                                                                          x[["x_friendly", "y_friendly"]].to_numpy().flatten()), axis=1)
                    df_phase["distance_to_unfriendly"] = df_phase.apply(lambda x: math.dist(x[["x_player", "y_player"]].to_numpy().flatten(),
                                                                                            x[["x_unfriendly", "y_unfriendly"]].to_numpy().flatten()), axis=1)
                    for condition in ["friendly", "unfriendly"]:
                        # condition = "friendly"
                        df_dist_vh_cond = df_phase.copy()
                        df_dist_vh_cond[f"distance"] = df_dist_vh_cond[f"distance_to_{condition}"] / 100
                        df_dist_vh_cond["Condition"] = condition
                        df_dist_vh_cond = df_dist_vh_cond[['VP', 'time', 'phase', 'event', 'Condition', 'distance']]

                        # Add Conditions
                        for idx_row, row in df_roles.iterrows():
                            # idx_row = 0
                            # row = df_roles.iloc[idx_row, :]
                            room = row["Rooms"]
                            role = row["Role"]
                            character = row["Character"]
                            df_dist_vh_cond["event"] = df_dist_vh_cond["event"].str.replace(character, role.capitalize())
                        df_dist_vh = pd.concat([df_dist_vh, df_dist_vh_cond])

    # Add participant scores
    df_movement["distance_to_previous_player_scaled"] = (df_movement["distance_to_previous_player"] - np.max(df_movement["distance_to_previous_player"])) / (np.min(df_movement["distance_to_previous_player"]) - np.max(df_movement["distance_to_previous_player"]))
    if wave == 2:
        df_movement["distance_to_previous_friendly_scaled"] = (df_movement["distance_to_previous_friendly"] - np.max(df_movement["distance_to_previous_friendly"])) / (np.min(df_movement["distance_to_previous_friendly"]) - np.max(df_movement["distance_to_previous_friendly"]))
        df_movement["distance_to_previous_unfriendly_scaled"] = (df_movement["distance_to_previous_unfriendly"] - np.max(df_movement["distance_to_previous_unfriendly"])) / (np.min(df_movement["distance_to_previous_unfriendly"]) - np.max(df_movement["distance_to_previous_unfriendly"]))

    df_movement = df_movement.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                               'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                               'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
    df_movement = df_movement.drop(columns=['ID'])

    df_distance = df_distance.reset_index(drop=True)
    df_distance["VP"] = df_distance["VP"].astype("int")
    df_distance = df_distance.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                               'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                               'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
    df_distance = df_distance.drop(columns=['ID'])

    df_dist_vh = df_dist_vh.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                             'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                             'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                             'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                             'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
    df_dist_vh = df_dist_vh.drop(columns=['ID'])

    return df_movement, df_distance, df_dist_vh


if __name__ == '__main__':
    wave = 1
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

    df_ratings, problematic_subjects = preproc_ratings.create_ratings(vps, filepath, problematic_subjects, df_scores)

    vps = [vp for vp in vps if not vp in problematic_subjects]

    df_events = get_phases(vps, filepath, wave, df_scores)
    df_events = df_events.loc[~(df_events["VP"].isin(problematic_subjects))]
    df_events.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'events.csv'), decimal='.', sep=';', index=False)

    df_movement, df_distance, df_dist_vh = get_distances(vps, filepath, wave, df_scores)
    df_movement.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'movement.csv'), decimal='.', sep=';', index=False)
    df_distance.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'walking_distance.csv'), decimal='.', sep=';', index=False)
    df_dist_vh.to_csv(os.path.join(dir_path, f'Data-Wave{wave}', 'distance_vh.csv'), decimal='.', sep=';', index=False)
