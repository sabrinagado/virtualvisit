# =============================================================================
# Scores
# source: Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def drop_consecutive_duplicates(df, subset, keep="first", times="timestamp", tolerance=0.1):
    if keep == "first":
        df = df.loc[(df[subset].shift(1) != df[subset]) | ((df[times] - df[times].shift(1)).dt.total_seconds() >= tolerance)]
    elif keep == "last":
        df = df.loc[(df[subset].shift(-1) != df[subset]) | ((df[times].shift(-1) - df[times]).dt.total_seconds() >= tolerance)]
    return df


dir_path = os.getcwd()
start = 1
end = 11
vps = np.arange(start, end + 1)

for vp in vps:
    # vp = vps[1]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

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

    df_event["Condition"] = ""
    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        room = row["Rooms"]
        role = row["Role"]
        df_event.loc[df_event["event"].str.contains(room), "Condition"] = role
    df_event["VP"] = int(vp)
    df_event = df_event[["VP", "timestamp", "event", "Condition", "duration"]]

    df_event.to_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';', index=False, mode='a',
                    header=not (os.path.exists(os.path.join(dir_path, 'Data', 'events.csv'))))

# Add Subject Data
df_events = pd.read_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';')
df_events = df_events.iloc[:, 0:5]

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_events = df_events.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                               'SSQ', 'SSQ-N', 'SSQ-O', 'SSQ-D', 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS',
                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_events = df_events.drop(columns=['ID'])
df_events.to_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';', index=False)


for vp in vps:
    # vp = vps[1]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    # Get Distances
    files = [item for item in os.listdir(os.path.join(dir_path, 'Data', 'VP_' + vp)) if (item.endswith(".csv"))]
    file = [file for file in files if "distance" in file][0]
    df = pd.read_csv(os.path.join(dir_path, 'Data', 'VP_' + vp, event_file), sep=';', decimal='.')

    # Get Events
    files = [item for item in os.listdir(os.path.join(dir_path, 'Data', 'VP_' + vp)) if (item.endswith(".csv"))]
    event_file = [file for file in files if "distance" in file][0]
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

    df_event["Condition"] = ""
    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        room = row["Rooms"]
        role = row["Role"]
        df_event.loc[df_event["event"].str.contains(room), "Condition"] = role
    df_event["VP"] = int(vp)
    df_event = df_event[["VP", "timestamp", "event", "Condition", "duration"]]

    df_event.to_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';', index=False, mode='a',
                    header=not (os.path.exists(os.path.join(dir_path, 'Data', 'events.csv'))))

# Add Subject Data
df_events = pd.read_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';')
df_events = df_events.iloc[:, 0:5]

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_events = df_events.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                               'SSQ', 'SSQ-N', 'SSQ-O', 'SSQ-D', 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS',
                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_events = df_events.drop(columns=['ID'])
df_events.to_csv(os.path.join(dir_path, 'Data', 'events.csv'), decimal='.', sep=';', index=False)
