# =============================================================================
# ECG
# sensor: movisens
# study: Virtual Visit
# =============================================================================
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import neurokit2 as nk
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import matplotlib
from matplotlib import pyplot as plt

from Code.toolbox import utils
import mne

plt.ion()
matplotlib.use('QtAgg')


# % ===========================================================================
# Read in Data, Add Timestamps and Events
# =============================================================================
dir_path = os.getcwd()
start = 1
end = 11
vps = np.arange(start, end + 1)


def find_nearest(array, value, method="nearest"):
    # method: "smaller", "bigger", "nearest"
    array = np.asarray([0] + list(array))
    idx = (np.abs(array - value)).argmin()
    if method == "smaller":
        nearest = array[idx - 1] if array[idx] > value else array[idx]
    elif method == "bigger":
        nearest = array[idx + 1] if array[idx] < value else array[idx]
    else:
        nearest = array[idx]
    return nearest


def ecg_custom_process(raw, mne_events, vp, phase, sampling_rate=1024, method_clean=None, manual_correction=False):
    # Get data
    ecg_signal = raw.get_data(picks=['ECG']).flatten()

    # Use neurokit cleaning function:
    # 0.5 Hz high-pass butterworth filter (order = 5)
    # followed by powerline filtering (see signal_filter());  by default, powerline = 50
    if method_clean:  # "neurokit"
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=raw.info['sfreq'], method="neurokit")
    else:
        ecg_cleaned = ecg_signal.copy()

    # Get rpeaks
    peak_arr, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    rpeaks = rpeaks['ECG_R_Peaks']
    peak_arr = peak_arr['ECG_R_Peaks']

    # Manually correct rpeaks and add bad data spans
    if manual_correction:
        rpeak_events = pd.DataFrame()
        rpeak_events['Samples'] = rpeaks
        rpeak_events['MNE'] = [0] * len(rpeaks)
        rpeak_events['Condition'] = [6] * len(rpeaks)
        rpeak_events = rpeak_events.to_numpy()
        rpeak_annot = mne.annotations_from_events(events=rpeak_events, event_desc=event_dict_rev, sfreq=raw.info['sfreq'])
        raw.set_annotations(rpeak_annot)

        """
         USAGE: 
         - Use "a" to make annotations
         - "Add new label ("BAD_")
         - Activate "Draggable Edges"
         - To add a bad data span, mark the "BAD_"-category and left click and drag across the respective data span.
         - To remove a bad data span, mark the "BAD_"-category and right click on the respective data span.
         - To add a rpeak, mark the "rpeak"-category and left click on the respective location.
         - To remove a rpeak, mark the "rpeak"-category and left click and drag across the respective peak, then right click on this data span.
         - To end annotation close the window.
         - Use + / - to adapt scaling.
        """
        try:
            raw.plot(duration=20, scalings=1, block=True, title=f"Manually correct rpeaks and add bad data spans (VP: {vp}, Phase: {phase})")
            rpeaks = mne.events_from_annotations(raw, regexp='rpeak')[0][:, 0]
            print(f"Annotations at {rpeaks}")
        except:
            print("Manual annotation failed.")

        # Handle bad epochs
        try:
            bad_events = pd.DataFrame(mne.events_from_annotations(raw, regexp='BAD_')[0][:, 0], columns=['sample'])
            durations = []
            for annotation in raw.annotations:
                # annotation = raw.annotations[0]
                if annotation['description'] == 'BAD_':
                    durations.append(annotation['duration'] * raw.info['sfreq'])
                    print(f"Onset bad epoch after {annotation['onset']} s, Duration: {annotation['duration']} s")
            bad_events['duration'] = durations
            bad_events['duration'] = bad_events['duration'].astype("int")

            # Adapt rpeaks
            for idx_row in range(0, len(bad_events)):
                # idx_row = 0
                row = bad_events.iloc[idx_row]

                # Get nearest rpeaks around bad data span
                start = find_nearest(rpeaks, value=int(row['sample']), method="smaller")
                try:
                    end = find_nearest(rpeaks, value=int(row['sample'] + row['duration']), method="bigger")
                except:
                    end = len(ecg_cleaned)
                # Delete data before and after rpeak (keep first)
                ecg_cleaned[start:end] = np.nan
                ecg_cleaned = ecg_cleaned[~np.isnan(ecg_cleaned)]
                ecg_signal[start:end] = np.nan
                ecg_signal = ecg_signal[~np.isnan(ecg_signal)]

                # Adapt detected rpeaks and subtract removed samples
                rpeaks = rpeaks[(rpeaks <= start) | (rpeaks >= end)]
                rpeaks[rpeaks >= end] = rpeaks[rpeaks >= end] - int(end - start)
                rpeaks = np.unique(rpeaks)

                # Subtract removed samples from mne_events and bad_events
                mne_events['Samples'] = np.where(mne_events['Samples'] >= end, mne_events['Samples'] - int(end - start), mne_events['Samples'])
                bad_events['sample'] = np.where(bad_events['sample'] >= end, bad_events['sample'] - int(end - start), bad_events['sample'])
                print(f"Dropped {int(end - start)} samples")

            if len(rpeaks) == 0:
                raise Exception("Whole phase marked as bad")

            # Create new MNE raw file without bad epochs and with adapted rpeaks
            info = mne.create_info(ch_names=["ECG"], sfreq=raw.info['sfreq'], ch_types=['ecg'])
            raw = mne.io.RawArray(np.reshape(ecg_cleaned, (1, len(ecg_cleaned))), info)

            rpeak_events = pd.DataFrame()
            rpeak_events['Samples'] = rpeaks
            rpeak_events['MNE'] = [0] * len(rpeaks)
            rpeak_events['Condition'] = [6] * len(rpeaks)
            rpeak_events = rpeak_events.to_numpy()
            rpeak_annot = mne.annotations_from_events(events=rpeak_events, event_desc=event_dict_rev, sfreq=raw.info['sfreq'])
            raw.set_annotations(rpeak_annot)
            peak_arr = np.zeros(len(ecg_cleaned))
            peak_arr[rpeaks[:-1]] = 1

            # Control plot
            raw.plot(duration=20, scalings='auto', block=True, title=f"Check (VP: {vp}, Phase: {phase})")
        except:
            print("No bad events.")

    # Get heart rate for every sample in MNE raw using neurokit function
    try:
        rate = nk.signal_rate(rpeaks, sampling_rate=raw.info['sfreq'], desired_length=len(ecg_cleaned) + 1)[:-1]
        quality = nk.ecg_quality(ecg_cleaned, sampling_rate=raw.info['sfreq'])

        # Prepare output
        signals = pd.DataFrame({"ECG_Raw": ecg_signal,
                                "ECG_Clean": ecg_cleaned,
                                "ECG_Rate": rate,
                                "ECG_Quality": quality,
                                "ECG_R_Peaks": peak_arr})
        nk_info = rpeaks
        return signals, nk_info, mne_events
    except:
        return pd.DataFrame(), np.array([]), mne_events


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

    # Get ECG data
    try:
        file_path = os.path.join(dir_path, 'Data', 'VP_' + vp, 'ECG')
        folder = [item for item in os.listdir(file_path)][0]
        file_path = os.path.join(file_path, folder)

        # Get start time of recording
        start_time = ET.parse(os.path.join(file_path, 'unisens.xml')).getroot().attrib['timestampStart']
        start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%f')
        duration = float(ET.parse(os.path.join(file_path, 'unisens.xml')).getroot().attrib['duration'])
        sampling_rate = 1024
        lsb = 0.0026858184230029595
        baseline = 2048

        df_ecg = pd.read_csv(os.path.join(file_path, 'ecg.csv'), header=None, names=['ecg [unscaled]'])
        df_ecg['ecg [mV]'] = (df_ecg['ecg [unscaled]'] - baseline) * lsb
    except:
        print("no ecg file")
        continue

    # Add timestamps based on sampling rate and duration
    timestamps = pd.timedelta_range(start=0, end=timedelta(seconds=duration), periods=len(df_ecg)) / pd.to_timedelta(1, unit='ms')
    sr_check, fs = utils.get_sampling_rate(timestamps)
    timeseries = pd.date_range(start=start_time, end=start_time + timedelta(seconds=duration), periods=len(df_ecg))
    # df_ecg['time'] = timeseries
    df_ecg['time'] = pd.to_datetime(timeseries, unit="ns", utc=True)
    df_ecg["timestamp"] = df_ecg["time"].apply(lambda t: t.replace(tzinfo=None))
    df_ecg = df_ecg.drop(columns=["time"])

    # Get Events
    files = [item for item in os.listdir(os.path.join(dir_path, 'Data', 'VP_' + vp)) if (item.endswith(".csv"))]
    event_file = [file for file in files if "event" in file][0]
    df_event = pd.read_csv(os.path.join(dir_path, 'Data', 'VP_' + vp, event_file), sep=';', decimal='.')

    if pd.to_datetime(df_event.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26"):
        df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=2)
    else:
        df_event["timestamp"] = pd.to_datetime(df_event["timestamp"]) + timedelta(hours=1)
    df_event["timestamp"] = df_event["timestamp"].apply(lambda t: t.replace(tzinfo=None))

    # Add ECG and EDA markers:
    for physio in ["ECG", "EDA"]:
        # physio = "EDA"
        try:
            file_path_physio = os.path.join(dir_path, 'Data', 'VP_' + vp, physio)
            folder = [item for item in os.listdir(file_path_physio)][0]
            file_path_physio = os.path.join(file_path_physio, folder)
            start_time = ET.parse(os.path.join(file_path_physio, 'unisens.xml')).getroot().attrib['timestampStart']
            start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%f')
        except:
            print(f"no {physio} file")
            continue
        try:
            # Get marker file
            marker = pd.read_csv(os.path.join(file_path_physio, 'marker.csv'), sep=';', decimal='.', names=['sample'], header=None, usecols=[0])
            marker_samplingrate = 64
            marker_timepoints = [start_time + timedelta(seconds=marker["sample"][i] / marker_samplingrate) for i in np.arange(0, len(marker["sample"]))]
            df_marker = pd.DataFrame(marker_timepoints, columns=['timestamp'])
            df_marker['event'] = f"{physio}_marker"
            # Add markers to event file
            df_event = pd.concat([df_event, df_marker])
            df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)
        except:
            print(f"no {physio} marker file")
            continue

    # Get timepoint for resting state measurement
    df_rs = df_event.loc[df_event['timestamp'] < pd.to_datetime(df_event.loc[df_event["event"] == "EntryParticipantID", 'timestamp'].item() - timedelta(seconds=45), unit="ns")]
    df_rs = df_rs.loc[df_rs['event'].str.contains("marker")].reset_index(drop=True)
    df_rs = df_rs.drop_duplicates(subset=["event"], keep="last")
    if ("ECG_marker" in df_rs["event"].unique()) and ("EDA_marker" in df_rs["event"].unique()):
        for idx_row, row in df_rs.iterrows():
            # idx_row = 0
            if idx_row == len(df_rs) - 1:
                break
            rs_start = df_rs.loc[idx_row + 1, "timestamp"] + timedelta(seconds=5)
            rs_end = rs_start + timedelta(seconds=30)
            break
    else:
        rs_start = df_rs.loc[0, "timestamp"] + timedelta(seconds=10)
        rs_end = rs_start + timedelta(seconds=45)

    df_event = pd.concat([df_event, pd.DataFrame([[rs_start, "resting state"]], columns=["timestamp", "event"])])
    df_event = df_event.loc[~(df_event['event'].str.contains("marker"))]
    df_event = df_event.sort_values(by=["timestamp"]).reset_index(drop=True)

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
    df_rs = df_event.loc[df_event["event"].str.contains("resting")]
    df_rs["duration"] = 45
    dfs.append(df_rs)
    
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

    # Add event dict to describe events in MNE
    event_dict = {'resting state': 1,
                  'habituation_living': 2,
                  'habituation_dining': 3,
                  'habituation_office': 4,
                  'test_living': 12,
                  'test_dining': 13,
                  'test_office': 14,
                  'test_terrace': 15,
                  'unfriendly-interaction': 21,
                  'friendly-interaction': 22,
                  'neutral-interaction': 23,
                  'bryan-clicked': 31,
                  'emanuel-clicked': 32,
                  'ettore-clicked': 33,
                  'oskar-clicked': 34,
                  'rpeak': 6}
    event_dict_rev = {v: k for k, v in event_dict.items()}

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

    # Merge "name" and "event"-column to df_ecg
    df_ecg = pd.merge_asof(df_ecg, df_event[["timestamp", "name"]], on="timestamp", direction="backward").reset_index(drop=True)
    tolerance = timedelta(milliseconds=(1 / sampling_rate) * 1000)
    df_ecg = pd.merge_asof(df_ecg, df_event[["timestamp", "name", "event"]], by="name", on="timestamp", direction="nearest", tolerance=tolerance)

    # Iterate through experimental phases and check ECG data
    for idx_row, row in df_event.iterrows():
        # idx_row = 5
        # row = df_event.iloc[idx_row]

        phase = row['name']
        print(f"Phase: {phase}")
        if not "Interaction" in phase:
            continue

        # Get start and end point of phase
        start_phase = row['timestamp']
        end_phase = row['timestamp'] + pd.to_timedelta(row['duration'], unit="S")

        # Cut ECG dataset
        df_ecg_subset = df_ecg.loc[(df_ecg["timestamp"] >= start_phase) & (df_ecg["timestamp"] < end_phase + timedelta(seconds=1))]
        df_ecg_subset = df_ecg_subset.loc[df_ecg_subset['name'] == phase].reset_index(drop=True)

        # Create MNE events file
        mne_events = pd.DataFrame()
        mne_events['Samples'] = list(df_ecg_subset.dropna(subset="event").index)
        mne_events['MNE'] = [0] * len(df_ecg_subset.dropna(subset="event"))
        mne_events['Condition'] = df_ecg_subset.dropna(subset="event")['event'].to_list()

        # Create MNE info file and MNE raw file
        info = mne.create_info(ch_names=["ECG"], sfreq=sampling_rate, ch_types=['ecg'])
        # raw = mne.io.RawArray(np.reshape(np.array(df_ecg[["ecg [mV]", "event"]]), (2, len(df_ecg))), info)
        data = np.reshape(df_ecg_subset["ecg [mV]"].to_numpy(), (1, len(df_ecg_subset["ecg [mV]"])))
        raw = mne.io.RawArray(data, info)

        # 2 Hz high-pass filter in order to remove slow signal drifts
        raw.filter(picks=['ECG'], l_freq=10, h_freq=None)
        # raw.plot(duration=20, scalings='auto', block=True)

        # Get annotations from events and add duration
        try:
            annot_from_events = mne.annotations_from_events(events=mne_events.to_numpy(), event_desc=event_dict_rev,
                                                            sfreq=raw.info['sfreq'], orig_time=raw.info['meas_date'])
            annot_events = raw.annotations
            duration = df_event.loc[idx_row, "duration"]
            annot_events.append(onset=annot_from_events[0]['onset'], duration=duration, description=annot_from_events[0]['description'])
            raw.set_annotations(annot_events)

            # Add duration to MNE events file
            mne_events['Durations'] = annot_events.duration
            duration_pre = annot_events.duration[0]
            mne_events['Condition'] = mne_events['Condition'].astype(int)
        except Exception as e:
            print(e)
            continue

        # Use customized neurokit function to analyze ECG
        try:
            signals, peak_detection, mne_events = ecg_custom_process(raw, mne_events, vp=vp, phase=phase, sampling_rate=raw.info['sfreq'],
                                                                     method_clean="neurokit", manual_correction=True)
        except Exception as e:
            print(e)
            signals = []

        if len(signals) == 0:
            df_hr_temp = pd.DataFrame({'VP': [int(vp)],
                                       'Phase': [phase],
                                       'event_id': [idx_row],
                                       'HR (Mean)': [np.nan],
                                       'HR (Std)': [np.nan],
                                       'HRV (MeanNN)': [np.nan],
                                       'HRV (RMSSD)': [np.nan],
                                       'HRV (LF)': [np.nan],
                                       'HRV (HF)': [np.nan],
                                       'Proportion Usable Data': [0],
                                       'Duration': [0]})
            df_hr_temp.to_csv(os.path.join(dir_path, 'Data', 'hr.csv'), decimal='.', sep=';', index=False, mode='a',
                              header=not (os.path.exists(os.path.join(dir_path, 'Data', 'hr.csv'))))
            plt.close()
            continue

        if "Interaction" in phase:
            df_ecg_subset_save = signals.copy()
            df_ecg_subset_save["timestamp"] = df_ecg_subset["timestamp"]

            start_ecg = df_ecg_subset_save.loc[0, "ECG_Rate"]
            df_ecg_subset_save["ECG"] = df_ecg_subset_save["ECG_Rate"] - start_ecg

            start = df_ecg_subset_save.loc[0, "timestamp"]
            df_ecg_subset_save["time"] = pd.to_timedelta(df_ecg_subset_save["timestamp"] - start)
            df_ecg_subset_save = df_ecg_subset_save.set_index("time")
            df_ecg_subset_save = df_ecg_subset_save.resample("0.1S").mean()
            df_ecg_subset_save = df_ecg_subset_save.reset_index()
            df_ecg_subset_save["time"] = df_ecg_subset_save["time"].dt.total_seconds()
            df_ecg_subset_save["VP"] = int(vp)
            df_ecg_subset_save["event"] = phase
            df_ecg_subset_save = df_ecg_subset_save[["VP", "event", "time", "ECG"]]
            df_ecg_subset_save.to_csv(os.path.join(dir_path, 'Data', 'hr_interaction.csv'), decimal='.', sep=';',
                                      index=False, mode='a', header=not (os.path.exists(os.path.join(dir_path, 'Data', 'hr_interaction.csv'))))

        # Adapt duration in mne_events
        for idy_row, row in mne_events.iterrows():
            # idy_row = 0
            # row = mne_events.iloc[idy_row]
            if idy_row < len(mne_events) - 2:
                duration_post = (mne_events.loc[idy_row+1, 'Samples'] - mne_events.loc[idy_row, 'Samples']) / sampling_rate
                mne_events.loc[idy_row, 'Durations'] = duration_post
            else:
                duration_post = mne_events.loc[idy_row, 'Durations']
                mne_events.loc[idy_row, 'Durations'] = duration_post

        usable_data = duration_post/duration_pre

        # Create epochs for neurokit
        epochs = nk.epochs_create(signals, events=mne_events['Samples'].to_list(), sampling_rate=raw.info['sfreq'],
                                  epochs_start=0, epochs_end=mne_events['Durations'].to_list(),
                                  event_labels=mne_events['Condition'].to_list(), event_conditions=mne_events['Condition'].to_list())

        # HRV: Get MeanNN (mean of RR intervals), RMSSD (square root of the mean of the sum of successive differences between adjacent RR intervals), LF and HF
        try:
            hrv = nk.hrv(signals['ECG_R_Peaks'].to_numpy(), sampling_rate=sampling_rate)
        except Exception as e:
            hrv = pd.DataFrame([["", "", "", ""]], columns=['HRV_MeanNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF'])
            print(e)

        # HR: Get Mean and Std
        # Save as dataframe
        df_hr_temp = pd.DataFrame({'VP': [int(vp)],
                                   'Phase': [phase],
                                   'event_id': [idx_row],
                                   'HR (Mean)': [np.mean(signals['ECG_Rate'])],
                                   'HR (Std)': [np.std(signals['ECG_Rate'])],
                                   'HRV (MeanNN)': [hrv['HRV_MeanNN'][0]],
                                   'HRV (RMSSD)': [hrv['HRV_RMSSD'][0]],
                                   'HRV (LF)': [hrv['HRV_LF'][0]],
                                   'HRV (HF)': [hrv['HRV_HF'][0]],
                                   'Proportion Usable Data': [usable_data],
                                   'Duration': [duration_post]})
        df_hr_temp.to_csv(os.path.join(dir_path, 'Data', 'hr.csv'), decimal='.', sep=';', index=False, mode='a',
                          header=not (os.path.exists(os.path.join(dir_path, 'Data', 'hr.csv'))))

        plt.close()

# Add Subject Data
df_hr = pd.read_csv(os.path.join(dir_path, 'Data', 'hr.csv'), decimal='.', sep=';')
df_hr = df_hr.iloc[:, 0:11]
df_hr = df_hr.dropna(subset=['HR (Mean)'])

# Get conditions
dfs_hr = []
for vp in vps:
    # vp = vps[3]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    df_hr_vp = df_hr.loc[df_hr["VP"] == int(vp)]

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

    df_hr_vp["Condition"] = ""
    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        room = row["Rooms"]
        role = row["Role"]
        df_hr_vp.loc[df_hr_vp["Phase"].str.contains(room), "Condition"] = role
    # df_hr_vp = df_hr_vp[["VP", "Phase", "Condition", ""]]
    dfs_hr.append(df_hr_vp)

df_hr = pd.concat(dfs_hr)

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_hr = df_hr.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                               'SSQ', 'SSQ-N', 'SSQ-O', 'SSQ-D', 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS',
                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_hr = df_hr.drop(columns=['ID'])
df_hr.to_csv(os.path.join(dir_path, 'Data', 'hr.csv'), decimal='.', sep=';', index=False)
