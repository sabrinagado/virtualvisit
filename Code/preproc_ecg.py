# =============================================================================
# ECG
# sensor: movisens
# study: Virtual Visit
# =============================================================================
import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter
import neurokit2 as nk
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

from Code.toolbox import utils

from Code import preproc_scores, preproc_ratings, preproc_behavior

import mne

plt.ion()
matplotlib.use('QtAgg')

args = sys.argv
wave = int(args[1])
filepath = args[2]


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


def ecg_custom_process(raw, mne_events, event_dict_rev, vp, phase, sampling_rate=1024, method_clean=None, manual_correction=False):
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
    if len(rpeaks) > 3:
        mean_dist = np.mean([j - i for i, j in zip(rpeaks[:-1], rpeaks[1:])])
        sd_dist = np.std([j - i for i, j in zip(rpeaks[:-1], rpeaks[1:])])
        local_maxima = signal.find_peaks(ecg_cleaned, distance=mean_dist-np.min([2*sd_dist, 150]))[0]
        rpeaks = np.unique(np.concatenate((rpeaks, local_maxima)))

    # Manually correct rpeaks and add bad data spans
    if manual_correction:
        rpeak_events = pd.DataFrame()
        rpeak_events['Samples'] = rpeaks
        rpeak_events['MNE'] = [0] * len(rpeaks)
        rpeak_events['Condition'] = [100] * len(rpeaks)
        rpeak_events = rpeak_events.to_numpy()
        rpeak_annot = mne.annotations_from_events(events=rpeak_events, event_desc=event_dict_rev, sfreq=raw.info['sfreq'])
        raw.set_annotations(rpeak_annot, emit_warning=False, verbose=False)

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
            raw.plot(duration=20, scalings=1, block=True, title=f"Manually correct rpeaks and add bad data spans (VP: {vp}, Phase: {phase})", verbose=False)
            rpeaks = mne.events_from_annotations(raw, regexp='rpeak', verbose=False)[0][:, 0]
            # print(f"Annotations at {rpeaks}")
        except:
            print("Manual annotation failed.")

        # Handle bad epochs
        try:
            bad_events = pd.DataFrame(mne.events_from_annotations(raw, regexp='BAD_', verbose=False)[0][:, 0], columns=['sample'])
            durations = []
            for annotation in raw.annotations:
                # annotation = raw.annotations[0]
                if annotation['description'] == 'BAD_':
                    durations.append(annotation['duration'] * raw.info['sfreq'])
                    # print(f"Onset bad epoch after {annotation['onset']} s, Duration: {annotation['duration']} s")
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
                # print(f"Dropped {int(end - start)} samples")

            # Create new MNE raw file without bad epochs and with adapted rpeaks
            info = mne.create_info(ch_names=["ECG"], sfreq=raw.info['sfreq'], ch_types=['ecg'])
            raw = mne.io.RawArray(np.reshape(ecg_cleaned, (1, len(ecg_cleaned))), info, verbose=False)

            rpeak_events = pd.DataFrame()
            rpeak_events['Samples'] = rpeaks
            rpeak_events['MNE'] = [0] * len(rpeaks)
            rpeak_events['Condition'] = [100] * len(rpeaks)
            rpeak_events = rpeak_events.to_numpy()
            rpeak_annot = mne.annotations_from_events(events=rpeak_events, event_desc=event_dict_rev, sfreq=raw.info['sfreq'])
            raw.set_annotations(rpeak_annot, emit_warning=False, verbose=False)
            peak_arr = np.zeros(len(ecg_cleaned))
            peak_arr[rpeaks[:-1]] = 1

            # Control plot
            raw.plot(duration=20, scalings='auto', block=True, title=f"Check (VP: {vp}, Phase: {phase})", verbose=False)
        except:
            pass

        # If peaks < 4 --> no HR
        if len(rpeaks) < 4:
            return pd.DataFrame(), np.array([]), mne_events

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


def get_hr(vps, filepath, wave, df_scores):
    df_hr = pd.DataFrame()
    df_hr_interaction = pd.DataFrame()
    for vp in tqdm(vps):
        # vp = vps[0]
        vp = f"0{vp}" if vp < 10 else f"{vp}"
        # print(f"VP: {vp}")

        df_hr_vp = pd.DataFrame()
        df_hr_interaction_vp = pd.DataFrame()

        # Get ECG data
        try:
            file_path = os.path.join(filepath, 'VP_' + vp, 'ECG')
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

        # Get Conditions
        try:
            df_roles = preproc_behavior.get_conditions(vp, filepath)
        except:
            print(f"no conditions file for VP {vp}")
            continue

        # Get Events
        try:
            df_events_vp, events = preproc_behavior.get_events(vp, filepath, wave, df_roles)
        except:
            print(f"no events file for VP {vp}")
            continue

        df_events_vp = df_events_vp.loc[df_events_vp["duration"] > 1]

        # Add event dict to describe events in MNE
        codes, uniques = pd.factorize(df_events_vp['event'], sort=True)
        array_event_ids = np.arange(0, len(uniques))
        event_dict = {uniques[i]: array_event_ids[i] for i in range(len(uniques))}
        event_dict['rpeak'] = 100
        event_dict_rev = {v: k for k, v in event_dict.items()}

        # Add "event" columns (for MNE)
        df_events_vp["name"] = df_events_vp["event"]
        df_events_vp["event"] = codes

        # # Merge "name" and "event"-column to df_ecg
        df_ecg = pd.merge_asof(df_ecg, df_events_vp[["timestamp", "name"]], on="timestamp", direction="backward").reset_index(drop=True)
        tolerance = timedelta(milliseconds=(1 / sampling_rate) * 1000)
        df_ecg = pd.merge_asof(df_ecg, df_events_vp[["timestamp", "name", "event"]], by="name", on="timestamp", direction="nearest", tolerance=tolerance)

        # Iterate through experimental phases and check ECG data
        for idx_row, row in df_events_vp.iterrows():
            # idx_row = 21
            # row = df_events_vp.iloc[idx_row]
            phase = row['name']
            # print(f"Phase: {phase}")

            # Get start and end point of phase
            start_phase = row['timestamp']
            end_phase = row['timestamp'] + pd.to_timedelta(row['duration'], unit="S")

            # Cut ECG dataset
            df_ecg_subset = df_ecg.loc[(df_ecg["timestamp"] >= start_phase) & (df_ecg["timestamp"] < end_phase + timedelta(seconds=1))]
            df_ecg_subset["name"] = phase
            # df_ecg_subset = df_ecg_subset.loc[df_ecg_subset['name'] == phase].reset_index(drop=True)

            # Create MNE events file
            mne_events = pd.DataFrame()
            mne_events['Samples'] = list(df_ecg_subset.dropna(subset="event").index)
            mne_events['MNE'] = [0] * len(df_ecg_subset.dropna(subset="event"))
            mne_events['Condition'] = df_ecg_subset.dropna(subset="event")['event'].to_list()
            mne_events = mne_events.iloc[:1]

            # Create MNE info file and MNE raw file
            info = mne.create_info(ch_names=["ECG"], sfreq=sampling_rate, ch_types=['ecg'])
            # raw = mne.io.RawArray(np.reshape(np.array(df_ecg[["ecg [mV]", "event"]]), (2, len(df_ecg))), info)
            data = np.reshape(df_ecg_subset["ecg [mV]"].to_numpy(), (1, len(df_ecg_subset["ecg [mV]"])))
            raw = mne.io.RawArray(data, info, verbose=False)

            # 2 Hz high-pass filter in order to remove slow signal drifts
            raw.filter(picks=['ECG'], l_freq=5, h_freq=None, verbose=False)
            # raw.plot(duration=20, scalings='auto', block=True)

            # Get annotations from events and add duration
            try:
                annot_from_events = mne.annotations_from_events(events=mne_events.to_numpy(), event_desc=event_dict_rev,
                                                                sfreq=raw.info['sfreq'], orig_time=raw.info['meas_date'], verbose=False)
                annot_events = raw.annotations
                duration = df_events_vp.loc[idx_row, "duration"]
                annot_events.append(onset=annot_from_events[0]['onset'], duration=duration, description=annot_from_events[0]['description'])
                raw.set_annotations(annot_events, emit_warning=False, verbose=False)

                # Add duration to MNE events file
                mne_events['Durations'] = annot_events.duration
                duration_pre = annot_events.duration[0]
                mne_events['Condition'] = mne_events['Condition'].astype(int)
            except Exception as e:
                print(e)
                continue

            duration_pre = raw.times.max()

            # Use customized neurokit function to analyze ECG
            try:
                signals, peak_detection, mne_events = ecg_custom_process(raw, mne_events, event_dict_rev, vp=vp, phase=phase, sampling_rate=raw.info['sfreq'],
                                                                         method_clean="neurokit", manual_correction=True)
            except Exception as e:
                print("Interrupted!")
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
                                           'HRV (HF_nu)': [np.nan],
                                           'Proportion Usable Data': [0],
                                           'Duration': [0]})
                df_hr_vp = pd.concat([df_hr_vp, df_hr_temp])
                plt.close()
                continue

            if ("Interaction" in phase) or ("Click" in phase):
                df_ecg_subset_save = signals.copy()
                df_ecg_subset_save["timestamp"] = df_ecg_subset["timestamp"].reset_index(drop=True)

                start_ecg = df_ecg_subset_save.loc[0, "ECG_Rate"]
                df_ecg_subset_save["ECG"] = df_ecg_subset_save["ECG_Rate"] - start_ecg

                start = df_ecg_subset_save.loc[0, "timestamp"]
                df_ecg_subset_save["time"] = pd.to_timedelta(df_ecg_subset_save["timestamp"] - start)
                df_ecg_subset_save = df_ecg_subset_save.set_index("time")
                df_ecg_subset_save = df_ecg_subset_save.resample("0.1S").mean(numeric_only=True)
                df_ecg_subset_save = df_ecg_subset_save.reset_index()
                df_ecg_subset_save["time"] = df_ecg_subset_save["time"].dt.total_seconds()
                df_ecg_subset_save["VP"] = int(vp)
                df_ecg_subset_save["event"] = phase
                df_ecg_subset_save = df_ecg_subset_save[["VP", "event", "time", "ECG"]]
                df_hr_interaction_vp = pd.concat([df_hr_interaction_vp, df_ecg_subset_save])

            # duration
            duration_post = len(signals["ECG_Clean"]) / sampling_rate

            # HRV
            if duration_post >= 30:
                # Cut signal to 30 seconds (to make phases comparable)
                signals = signals[0:sampling_rate * 30]
                # hrv = nk.hrv(signals['ECG_R_Peaks'].to_numpy(), sampling_rate=sampling_rate)
                hrv_time = nk.hrv_time(signals['ECG_R_Peaks'].to_numpy(), sampling_rate=sampling_rate)
                hrv_freq = nk.hrv_frequency(signals['ECG_R_Peaks'].to_numpy(), sampling_rate=sampling_rate, psd_method='fft')
                # Normative units = 100 * (HF absolute power / (total absolute power − very low frequency absolute power (0–0.003 Hz))
                hrv_freq = hrv_freq.fillna(value=0)
                hrv_freq["HRV_HF_nu"] = 100 * (hrv_freq["HRV_HF"] / (hrv_freq["HRV_TP"] - hrv_freq["HRV_ULF"]))
            else:
                hrv_time = pd.DataFrame([["", ""]], columns=['HRV_MeanNN', 'HRV_RMSSD'])
                hrv_freq = pd.DataFrame([["", "", ""]], columns=['HRV_LF', 'HRV_HF', 'HRV_HF_nu'])

            # HR: Get Mean and Std
            # Save as dataframe
            df_hr_temp = pd.DataFrame({'VP': [int(vp)],
                                       'Phase': [phase],
                                       'event_id': [idx_row],
                                       'HR (Mean)': [np.mean(signals['ECG_Rate'])],
                                       'HR (Std)': [np.std(signals['ECG_Rate'])],
                                       'HRV (MeanNN)': [hrv_time['HRV_MeanNN'][0]],
                                       'HRV (RMSSD)': [hrv_time['HRV_RMSSD'][0]],
                                       'HRV (LF)': [hrv_freq['HRV_LF'][0]],
                                       'HRV (HF)': [hrv_freq['HRV_HF'][0]],
                                       'HRV (HF_nu)': [hrv_freq["HRV_HF_nu"][0]],
                                       'Proportion Usable Data': [round(np.max([duration_post / duration_pre, 1]), 2)],
                                       'Duration': [duration_post]})
            df_hr_vp = pd.concat([df_hr_vp, df_hr_temp])
            plt.close()

        # Add Conditions
        for idx_row, row in df_roles.iterrows():
            # idx_row = 0
            # row = df_roles.iloc[idx_row, :]
            room = row["Rooms"]
            role = row["Role"]
            character = row["Character"]

            df_hr_vp["Phase"] = df_hr_vp["Phase"].str.replace(character, role.capitalize())
            if wave == 1:
                df_hr_vp.loc[df_hr_vp["Phase"].str.contains(room), "Condition"] = role
            df_hr_vp.loc[df_hr_vp["Phase"].str.contains(role.capitalize()), "Condition"] = role
            df_hr_interaction_vp["event"] = df_hr_interaction_vp["event"].str.replace(character, role.capitalize())

        # Add Participant Data
        df_hr_vp = df_hr_vp.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                       'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                       'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                       'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                       'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
        df_hr_vp = df_hr_vp.drop(columns=['ID'])

        df_hr_interaction_vp = df_hr_interaction_vp.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                                               'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D','SSQ-diff',
                                                               'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP','MPS-SocP', 'MPS-SelfP',
                                                               'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                                               'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
        df_hr_interaction_vp = df_hr_interaction_vp.drop(columns=['ID'])

        df_hr = pd.concat([df_hr, df_hr_vp])
        df_hr_interaction = pd.concat([df_hr_interaction, df_hr_interaction_vp])
        df_hr.to_csv(os.path.join(filepath, 'hr.csv'), decimal='.', sep=';', index=False)
        df_hr_interaction.to_csv(os.path.join(filepath, 'hr_interaction.csv'), decimal='.', sep=';', index=False)
    return df_hr, df_hr_interaction


if __name__ == '__main__':
    # wave = 2
    # dir_path = os.getcwd()
    # filepath = os.path.join(dir_path, f'Data-Wave{wave}')
    if wave == 1:
        problematic_subjects = [1, 3, 12, 19, 33, 45, 46] + [7, 56, 61, 64]  # 7, 56, 61 and 64 have bad ECG signal quality
    elif wave == 2:
        problematic_subjects = [1, 2, 3, 4, 20, 29, 64] + [9, 11, 17, 29, 44, 47]

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

    try:
        df_hr = pd.read_csv(os.path.join(filepath, 'hr.csv'), decimal='.', sep=';')
        finished_subjects = list(df_hr["VP"].unique())
        print("existing hr.csv found, only adding new VPs")
    except:
        finished_subjects = []
        # print("no existing hr.csv found")

    vps = [vp for vp in vps if not vp in problematic_subjects + finished_subjects]

    df_hr, df_hr_interaction = get_hr(vps, filepath, wave, df_scores)
