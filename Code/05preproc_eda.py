# =============================================================================
# EDA
# sensor: movisens
# study: Virtual Visit
# =============================================================================
import os
import numpy as np
import pandas as pd
from scipy import signal
import scipy
import neurokit2 as nk
from statsmodels.tsa.api import SimpleExpSmoothing
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

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


def eda_custom_process(raw, vp, phase, sampling_rate=32, pipeline=None, correction=["manual", "sd", "ffat"], amplitude_min=0.02):
    # Get data
    # raw = mne.io.RawArray(data, info)
    # correction=["fft"]
    eda_signal = raw.get_data(picks=['EDA']).flatten()
    bad_events = pd.DataFrame(columns=['sample', 'duration'], dtype="int")

    if "manual" in correction:
        try:
            raw.plot(duration=20, scalings=1, block=True,
                     title=f"Manually add bad data spans (VP: {vp}, Phase: {phase})")
        except:
            print("Manual annotation failed.")

        # Handle bad epochs
        try:
            bad_events_annot = pd.DataFrame(mne.events_from_annotations(raw, regexp='BAD_')[0][:, 0], columns=['sample'])
            durations = []
            onsets = []
            for annotation in raw.annotations:
                # annotation = raw.annotations[0]
                if annotation['description'] == 'BAD_':
                    durations.append(annotation['duration'] * raw.info['sfreq'])
                    onsets.append(annotation['onset'])
                    print(f"Onset bad epoch after {annotation['onset']} s, Duration: {annotation['duration']} s")
            bad_events['duration'] = durations
            bad_events['duration'] = bad_events['duration'].astype("int")
            bad_events['duration'] = bad_events['duration'] / raw.info["sfreq"]
            bad_events['sample'] = bad_events_annot['sample']
            bad_events['onset'] = bad_events['sample'] / raw.info["sfreq"]
            bad_events = bad_events.reset_index(drop=True)
        except:
            print("No bad events manually added.")

    if "fft" in correction:
        for window_seconds, frequency_threshold, power_threshold in zip([2, 5], [10, 10], [15, 100]):
            # window_seconds, power_threshold = 2, 15
            window_samples = int(window_seconds * raw.info['sfreq'])
            fft = np.apply_along_axis(my_fft, 1, rolling_windows(eda_signal, window=window_samples))
            df_fft = pd.DataFrame(fft[:, 1:])
            # df_fft.index = df_fft.index/raw.info['sfreq']
            # ax = sns.heatmap(df_fft.transpose(), cmap="viridis", cbar_kws={'label': 'Power'})
            # ax.set_xlabel(f"{window_seconds} s Rolling Windows")
            # # ax.set_ylabel("Frequency Bins")
            # ax.invert_yaxis()
            # plt.tight_layout()
            # plt.savefig(os.path.join(save_path, f"heatmap_fft_VP{vp}_{round(window_seconds, 2)}.png"), dpi=300)
            # plt.close()

            bad_events_fft = pd.DataFrame(columns=['sample', 'duration'], dtype="int")
            for idx_row, row in df_fft.iterrows():
                # idx_row = 404
                # row = df_fft.iloc[idx_row, :]
                if row[:frequency_threshold].mean() > power_threshold:
                    start = idx_row
                    bad_event = pd.DataFrame({'sample': [idx_row], 'duration': [window_samples], 'onset': [idx_row]})
                    if (len(bad_events_fft) > 0) and start < int(bad_events_fft.loc[len(bad_events_fft) - 1, "sample"] + bad_events_fft.loc[len(bad_events_fft) - 1, "duration"]):
                        bad_events_fft.loc[len(bad_events_fft) - 1, "duration"] += 1
                    else:
                        bad_events_fft = pd.concat([bad_events_fft, bad_event], ignore_index=True, axis=0)
            bad_events_fft['duration'] = bad_events_fft['duration'] / raw.info["sfreq"]
            bad_events_fft['onset'] = bad_events_fft['sample'] / raw.info["sfreq"]

            bad_events = pd.concat([bad_events, bad_events_fft], ignore_index=True, axis=0)
            bad_events = bad_events.sort_values(by="sample")

            # Get annotations from events and add duration
            try:
                annot_events = raw.annotations
                for idx_row, row in bad_events.iterrows():
                    # idx_row = 0
                    # row = bad_events.iloc[idx_row, :]
                    annot_events.append(onset=row['onset'], duration=row['duration'], description=['BAD_'])
                raw.set_annotations(annot_events)
            except Exception as e:
                print(e)

    if "sd" in correction:
        sd_signal = np.std(eda_signal)
        sc_signal = sign_change(eda_signal)/raw.times.max()

        for window_seconds, threshold_sd, threshold_sc in zip([1, 5], [0.4, 0.4], [1.1, 1.1]):
            # window_seconds, threshold_sd, threshold_sc = 1, 0.4, 1.1
            print(window_seconds, threshold_sd, threshold_sc)
            window_samples = int(window_seconds * raw.info['sfreq'])

            # bad events for SD
            sd = np.apply_along_axis(np.std, 1, rolling_windows(eda_signal, window=window_samples))
            df_sd = pd.DataFrame(sd)

            bad_events_sd = pd.DataFrame(columns=['sample', 'duration'], dtype="int")
            for idx_row, row in df_sd.iterrows():
                # idx_row = 404
                # row = df_sd.iloc[idx_row, :]
                if row.item() > threshold_sd * sd_signal:
                    start = idx_row
                    bad_event = pd.DataFrame({'sample': [idx_row], 'duration': [window_samples], 'onset': [idx_row]})
                    if (len(bad_events_sd) > 0) and start < int(bad_events_sd.loc[len(bad_events_sd) - 1, "sample"] + bad_events_sd.loc[len(bad_events_sd) - 1, "duration"]):
                        bad_events_sd.loc[len(bad_events_sd) - 1, "duration"] += 1
                    else:
                        bad_events_sd = pd.concat([bad_events_sd, bad_event], ignore_index=True, axis=0)
            bad_events_sd['duration'] = bad_events_sd['duration'] / raw.info["sfreq"]
            bad_events_sd['onset'] = bad_events_sd['sample'] / raw.info["sfreq"]
            bad_events = pd.concat([bad_events, bad_events_sd], ignore_index=True, axis=0)
            bad_events = bad_events.sort_values(by="sample")

            # bad events for sign changes
            sign_changes = np.apply_along_axis(sign_change, 1, rolling_windows(eda_signal, window=window_samples))
            df_sc = pd.DataFrame(sign_changes / window_seconds)

            bad_events_sc = pd.DataFrame(columns=['sample', 'duration'], dtype="int")
            for idx_row, row in df_sd.iterrows():
                # idx_row = 404
                # row = df_sd.iloc[idx_row, :]
                if row.item() > threshold_sc * sc_signal:
                    start = idx_row
                    bad_event = pd.DataFrame({'sample': [idx_row], 'duration': [window_samples], 'onset': [idx_row]})
                    if (len(bad_events_sc) > 0) and start < int(bad_events_sc.loc[len(bad_events_sc) - 1, "sample"] + bad_events_sc.loc[len(bad_events_sc) - 1, "duration"]):
                        bad_events_sc.loc[len(bad_events_sc) - 1, "duration"] += 1
                    else:
                        bad_events_sc = pd.concat([bad_events_sc, bad_event], ignore_index=True, axis=0)
            bad_events_sc['duration'] = bad_events_sc['duration'] / raw.info["sfreq"]
            bad_events_sc['onset'] = bad_events_sc['sample'] / raw.info["sfreq"]
            bad_events = pd.concat([bad_events, bad_events_sc], ignore_index=True, axis=0)
            bad_events = bad_events.sort_values(by="sample")

            # Get annotations from events and add duration
            try:
                annot_events = raw.annotations
                for idx_row, row in bad_events.iterrows():
                    # idx_row = 0
                    # row = bad_events.iloc[idx_row, :]
                    annot_events.append(onset=row['onset'], duration=row['duration'], description=['BAD_'])
                raw.set_annotations(annot_events)
            except Exception as e:
                print(e)

    bad_events = pd.DataFrame({"onset": raw.annotations.onset, "duration": raw.annotations.duration})
    bad_events["end"] = bad_events["onset"] + bad_events["duration"]

    null_signal = pd.DataFrame(np.stack((raw.times, raw._data[0]), axis=1), columns=["times", "EDA"])
    null_signal = null_signal.loc[null_signal['EDA'] == 0].reset_index()
    if len(null_signal) > 0:
        onsets = []
        ends = []
        for row_idx, row in null_signal.iterrows():
            # row_idx = 14
            # row = null_signal.iloc[row_idx, :]
            if row_idx == len(null_signal) - 1:
                break
            if row_idx == 0:
                continue

            if null_signal.iloc[row_idx, 0] != null_signal.iloc[row_idx + 1, 0] - 1:
                ends = ends + [null_signal.iloc[row_idx, 1]]
            if null_signal.iloc[row_idx - 1, 0] == null_signal.iloc[row_idx, 0] - 1:
                continue
            else:
                onsets = onsets + [null_signal.iloc[row_idx, 1]]

        null_signal = pd.DataFrame(np.stack((onsets, ends), axis=1), columns=["onset", "end"])
        null_signal['duration'] = null_signal['end'] - null_signal['onset']
        null_signal = null_signal[["onset", "duration", "end"]]
        null_signal = null_signal.loc[null_signal["duration"] > 0]

        bad_events = pd.concat([bad_events, null_signal])
        bad_events = bad_events.sort_values(by="onset").reset_index(drop=True)

    if len(bad_events) > 1:
        bad_events = merge_intervals(bad_events[["onset", "end"]].to_numpy())
        bad_events.columns = ["onset", "end"]
        bad_events["duration"] = bad_events['end'] - bad_events['onset']

        end_prev = 0
        for idx_row, row in bad_events.iterrows():
            # idx_row = 2
            # row = bad_events.iloc[idx_row, :]
            start = row['onset']
            if (start - end_prev) < 3:
                bad_events.loc[idx_row, "onset"] = end_prev
            end_prev = row['end']

        bad_events = merge_intervals(bad_events[["onset", "end"]].to_numpy())
        bad_events.columns = ["onset", "end"]
        bad_events["duration"] = bad_events['end'] - bad_events['onset']

    if len(bad_events) > 0:
        # Get annotations from events and add duration
        annotations = mne.Annotations(bad_events["onset"], bad_events["duration"], ['BAD_'] * len(bad_events))
        raw.set_annotations(annotations)

    # raw.plot(duration=20, scalings=1, block=True, title=f"Check bad data spans after automatic identification: VP {vp}, Phase {phase}")

    try:
        # Get signal
        eda_signal = raw.get_data(picks=['EDA'], reject_by_annotation="omit").flatten()

        if pipeline == "neurokit":
            # 3 Hz low-pass butterworth filter (order = 4)
            eda_filtered = nk.eda_clean(eda_signal, sampling_rate=sampling_rate, method="neurokit")

            # Exponential Smoothing
            fit_ses = SimpleExpSmoothing(eda_filtered, initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)
            eda_filtered = fit_ses.fittedvalues

            # Decompose into phasic and tonic components
            eda_decomposed = nk.eda_phasic(eda_filtered, sampling_rate=sampling_rate, method='smoothmedian')

            # Find peaks in phasic signal
            peak_signal, info = nk.eda_peaks(
                eda_decomposed["EDA_Phasic"].values,
                sampling_rate=sampling_rate, method="neurokit", amplitude_min=amplitude_min)
            info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

            # Store
            signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_filtered})
            signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

            scr_onsets = signals.loc[(signals["SCR_Onsets"] == 1)].index
            scr_peaks = signals.loc[(signals["SCR_Peaks"] == 1) & (signals["SCR_RiseTime"] > 0.1)].index

        else:
            # 1 Hz low-pass butterworth filter
            rolloff = 12
            lpfreq = 2
            eda_filtered = np.concatenate(
                (np.repeat(eda_signal[0], 100), eda_signal, np.repeat(eda_signal[-1], 100)))  # zero padding
            eda_filtered[np.isnan(eda_filtered)] = np.nanmean(eda_filtered)
            b, a = signal.butter(int(rolloff / 6), lpfreq * (1 / (sampling_rate / 2)))  # low-pass filter
            eda_filtered = signal.filtfilt(b, a, eda_filtered)  # apply filter
            eda_filtered = eda_filtered[100:-100]

            # Get local minima and maxima
            local_maxima = signal.argrelextrema(eda_filtered, np.greater)[0]
            peak_values = list(eda_filtered[local_maxima])
            peak_times = list(local_maxima)

            local_minima = signal.argrelextrema(eda_filtered, np.less)[0]
            onset_values = list(eda_filtered[local_minima])
            onset_times = list(local_minima)

            scr_onsets = []
            scr_peaks = []
            scr_amplitudes = []
            scr_risetimes = []
            for onset_idx, onset in enumerate(onset_times):
                # onset_idx = 0
                # onset = onset_times[onset_idx]
                subsequent_peak_times = [peak_time for peak_time in peak_times if (onset-peak_time) < 0]
                if len(subsequent_peak_times) > 0:
                    peak_idx = list(peak_times).index(min(subsequent_peak_times, key=lambda x: abs(x-onset)))
                    rise_time = (peak_times[peak_idx] - onset) / raw.info["sfreq"]
                    amplitude = peak_values[peak_idx] - onset_values[onset_idx]
                    if (rise_time > 0.1) & (rise_time < 10) & (amplitude >= 0.05):
                        scr_onsets.append(onset)
                        scr_peaks.append(peak_times[peak_idx])
                        scr_amplitudes.append(amplitude)
                        scr_risetimes.append(rise_time)

        if len(bad_events) > 0:
            # Set bad annotations for the omitted periods
            duration_prevs = [0]
            for idx_row, row in bad_events.iterrows():
                # idx_row = 2
                # row = bad_events.iloc[idx_row, :]
                bad_events.loc[idx_row, "onset_corr"] = bad_events.loc[idx_row, "onset"] - np.sum(duration_prevs)
                duration_prevs.append(bad_events.loc[idx_row, "duration"])
            bad_annotations = mne.Annotations(bad_events["onset_corr"], [0] * len(bad_events), ['CUT_'] * len(bad_events))
            bad_annot_samples = [int(x) for x in list(bad_events["onset_corr"] * raw.info["sfreq"])]

            # Correct SCRs for the omitted periods
            scr_peaks_corr = [scr_peak for scr_peak in scr_peaks if (
                    np.min([abs(scr_peak - bad_sample) for bad_sample in bad_annot_samples]) > raw.info["sfreq"]/2)]
            scr_onsets_corr = [scr_onset for scr_onset in scr_onsets if (
                    np.min([abs(scr_onset - bad_sample) for bad_sample in bad_annot_samples]) > raw.info["sfreq"]/2)]
        else:
            scr_peaks_corr = scr_peaks
            scr_onsets_corr = scr_onsets

        scr_onsets_corr = [scr_onset for scr_onset in scr_onsets_corr if (
                np.min([abs(scr_onset - peak_sample) for peak_sample in scr_peaks_corr]) < (2 * raw.info["sfreq"]))]
        scr_peaks_corr = [scr_peak for scr_peak in scr_peaks_corr if (
                np.min([abs(scr_peak - onset_sample) for onset_sample in scr_onsets_corr]) < (2 * raw.info["sfreq"]))]

        scr_events = pd.DataFrame()
        scr_events['Samples'] = list(scr_onsets_corr) + list(scr_peaks_corr)
        scr_events['MNE'] = [0] * (len(scr_onsets_corr) + len(scr_peaks_corr))
        scr_events['Condition'] = [6] * len(scr_onsets_corr) + [7] * len(scr_peaks_corr)
        scr_events = scr_events.to_numpy()
        scr_annot = mne.annotations_from_events(events=scr_events, event_desc=event_dict_rev, sfreq=raw.info['sfreq'])
        if len(bad_events) > 0:
            scr_annot = scr_annot.__add__(bad_annotations)

        # Create new MNE raw file without bad epochs and with adapted rpeaks
        info = mne.create_info(ch_names=["EDA"], sfreq=raw.info['sfreq'], ch_types=['bio'])
        raw = mne.io.RawArray(np.reshape(eda_filtered, (1, len(eda_filtered))), info)
        raw.set_annotations(scr_annot)

        # Control plot
        # raw.plot(duration=20, scalings='auto', block=True, title=f"Check SCR: VP {vp}, Phase {phase})")

        index_peaks = [scr_peaks.index(peak) for peak in scr_peaks_corr]

        signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_filtered})
        peak_signals = {}
        peak_signals["SCR_Onsets"] = _signal_from_indices(scr_onsets_corr, len(eda_signal), 1)
        peak_signals["SCR_Onsets"] = peak_signals["SCR_Onsets"].astype("int64")  # indexing of feature using 1 and 0
        peak_signals["SCR_Peaks"] = _signal_from_indices(scr_peaks_corr, len(eda_signal), 1)
        peak_signals["SCR_Peaks"] = peak_signals["SCR_Peaks"].astype("int64")  # indexing of feature using 1 and 0
        peak_indices, values = _signal_sanitize_indices(scr_peaks_corr, np.array(scr_amplitudes)[index_peaks])  # Sanitize indices and values
        peak_signals["SCR_Amplitude"] = _signal_from_indices(peak_indices, len(eda_signal), values)  # Append peak values to signal
        peak_indices, values = _signal_sanitize_indices(scr_peaks_corr, np.array(scr_risetimes)[index_peaks])  # Sanitize indices and values
        peak_signals["SCR_RiseTime"] = _signal_from_indices(peak_indices, len(eda_signal), values)  # Append peak values to signal
        peak_signals = pd.DataFrame(peak_signals)

        signals = pd.concat([signals, peak_signals], axis=1)

        info = {}
        info["SCR_Onsets"] = np.array(scr_onsets_corr)
        info["SCR_Peaks"] = np.array(scr_peaks_corr)
        info["SCR_Amplitude"] = np.array(scr_amplitudes)[index_peaks]
        info["SCR_RiseTime"] = np.array(scr_risetimes)[index_peaks]
        info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

        return signals, info

    except:
        return pd.DataFrame(), np.array([])


def rolling_windows(eda_signal, window):
    # window = 100
    if window > eda_signal.shape[0]:
        raise ValueError(f'Specified `window` length of {window} exceeds length of `signal`, {eda_signal.shape[0]}.')
    if eda_signal.ndim == 1:
        eda_signal = eda_signal.reshape(-1, 1)
    shape = (eda_signal.shape[0] - window + 1, window) + eda_signal.shape[1:]
    strides = (eda_signal.strides[0],) + eda_signal.strides
    windows = np.squeeze(np.lib.stride_tricks.as_strided(eda_signal, shape=shape, strides=strides))
    # In cases where window == len(a), we actually want to "unsqueeze" to 2d.
    #     I.e., we still want a "windowed" structure with 1 window.
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def my_fft(s):
    return np.abs(scipy.fft.fft(s))[:int(len(s)/2)]


def sign_change(s):
    return np.sum([np.sum(np.diff(s) > 0), np.sum(np.diff(s) < 0)])


def merge_intervals(intervals):
    # Sort the array on the basis of start values of intervals.
    intervals.sort()
    stack = []
    # insert first interval into stack
    stack.append(intervals[0])
    for i in intervals[1:]:
        # Check for overlapping interval,
        # if interval overlap
        if stack[-1][0] <= i[0] <= stack[-1][-1]:
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)
    return pd.DataFrame(stack)


def _signal_from_indices(indices, desired_length=None, value=1):
    """**Generates array of 0 and given values at given indices**

    Used in *_findpeaks to transform vectors of peak indices to signal.

    """
    signal = pd.Series(np.zeros(desired_length, dtype=float))

    if isinstance(indices, list) and (not indices):  # skip empty lists
        return signal
    if isinstance(indices, np.ndarray) and (indices.size == 0):  # skip empty arrays
        return signal

    # Force indices as int
    if isinstance(indices[0], float):
        indices = indices[~np.isnan(indices)].astype(int)

    # Appending single value
    if isinstance(value, (int, float)):
        signal[indices] = value
    # Appending multiple values
    elif isinstance(value, (np.ndarray, list)):
        for index, val in zip(indices, value):
            signal.iloc[index] = val
    else:
        if len(value) != len(indices):
            raise ValueError(
                "NeuroKit error: _signal_from_indices(): The number of values "
                "is different from the number of indices."
            )
        signal[indices] = value

    return signal


def _signal_sanitize_indices(indices, values):
    # Check if nan in indices
    if np.sum(np.isnan(indices)) > 0:
        to_drop = np.argwhere(np.isnan(indices))[0]
        for i in to_drop:
            indices = np.delete(indices, i)
            values = np.delete(values, i)

    return indices, values


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

    # Get EDA data
    try:
        file_path = os.path.join(dir_path, 'Data', 'VP_' + vp, 'EDA')
        folder = [item for item in os.listdir(file_path)][0]
        file_path = os.path.join(file_path, folder)

        # Get start time of recording
        start_time = ET.parse(os.path.join(file_path, 'unisens.xml')).getroot().attrib['timestampStart']
        start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%f')
        duration = float(ET.parse(os.path.join(file_path, 'unisens.xml')).getroot().attrib['duration'])
        sampling_rate = 32
        lsb = 0.0030518508225365697

        df_eda = pd.read_csv(os.path.join(file_path, 'eda.csv'), header=None, names=['eda [unscaled]'])
        df_eda['eda [µS]'] = (df_eda['eda [unscaled]']) * lsb
    except:
        print("no eda file")
        continue

    # Add timestamps based on sampling rate and duration
    timestamps = pd.timedelta_range(start=0, end=timedelta(seconds=duration), periods=len(df_eda)) / pd.to_timedelta(1, unit='ms')
    sr_check, fs = utils.get_sampling_rate(timestamps)
    timeseries = pd.date_range(start=start_time, end=start_time + timedelta(seconds=duration), periods=len(df_eda))
    # df_eda['time'] = timeseries
    df_eda['time'] = pd.to_datetime(timeseries, unit="ns", utc=True)
    df_eda["timestamp"] = df_eda["time"].apply(lambda t: t.replace(tzinfo=None))
    df_eda = df_eda.drop(columns=["time"])

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
            marker = pd.read_csv(os.path.join(file_path_physio, 'marker.csv'), sep=';', decimal='.', names=['sample'],
                                 header=None, usecols=[0])
            marker_samplingrate = 64
            marker_timepoints = [start_time + timedelta(seconds=marker["sample"][i] / marker_samplingrate) for i in
                                 np.arange(0, len(marker["sample"]))]
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
                  'onset': 6,
                  'peak': 7,
                  '_BAD': 99}
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
    df_eda = pd.merge_asof(df_eda, df_event[["timestamp", "name"]], on="timestamp", direction="backward").reset_index(drop=True)
    tolerance = timedelta(milliseconds=(1 / sampling_rate) * 1000)
    df_eda = pd.merge_asof(df_eda, df_event[["timestamp", "name", "event"]], by="name", on="timestamp",
                           direction="nearest", tolerance=tolerance)

    # Iterate through experimental phases and check EDA data
    for idx_row, row in df_event.iterrows():
        # idx_row = 5
        # row = df_event.iloc[idx_row]
        phase = row['name']
        print(f"Phase: {phase}")

        # Get start and end point of phase
        start_phase = row['timestamp']
        end_phase = row['timestamp'] + pd.to_timedelta(row['duration'], unit="S")

        # Cut EDA dataset
        df_eda_subset = df_eda.loc[(df_eda["timestamp"] >= start_phase) & (df_eda["timestamp"] < end_phase + timedelta(seconds=1))]
        df_eda_subset = df_eda_subset.loc[df_eda_subset['name'] == phase].reset_index(drop=True)

        # Create MNE events file
        mne_events = pd.DataFrame()
        mne_events['Samples'] = list(df_eda_subset.dropna(subset="event").index)
        mne_events['MNE'] = [0] * len(df_eda_subset.dropna(subset="event"))
        mne_events['Condition'] = df_eda_subset.dropna(subset="event")['event'].to_list()

        # Create MNE info file and MNE raw file
        info = mne.create_info(ch_names=["EDA"], sfreq=sampling_rate, ch_types=['bio'])
        # raw = mne.io.RawArray(np.reshape(np.array(df_eda[["ecg [mV]", "event"]]), (2, len(df_eda))), info)
        data = np.reshape(df_eda_subset['eda [µS]'].to_numpy(), (1, len(df_eda_subset['eda [µS]'])))
        raw = mne.io.RawArray(data, info)

        duration_pre = raw.times.max()

        # Use customized function to analyse EDA
        try:
            signals, scr_detection = eda_custom_process(raw, vp=vp, phase=phase, sampling_rate=raw.info['sfreq'],
                                                        pipeline=None, correction=["fft"], amplitude_min=0.02)
        except:
            print("Problem with EDA processing")
            signals = []

        if len(signals) == 0:
            df_eda_temp = pd.DataFrame({'VP': [int(vp)],
                                        'Phase': [phase],
                                        'event_id': [idx_row],
                                        'SCL (Mean)': [np.nan],
                                        'SCL (Std)': [np.nan],
                                        'SCR (absolute Number)': [np.nan],
                                        'SCR (Peaks per Minute)': [np.nan],
                                        'Duration': [0],
                                        'Proportion Usable Data': [0]})
            df_eda_temp.to_csv(os.path.join(dir_path, 'Data', 'eda.csv'), decimal='.', sep=';', index=False, mode='a',
                               header=not (os.path.exists(os.path.join(dir_path, 'Data', 'eda.csv'))))
            plt.close()
            continue

        duration_post = len(signals["EDA_Clean"]) / sampling_rate

        # Save as dataframe (SCL and SCR)
        df_eda_temp = pd.DataFrame({'VP': [int(vp)],
                                    'Phase': [phase],
                                    'event_id': [idx_row],
                                    'SCL (Mean)': [np.mean(signals['EDA_Clean'])],
                                    'SCL (Std)': [np.std(signals['EDA_Clean'])],
                                    'SCR (absolute Number)': [np.sum(signals['SCR_Peaks'])],
                                    'SCR (Peaks per Minute)': [np.sum(signals['SCR_Peaks']) / duration_post],
                                    'Duration': [duration_post],
                                    'Proportion Usable Data': [duration_pre / duration_post]})
        df_eda_temp.to_csv(os.path.join(dir_path, 'Data', 'eda.csv'), decimal='.', sep=';', index=False, mode='a',
                           header=not (os.path.exists(os.path.join(dir_path, 'Data', 'eda.csv'))))
        plt.close()

        if "Interaction" in phase:
            df_eda_subset_save = signals.copy()
            df_eda_subset_save["timestamp"] = df_eda_subset["timestamp"]

            start_eda = df_eda_subset_save.loc[0, "EDA_Clean"]
            df_eda_subset_save["EDA"] = df_eda_subset_save["EDA_Clean"] - start_eda

            start = df_eda_subset_save.loc[0, "timestamp"]
            df_eda_subset_save["time"] = pd.to_timedelta(df_eda_subset_save["timestamp"] - start)
            df_eda_subset_save = df_eda_subset_save.set_index("time")
            df_eda_subset_save = df_eda_subset_save.resample("0.1S").mean()
            df_eda_subset_save = df_eda_subset_save.reset_index()
            df_eda_subset_save["time"] = df_eda_subset_save["time"].dt.total_seconds()
            df_eda_subset_save["VP"] = int(vp)
            df_eda_subset_save["event"] = phase
            df_eda_subset_save = df_eda_subset_save[["VP", "event", "time", "EDA"]]
            df_eda_subset_save.to_csv(os.path.join(dir_path, 'Data', 'eda_interaction.csv'), decimal='.', sep=';', index=False,
                                 mode='a', header=not (os.path.exists(os.path.join(dir_path, 'Data', 'eda_interaction.csv'))))

# Add Subject Data
df_eda = pd.read_csv(os.path.join(dir_path, 'Data', 'eda.csv'), decimal='.', sep=';')
df_eda = df_eda.iloc[:, 0:9]
df_eda = df_eda.dropna(subset=['SCL (Mean)'])

# Get conditions
dfs_eda = []
for vp in vps:
    # vp = vps[3]
    vp = f"0{vp}" if vp < 10 else f"{vp}"
    print(f"VP: {vp}")

    df_eda_vp = df_eda.loc[df_eda["VP"] == int(vp)]

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

    df_eda_vp["Condition"] = ""
    for idx_row, row in df_roles.iterrows():
        # idx_row = 0
        # row = df_roles.iloc[idx_row, :]
        room = row["Rooms"]
        role = row["Role"]
        df_eda_vp.loc[df_eda_vp["Phase"].str.contains(room), "Condition"] = role
    dfs_eda.append(df_eda_vp)

df_eda = pd.concat(dfs_eda)

df_scores = pd.read_csv(os.path.join(dir_path, 'Data', 'scores_summary.csv'), decimal=',', sep=';')
df_eda = df_eda.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                 'SSQ', 'SSQ-N', 'SSQ-O', 'SSQ-D', 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS',
                                 'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                 'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
df_eda = df_eda.drop(columns=['ID'])
df_eda.to_csv(os.path.join(dir_path, 'Data', 'eda.csv'), decimal='.', sep=';', index=False)
