# =============================================================================
# EDA
# sensor: movisens
# study: Virtual Visit
# =============================================================================
import os
import sys
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
from tqdm import tqdm

from Code.toolbox import utils

from Code import preproc_scores, preproc_ratings, preproc_behavior

import mne

plt.ion()
matplotlib.use('QtAgg')

args = sys.argv
wave = int(args[1])
filepath = args[2]


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


def eda_custom_process(raw, event_dict_rev, vp, phase, sampling_rate=32, pipeline=None, correction=["manual", "sd", "fft"], amplitude_min=0.02):
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
        if raw.times.max() >= 5:
            windows, frequencies, powers = [2, 5], [10, 10], [15, 100]
        else:
            windows, frequencies, powers = [2], [10], [15]
        for window_seconds, frequency_threshold, power_threshold in zip(windows, frequencies, powers):
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
        eda_signal = raw.get_data(picks=['EDA'], reject_by_annotation="omit", verbose=False).flatten()
        if len(eda_signal) == 0:
            raise Exception('bad signal quality')

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
            # low-pass butterworth filter
            rolloff = 12
            lpfreq = 2
            eda_filtered = np.concatenate((np.repeat(eda_signal[0], 100), eda_signal, np.repeat(eda_signal[-1], 100)))  # zero padding
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
                    if (rise_time > 0.1) & (rise_time < 10) & (amplitude >= amplitude_min):
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

        scr_onsets_corr = [scr_onset for scr_onset in scr_onsets_corr if (np.min([abs(scr_onset - peak_sample) for peak_sample in scr_peaks_corr]) < (2 * raw.info["sfreq"]))]
        if len(scr_onsets_corr) > 0:
            scr_peaks_corr = [scr_peak for scr_peak in scr_peaks_corr if (np.min([abs(scr_peak - onset_sample) for onset_sample in scr_onsets_corr]) < (2 * raw.info["sfreq"]))]

        if len(scr_onsets_corr) > 0:
            scr_events = pd.DataFrame()
            scr_events['Samples'] = list(scr_onsets_corr) + list(scr_peaks_corr)
            scr_events['MNE'] = [0] * (len(scr_onsets_corr) + len(scr_peaks_corr))
            scr_events['Condition'] = [100] * len(scr_onsets_corr) + [101] * len(scr_peaks_corr)

            scr_events = scr_events.to_numpy()
            scr_annot = mne.annotations_from_events(events=scr_events, event_desc=event_dict_rev, sfreq=raw.info['sfreq'])
            if len(bad_events) > 0:
                scr_annot = scr_annot.__add__(bad_annotations)

            # Create new MNE raw file without bad epochs and with onsets and peaks
            info = mne.create_info(ch_names=["EDA"], sfreq=raw.info['sfreq'], ch_types=['bio'])
            raw = mne.io.RawArray(np.reshape(eda_filtered, (1, len(eda_filtered))), info, verbose=False)
            raw.set_annotations(scr_annot)

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

            info = {}
            info["SCR_Onsets"] = np.array(scr_onsets_corr)
            info["SCR_Peaks"] = np.array(scr_peaks_corr)
            info["SCR_Amplitude"] = np.array(scr_amplitudes)[index_peaks]
            info["SCR_RiseTime"] = np.array(scr_risetimes)[index_peaks]
            info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info
        else:
            info = mne.create_info(ch_names=["EDA"], sfreq=raw.info['sfreq'], ch_types=['bio'])
            raw = mne.io.RawArray(np.reshape(eda_filtered, (1, len(eda_filtered))), info, verbose=False)
            signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_filtered})
            peak_signals = pd.DataFrame()
            info = {}
            info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

        # Control plot
        # raw.plot(duration=20, scalings='auto', block=True, title=f"Check SCR: VP {vp}, Phase {phase})")

        signals = pd.concat([signals, peak_signals], axis=1)
        return signals, info
    except:
        return pd.DataFrame(columns=["EDA_Raw", "EDA_Clean"]), np.array([])


def get_eda(vps, filepath, wave, df_scores):
    df_scl = pd.DataFrame()
    df_scr_interaction = pd.DataFrame()
    for vp in tqdm(vps):
        # vp = vps[0]
        vp = f"0{vp}" if vp < 10 else f"{vp}"
        # print(f"VP: {vp}")

        df_scl_vp = pd.DataFrame()
        df_scr_interaction_vp = pd.DataFrame(columns=["VP", "event", "time", "EDA"])

        # Get EDA data
        try:
            file_path = os.path.join(filepath, 'VP_' + vp, 'EDA')
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
        event_dict['onset'] = 100
        event_dict['peak'] = 101
        event_dict_rev = {v: k for k, v in event_dict.items()}

        # Add "event" columns (for MNE)
        df_events_vp["name"] = df_events_vp["event"]
        df_events_vp["event"] = codes

        # Merge "name" and "event"-column to df_ecg
        df_eda = pd.merge_asof(df_eda, df_events_vp[["timestamp", "name"]], on="timestamp", direction="backward").reset_index(drop=True)
        tolerance = timedelta(milliseconds=(1 / sampling_rate) * 1000)
        df_eda = pd.merge_asof(df_eda, df_events_vp[["timestamp", "name", "event"]], by="name", on="timestamp", direction="nearest", tolerance=tolerance)

        # Iterate through experimental phases and check EDA data
        for idx_row, row in df_events_vp.iterrows():
            # idx_row = 2
            # row = df_events_vp.iloc[idx_row]
            phase = row['name']
            # print(f"Phase: {phase}")

            # Get start and end point of phase
            start_phase = row['timestamp']
            end_phase = row['timestamp'] + pd.to_timedelta(row['duration'], unit="S")

            # Cut EDA dataset
            df_eda_subset = df_eda.loc[(df_eda["timestamp"] >= start_phase) & (df_eda["timestamp"] < end_phase + timedelta(seconds=1))]
            df_eda_subset["name"] = phase
            # df_eda_subset = df_eda_subset.loc[df_eda_subset['name'] == phase].reset_index(drop=True)

            # Create MNE events file
            mne_events = pd.DataFrame()
            mne_events['Samples'] = list(df_eda_subset.dropna(subset="event").index)
            mne_events['MNE'] = [0] * len(df_eda_subset.dropna(subset="event"))
            mne_events['Condition'] = df_eda_subset.dropna(subset="event")['event'].to_list()

            # Create MNE info file and MNE raw file
            info = mne.create_info(ch_names=["EDA"], sfreq=sampling_rate, ch_types=['bio'])
            # raw = mne.io.RawArray(np.reshape(np.array(df_eda[["ecg [mV]", "event"]]), (2, len(df_eda))), info)
            data = np.reshape(df_eda_subset['eda [µS]'].to_numpy(), (1, len(df_eda_subset['eda [µS]'])))
            raw = mne.io.RawArray(data, info, verbose=False)

            duration_pre = raw.times.max()

            # Use customized function to analyse EDA
            try:
                signals, scr_detection = eda_custom_process(raw, event_dict_rev, vp=vp, phase=phase, sampling_rate=raw.info['sfreq'],
                                                            pipeline=None, correction=["fft"], amplitude_min=0.02)
            except:
                signals = pd.DataFrame(columns=["EDA_Raw", "EDA_Clean"])
                # print("Problem with EDA processing")

            if len(signals.columns) < 3:
                df_eda_temp = pd.DataFrame({'VP': [int(vp)],
                                            'Phase': [phase],
                                            'event_id': [idx_row],
                                            'SCL (Mean)': [np.mean(signals['EDA_Clean'])],
                                            'SCL (Std)': [np.std(signals['EDA_Clean'])],
                                            'SCR (absolute Number)': [np.nan],
                                            'SCR (Peaks per Minute)': [np.nan],
                                            'Duration': [0],
                                            'Proportion Usable Data': [0]})
                df_scl_vp = pd.concat([df_scl_vp, df_eda_temp])

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
                                        'Proportion Usable Data': [round(np.max([duration_post / duration_pre, 1]), 2)]})
            df_scl_vp = pd.concat([df_scl_vp, df_eda_temp])
            plt.close()

            if ("Interaction" in phase) or ("Click" in phase) or (("Visible" in phase) and not ("Actor" in phase)):
                if (wave == 2) & ((duration_post < 5.9) | (df_scr_interaction_vp["event"].str.contains(phase).any())):
                    continue

                df_eda_subset_save = signals.copy()
                df_eda_subset_save["timestamp"] = df_eda_subset["timestamp"].reset_index(drop=True)

                start_eda = df_eda_subset_save.loc[0, "EDA_Clean"]
                df_eda_subset_save["EDA"] = df_eda_subset_save["EDA_Clean"] - start_eda

                start = df_eda_subset_save.loc[0, "timestamp"]
                df_eda_subset_save["time"] = pd.to_timedelta(df_eda_subset_save["timestamp"] - start)
                df_eda_subset_save = df_eda_subset_save.set_index("time")
                df_eda_subset_save = df_eda_subset_save.resample("0.1S").mean(numeric_only=True)
                df_eda_subset_save = df_eda_subset_save.reset_index()
                df_eda_subset_save["time"] = df_eda_subset_save["time"].dt.total_seconds()
                df_eda_subset_save["VP"] = int(vp)
                df_eda_subset_save["event"] = phase
                df_eda_subset_save = df_eda_subset_save[["VP", "event", "time", "EDA"]]
                df_scr_interaction_vp = pd.concat([df_scr_interaction_vp, df_eda_subset_save])

        # Add Conditions
        for idx_row, row in df_roles.iterrows():
            # idx_row = 0
            # row = df_roles.iloc[idx_row, :]
            room = row["Rooms"]
            role = row["Role"]
            character = row["Character"]
            df_scl_vp["Phase"] = df_scl_vp["Phase"].str.replace(character, role.capitalize())

            if wave == 1:
                df_scl_vp.loc[df_scl_vp["Phase"].str.contains(room), "Condition"] = role
            df_scl_vp.loc[df_scl_vp["Phase"].str.contains(role.capitalize()), "Condition"] = role
            if len(df_scr_interaction_vp) > 0:
                df_scr_interaction_vp["event"] = df_scr_interaction_vp["event"].str.replace(character, role.capitalize())

        # Add Participant Data
        df_scl_vp = df_scl_vp.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                     'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                     'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                     'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                     'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
        df_scl_vp = df_scl_vp.drop(columns=['ID'])

        if len(df_scr_interaction_vp) > 0:
            df_scr_interaction_vp = df_scr_interaction_vp.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                                 'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                                 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                                 'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                                 'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
            df_scr_interaction_vp = df_scr_interaction_vp.drop(columns=['ID'])

        df_scl = pd.concat([df_scl, df_scl_vp])
        df_scr_interaction = pd.concat([df_scr_interaction, df_scr_interaction_vp])

        df_scl.to_csv(os.path.join(filepath, 'eda.csv'), decimal='.', sep=';', index=False)
        df_scr_interaction.to_csv(os.path.join(filepath, 'eda_interaction.csv'), decimal='.', sep=';', index=False)
    return df_scl, df_scr_interaction


if __name__ == '__main__':
    # wave = 2
    # dir_path = os.getcwd()
    # filepath = os.path.join(dir_path, f'Data-Wave{wave}')
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

    try:
        df_eda = pd.read_csv(os.path.join(filepath, 'eda.csv'), decimal='.', sep=';')
        finished_subjects = list(df_eda["VP"].unique())
        print("existing eda.csv found, only adding new VPs")
    except:
        finished_subjects = []
        # print("no existing eda.csv found")

    vps = [vp for vp in vps if not vp in problematic_subjects + finished_subjects]

    df_scl, df_scr_interaction = get_eda(vps, filepath, wave, df_scores)
