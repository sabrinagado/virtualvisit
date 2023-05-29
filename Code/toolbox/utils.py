import numpy as np


def find_nearest_idx(array: list, value: float) -> int:
    """
    Get nearest index by value

    Parameters
    ----------
    array: list like array with values to be searched
    value: float value to search in the array

    Returns
    -------
    index of the nearest value in the array

    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def create_mne_epoch_events(timestamps: list, markers: list) -> np.ndarray:
    """
    Create a (samples x [sample nr, '0', event]) numpy array for mne epochs creation

    Parameters
    ----------
    timestamps: list like 1-d array
    markers: 2d array (timestamps x event)

    Returns
    -------
    array (samples x [sample nr, '0', event])

    """
    markers = np.asarray(markers)

    idx_list = [find_nearest_idx(timestamps, float(t)) for t in markers[:, 0]]
    event_marker = np.insert(markers, 0, idx_list, axis=1)
    event_marker[:, 1] = 0
    return event_marker.astype(int)


def create_epoch_events(timestamps: list, markers: list) -> np.ndarray:
    """
    Create a (samples x [sample nr, event]) numpy array

    Parameters
    ----------
    timestamps: list like 1-d array
    markers: 2d array (timestamps x event)

    Returns
    -------
    list 1-d

    """
    markers = np.asarray(markers)
    idx_list = [find_nearest_idx(timestamps, float(t[0])) for t in markers]
    markers[:, 0] = idx_list
    return markers.astype(int)


def resample_markers(markers: np.ndarray, fs: int, in_len: float, out_len: float) -> np.ndarray:
    """

    Resample markers

    Parameters
    ----------
    markers: numpy array (samples x [sample nr, ...]) sample number has to be the first column. The other
        columns will be adopted to the new generated marker samples
    fs: sample rate
    in_len: the assumed income length of an epoch in seconds
    out_len: the outcome length of an epoch in seconds

    Returns
    -------
    New markers array (samples x [sample nr, ...])
    """

    assert in_len != 0 or out_len != 0, "in_len and out_len cannot be 0"

    out_sample_len = fs * out_len

    new_markers = []

    if in_len > out_len:  # Up sampling
        generated_samples = int(in_len / out_len)
        for m in markers:
            for i in range(generated_samples):
                row = [m[0] + (i * out_sample_len)] + m[1:].tolist()
                new_markers.append(row)
    else:  # Down sampling
        take_samples = int(out_len / in_len)
        for i in range(len(markers)):
            if i % take_samples == 0:  # every take_samples
                new_markers.append(markers[i])

    return np.asarray(new_markers)


def create_epochs(data: list, events: list, fs: int, t_min: float, t_max: float, baseline: list = None,
                  channels='all') -> np.ndarray:
    """
    Creates an epoch numpy array (epoch x sample x channel)

    Parameters
    ----------
    data: time series array (sample x channel)
    events: list like array with sample numbers corresponding to events
    fs: sample rate (Hz)
    t_min: start time before event in seconds
    t_max: end time after event in seconds
    baseline: apply baseline correction with a period between (a, b) in seconds
        If none, do not apply baseline correction.
        If tuple (a, b), calculate the mean of interval between a and b and apply correction over epochs
        If (a, None), calculate the mean of interval from a to end of data
        If (None, b), calculate the mean of interval from begin to b of data
        Note the interval notation [a, b) where the index at a is included and b is not included. If you want
        to include index at b, do this (a, b + 1)
    channels: pick channels (0, 1, ...) where the baseline correction should be applied.
        Default is 'all' which takes all channels.
        Insert int value for one channel
        Insert tuple or list of indexes for choosing more channels

    Returns
    -------
    epochs array (epoch x sample x channel)

    """
    epochs = []
    d = np.asarray(data)

    for e in events:
        start = int(e) + t_min * fs  # Start sample index
        end = int(e) + t_max * fs  # End sample index

        epo = d[int(start):int(end)]  # Slice subset from data from start to end

        # Check if there is enough data in the epo, since there might be not enough data in the end
        if epo.shape[0] < int(end) - int(start):
            continue

        # Apply baseline correction if value is not None
        if baseline is not None:
            b_start = int(e) + baseline[0] * fs if baseline[0] is not None else start
            b_end = int(e) + baseline[1] * fs if baseline[1] is not None else end

            base = d[int(b_start):int(b_end)].mean(axis=0)  # Calculate mean

            epo = epo - base  # Apply baseline correction

        epochs.append(epo)

    return np.asarray(epochs)


def get_sampling_rate(timestamp):
    """

    Estimates the sampling rate from the data timestamp
    Parameters
    ----------
    timestamp: 1d numpy array with timestamps

    Returns
    -------
    sampling rate
    """
    fs = 0
    for i in range(len(timestamp) - 1):
        fs += timestamp[i + 1] - timestamp[i]

    # fs: sampling intervall (every X ms)
    fs = fs / (len(timestamp) - 1)

    # sampling rate in Hz (Samples per Second)
    if fs > 0:
        sr = round(1000 / fs, 2)
    else:
        sr = 0

    return sr, fs


def remove_empty_tuples(tuples):
    dropped = []
    reason = []
    for i, t in enumerate(tuples):
        if t != ():
            dropped.append(i)
            reason.append(t[0])
    return dropped, reason
