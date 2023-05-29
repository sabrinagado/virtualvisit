import numpy as np
from scipy import signal
from numpy import inf


def filtering(data, freq, fs, f_type, filter_design, order=5, axis=-1):
    """ This functions filters a 2-D numpy array

    Parameters
    ----------
    data : 2-D numpy array (channels, samples)
    freq: cut-off freq - list of float - can be a list with one or two elements, depends if low- high- or bandpass is choosen
    fs: sampling rate - float
    order: fitler order - float
    f_type: type of filter - string with 'lowpass', 'highpass' or 'bandpass'
    filter_design: wheter to perform a IIR or FIR fitler - string with 'IIR' (uses a butterworth filter) or 'FIR' (uses the standard hamming window)
    order: filer order - int - for IIR typical order is 5 - for FIR the order will be determined by the cut_off frequencies
    axis: The axis of data to which the filter is applied. the default is over the last axis (i.e. axis=-1).

    Example
    -------
    freq = [1, 35]
    fs = 128
    f_type = 'bandpass'
    filter_design = 'FIR'

    filtered_data = processing.filtering(data, freq, fs, f_type, filter_design)

    """

    if filter_design == 'IIR':

        if f_type == 'notch':
            nyq = 0.5 * fs  # get the Nyquist frequeny - half the sampling frequency
            low = freq[0] / nyq
            high = freq[1] / nyq
            normal_cutoff = [low, high]

            # get the filter coefficients
            b, a = signal.iirfilter(order, normal_cutoff, btype='bandstop', analog=False)

            # use a notch filter with iir response bandstop
            # data_filtered = signal.lfilter(b, a, data, axis=axis)
            data_filtered = signal.filtfilt(b, a, data, padtype='constant', padlen=int(round(order * 1.5)),
                                            method='pad', axis=axis)

        if f_type == 'lowpass':
            nyq = 0.5 * fs  # get the Nyquist frequeny - half the sampling frequency
            normal_cutoff = freq[0] / nyq  # get the nromalized frequencies

            # get the filter coefficients
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

            # use zero-phase lag filter - digital filter forward and backward to a signal - padding the data at the
            # beginning and end, this avoids artefacts at the edges
            data_filtered = signal.filtfilt(b, a, data, padtype='constant', padlen=int(round(order * 1.5)),
                                            method='pad', axis=axis)

        if f_type == 'highpass':
            nyq = 0.5 * fs  # get the Nyquist frequeny - half the sampling frequency
            normal_cutoff = freq[0] / nyq  # get the nromalized frequencies

            # get the filter coefficients
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

            # use zero-phase lag filter - digital filter forward and backward to a signal - padding the data at the
            # beginning and end, this avoids artefacts at the edges
            data_filtered = signal.filtfilt(b, a, data, padtype='constant', padlen=int(round(order * 1.5)),
                                            method='pad', axis=axis)

        if f_type == 'bandpass':
            nyq = 0.5 * fs  # get the Nyquist frequeny - half the sampling frequency
            low = freq[0] / nyq
            high = freq[1] / nyq
            normal_cutoff = [low, high]

            # get the filter coefficients
            b, a = signal.butter(order, normal_cutoff, btype='band', analog=False)

            # use zero-phase lag filter - digital filter forward and backward to a signal - padding the data at the
            # beginning and end, this avoids artefacts at the edges
            data_filtered = signal.filtfilt(b, a, data, padtype='constant', padlen=int(round(order * 1.5)),
                                            method='pad', axis=axis)

    if filter_design == 'FIR':

        if f_type == 'lowpass':
            nyq = 0.5 * fs  # get the Nyquist frequeny - half the sampling frequency
            normal_cutoff = freq[0] / nyq  # get the nromalized frequencies

            order = np.max([3 * round(fs / freq[0]), order])
            # get the filter coefficients
            taps = signal.firwin(order + 1, normal_cutoff, window='hamming', pass_zero='lowpass')

            # use zero-phase lag filter - digital filter forward and backward to a signal - padding the data at the
            # beginning and end, this avoids artefacts at the edges
            data_filtered = signal.filtfilt(taps, 1.0, data, padtype='constant', padlen=int(round(order * 1.5)),
                                            method='pad', axis=axis)

        if f_type == 'highpass':
            nyq = 0.5 * fs  # get the Nyquist frequeny - half the sampling frequency
            normal_cutoff = freq[0] / nyq  # get the nromalized frequencies

            # overwrite the default order setting
            order = np.max([3 * round(fs / freq[0]), order])
            # get the filter coefficients
            taps = signal.firwin(order + 1, normal_cutoff, window='hamming', pass_zero='highpass')

            # use zero-phase lag filter - digital filter forward and backward to a signal - padding the data at the
            # beginning and end, this avoids artefacts at the edges
            data_filtered = signal.filtfilt(taps, 1.0, data, padtype='constant', padlen=int(round(order * 1.5)),
                                            method='pad', axis=axis)

        if f_type == 'bandpass':
            nyq = 0.5 * fs  # get the Nyquist frequeny - half the sampling frequency
            low = freq[0] / nyq
            high = freq[1] / nyq
            normal_cutoff = [low, high]  # get the nromalized frequencies

            # get the filter coefficients
            taps = signal.firwin(order + 1, normal_cutoff, window='hamming', pass_zero='bandpass')

            # use zero-phase lag filter - digital filter forward and backward to a signal - padding the data at the
            # beginning and end, this avoids artefacts at the edges
            data_filtered = signal.filtfilt(taps, 1.0, data, padtype='constant', padlen=int(round(order * 1.5)),
                                            method='pad', axis=axis)

    return data_filtered


def ampl_reject(epo, channels, amp):
    """Apply amplitude rejection

    Reject epochs which will not fulfill the min/max requirements
    of specific channels

    Parameter
    ---------
    epo: mne.epochs
        mne epoch object
    channels: List[str]
        List of channels which should be considered
    amp: List[float]
        If there is only one value, it will apply for max and min
        If there is two value, the first value is min and the second is max

    Returns
    -------
    epo: mne.epochs
        mne epoch object with dropped epochs

    Raises
    ------
    ValueError
        If amp list have less than 1 or more than 2 values

    """

    # Convert list to numpy array for better handling
    data = np.asarray(epo.get_data())

    # Contains indices of epochs which will be dropped
    drop_indices = []

    # Determine indices of channels
    ch_index = [epo.ch_names.index(ch) for ch in channels]

    # Determine drop indices
    for i in range(len(data)):
        ch_dat = data[i][ch_index]

        if len(amp) > 2 or len(amp) < 1:
            raise ValueError('amp list should have one or two values')

        if len(amp) == 1:
            for ch in ch_dat:
                if max(abs(signal.detrend(ch))) > amp[0]:
                    drop_indices.append(i)
                    break

        if len(amp) == 2:
            for ch in ch_dat:
                if min(signal.detrend(ch)) < amp[0] or max(signal.detrend(ch)) > amp[1]:
                    drop_indices.append(i)
                    break

    # If there is no drop, do nothing
    if len(drop_indices) > 0:
        epo.drop(drop_indices, reason="amplitude rejection")

    return epo