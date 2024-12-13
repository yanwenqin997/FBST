import numpy as np
from scipy import signal
def fft_HR(y, fs):
    # input y:signals; fs：Sampling frequency
    T = 1 / fs  # Sampling interval
    N = len(y)
    t = np.arange(N) * T
    df = fs / N;
    f = df * np.arange(int(N / 2))
    half_x = t[range(int(N / 2))]
    fft_y = np.fft.fft(y)
    abs_y = np.abs(fft_y)
    angle_y = 180 * np.angle(fft_y) / np.pi
    gui_y = abs_y / N
    gui_half_y = gui_y[range(int(N / 2))]
    peak_id, peak_property = signal.find_peaks(gui_half_y, height=np.max(gui_half_y) / 1.00001, distance=fs / 2 - 1)
    peak_freq = f[peak_id]
    peak_height = peak_property['peak_heights']
    hr = peak_freq * 60
    return hr

def filt(x,fs,low_cutoff,hight_cutoff):
    """
    Filter the signal
    x: The sequence to be processed
    fs：Sampling frequency
    low_cutoff: Frequency cutoff
    high_cutoff:Frequency cut-off upper limit
    return：The filtered signal
    """
    x=(x-np.mean(x))/np.std(x)
    Wn_low = 2*low_cutoff/fs
    Wn_high = 2*hight_cutoff/fs
    m,n = signal.butter(3,[Wn_low,Wn_high],'bandpass')
    x_filter=signal.filtfilt(m,n,x)
    return x_filter


def fft_snr(y, fs):
    # input y:signals;fs：Sampling frequency
    T = 1 / fs
    y = (y - np.mean(y, 0)) / np.std(y, 0)
    N = len(y)
    t = np.arange(N) * T

    df = fs / N;
    f = df * np.arange(int(N / 2))

    half_x = t[range(int(N / 2))]

    fft_y = np.fft.fft(y)

    abs_y = np.abs(fft_y)
    angle_y = 180 * np.angle(fft_y) / np.pi
    gui_y = abs_y / N
    gui_half_y = gui_y[range(int(N / 2))]
    f_min = np.min(np.argwhere(f > 0.1))
    f_max = np.max(np.argwhere(f < 0.7)) + 1
    signal_power = np.sum(np.square(gui_half_y[f_min:f_max]))
    noise_power = np.sum(np.square(gui_half_y[0:f_min])) + np.sum(np.square(gui_half_y[f_max:]))
    SNR = signal_power / noise_power
    return SNR
