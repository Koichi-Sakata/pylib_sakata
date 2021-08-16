# Copyright (c) 2021 Koichi Sakata

# fft_axis, fft = fft(data, dt)
# fft_axis, fft_mean = fft_ave(data, dt, windivnum=4, overlap=0.5)
# freqresp, coh = tfestimate(x, y, freq, dt, windivnum=4, overlap=0.5)


import numpy as np
from matplotlib import mlab
from scipy import signal, interpolate


def fft(data, dt):
    framesize = len(data)
    fft = np.abs(np.fft.fft(data)/(framesize/2))
    fft_axis = np.linspace(0, 1/dt, framesize)
    return fft_axis, fft


def fft_ave(data, dt, windivnum=4, overlap=0.5):
    framesize = int(_floorpow2(len(data))/windivnum)

    # Overlaping
    Tlen = len(data) * dt                       # Time of data length
    Tfc = framesize * dt                        # Time of frame length
    x_ol = framesize * (1 - overlap)            # オーバーラップ時のフレームずらし幅
    N_ave = int( (Tlen - (Tfc * overlap)) / (Tfc * (1-overlap)) ) # data number for average
    data_array = []
    for i in range(N_ave):
        ps = int(x_ol * i)                      # 切り出し位置をループ毎に更新
        data_array.append(data[ps:ps+framesize:1])  # 切り出し位置psからフレームサイズ分抽出して配列に追加

    # Hanning Window
    han = signal.hanning(framesize)
    acf = 1 / (sum(han) / framesize)            # Amplitude Correction Factor
    for i in range(N_ave):
        data_array[i] = data_array[i] * han

    # FFT average
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf*np.abs(np.fft.fft(data_array[i])/(framesize/2)))
    fft_axis = np.linspace(0, 1/dt, framesize)
    fft_mean = np.sqrt(np.mean(np.array(fft_array)**2, axis=0))
    return fft_axis, fft_mean


def tfestimate(x, y, freq, dt, windivnum=4, overlap=0.5):
    x = signal.detrend(x, type='constant')
    y = signal.detrend(y, type='constant')
    NFFT = int(_floorpow2(len(x))/windivnum)
    Pxy, freq_tmp = mlab.csd(x, y, NFFT=NFFT, Fs=int(1/dt), window=mlab.window_hanning, noverlap=int(NFFT*overlap))
    Pxx, freq_tmp = mlab.psd(x, NFFT=NFFT, Fs=int(1/dt), window=mlab.window_hanning, noverlap=int(NFFT*overlap))
    coh_tmp, freq_tmp = mlab.cohere(x, y, NFFT=NFFT, Fs=int(1/dt), window=mlab.window_hanning, noverlap=int(NFFT*overlap))
    frd = Pxy/Pxx
    mag_tmp = np.abs(frd)
    phase_tmp = np.angle(frd)    
    real_tmp = mag_tmp*np.cos(phase_tmp)
    imag_tmp = mag_tmp*np.sin(phase_tmp)
    
    # Interpolation
    f_real = interpolate.interp1d(freq_tmp, real_tmp)
    real = f_real(freq)
    f_imag = interpolate.interp1d(freq_tmp, imag_tmp)
    imag = f_imag(freq)
    freqresp = real+imag*1.j
    
    f_coh = interpolate.interp1d(freq_tmp, coh_tmp)
    coh = f_coh(freq)
    
    return freqresp, coh

    
def _floorpow2(x):
    #2のべき乗に切り下げて丸め込み x>1
    return int(bin(1)+'0'*(len(bin(int(x)))-3), base=2)


def _ceilpow2(x):
    #2のべき乗に切り上げて丸め込み x>1
    return int(bin(1)+'0'*(len(bin(int(x)))-2), base=2)
