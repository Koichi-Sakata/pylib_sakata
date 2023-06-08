# Copyright (c) 2021 Koichi Sakata

# class FreqResp(freq, resp, dt=0)
# fft_axis, fft = fft(data, dt)
# fft_axis, fft_mean = fft_ave(data, dt, windivnum=4, overlap=0.5)
# freqresp, coh = tfestimate(x, y, freq, dt, windivnum=4, overlap=0.5)
# freqresp = frdresize(freqresp, freq)
# y = frdsim(freqresp, u, dt)


import numpy as np
from matplotlib import mlab
from scipy import signal, interpolate


class FreqResp:

    def __init__(self, freq, resp, dt=0):
        if type(freq) == list:
            freq = np.array(freq)
        if type(resp) == list:
            resp = np.array(resp)

        self.freq = freq
        self.resp = resp
        self.dt = dt

    def __repr__(self):
        outstr = ""
        outstr += "\n" + "freq = array(" + str(self.freq) + ")"
        outstr += "\n" + "resp = array(" + str(self.resp) + ")" + "\n"
        # See if this is a discrete time system with specific sampling time
        if not (self.dt is None) and type(self.dt) != bool and self.dt > 0:
            # TODO: replace with standard calls to lti functions
            outstr += "\ndt = " + self.dt.__str__() + "\n"
        return outstr

    def __neg__(self):
        """Negate a FRD."""
        return FreqResp(self.freq, -1.0 * self.resp, self.dt)

    def __add__(self, other):
        """Add two FRDs (parallel connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            other = FreqResp(self.freq, np.array([other]*len(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                print('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return FreqResp(self.freq, self.resp + other.resp, self.dt)

    def __radd__(self, other):
        """Right add two FRDs (parallel connection)."""
        return self + other

    def __sub__(self, other):
        """Subtract two FRDs."""
        return self + (-other)

    def __rsub__(self, other):
        """Right subtract two FRDs."""
        return other + (-self)

    def __mul__(self, other):
        """Multiply two zpk models (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            other = FreqResp(self.freq, np.array([other]*len(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                print('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return FreqResp(self.freq, self.resp * other.resp, self.dt)

    def __rmul__(self, other):
        """Right multiply two zpk models (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            other = FreqResp(self.freq, np.array([other]*len(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                print('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return other * self

    def __truediv__(self, other):
        """Divide two FRDs."""
        if isinstance(other, (int, float, complex, np.number)):
            other = FreqResp(self.freq, np.array([other]*len(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                print('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return FreqResp(self.freq, self.resp / other.resp, self.dt)

    def __rtruediv__(self, other):
        """Right divide two FRDs."""
        if isinstance(other, (int, float, complex, np.number)):
            other = FreqResp(self.freq, np.array([other]*len(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                print('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return other / self

    def __pow__(self, other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return FreqResp(self.freq, np.array([1.0]*len(self.resp)), self.dt)  # unity
        if other > 0:
            return self * (self ** (other - 1))
        if other < 0:
            return (1.0 / self) * (self ** (other + 1))


def fft(data, dt):
    framesize = len(data)
    fft = np.abs(np.fft.fft(data) / (framesize / 2))
    fft_axis = np.linspace(0, 1 / dt, framesize)
    return fft_axis, fft


def fft_ave(data, dt, windivnum=4, overlap=0.5):
    framesize = int(_floorpow2(len(data)) / windivnum)

    # Overlaping
    Tlen = len(data) * dt  # Time of data length
    Tfc = framesize * dt  # Time of frame length
    x_ol = framesize * (1 - overlap)  # オーバーラップ時のフレームずらし幅
    N_ave = int((Tlen - (Tfc * overlap)) / (Tfc * (1 - overlap)))  # data number for average
    data_array = []
    for i in range(N_ave):
        ps = int(x_ol * i)  # 切り出し位置をループ毎に更新
        data_array.append(data[ps:ps + framesize:1])  # 切り出し位置psからフレームサイズ分抽出して配列に追加

    # Hanning Window
    han = signal.windows.hann(framesize)
    acf = 1 / (sum(han) / framesize)  # Amplitude Correction Factor
    for i in range(N_ave):
        data_array[i] = data_array[i] * han

    # FFT average
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf * np.abs(np.fft.fft(data_array[i]) / (framesize / 2)))
    fft_axis = np.linspace(0, 1 / dt, framesize)
    fft_mean = np.sqrt(np.mean(np.array(fft_array) ** 2, axis=0))
    return fft_axis, fft_mean


def tfestimate(x, y, freq, dt, windivnum=4, overlap=0.5):
    x = signal.detrend(x, type='constant')
    y = signal.detrend(y, type='constant')
    NFFT = int(_floorpow2(len(x)) / windivnum)
    Pxy, freq_tmp = mlab.csd(x, y, NFFT=NFFT, Fs=int(1 / dt), window=mlab.window_hanning, noverlap=int(NFFT * overlap))
    Pxx, freq_tmp = mlab.psd(x, NFFT=NFFT, Fs=int(1 / dt), window=mlab.window_hanning, noverlap=int(NFFT * overlap))
    coh_tmp, freq_tmp = mlab.cohere(x, y, NFFT=NFFT, Fs=int(1 / dt), window=mlab.window_hanning,
                                    noverlap=int(NFFT * overlap))
    frd = Pxy / Pxx
    mag_tmp = np.abs(frd)
    phase_tmp = np.angle(frd)
    real_tmp = mag_tmp * np.cos(phase_tmp)
    imag_tmp = mag_tmp * np.sin(phase_tmp)

    # Interpolation
    f_real = interpolate.interp1d(freq_tmp, real_tmp)
    real = f_real(freq)
    f_imag = interpolate.interp1d(freq_tmp, imag_tmp)
    imag = f_imag(freq)
    resp = real + imag * 1.j

    f_coh = interpolate.interp1d(freq_tmp, coh_tmp)
    coh = f_coh(freq)

    return FreqResp(freq, resp, dt), coh


def frdresize(freqresp, freq):
    f_real = interpolate.interp1d(freqresp.freq, np.real(freqresp.resp), kind='linear', bounds_error=False, fill_value=1.)
    real = f_real(freq)
    f_imag = interpolate.interp1d(freqresp.freq, np.imag(freqresp.resp), kind='linear', bounds_error=False, fill_value=0.)
    imag = f_imag(freq)
    resp = real + imag * 1.j
    return FreqResp(freq, resp, freqresp.dt)


def frdsim(freqresp, x, dt):
    # Resie freqresp
    lenInput = len(x)
    lenInputHalf = round((lenInput + 1)/2)
    freq_tmp = np.arange(lenInput)/lenInput/dt
    freq = freq_tmp[0:lenInputHalf]
    freqresp_resize = frdresize(freqresp, freq)

    # Multiplication at frequency domain for first-half part
    x_fft = np.fft.fft(x)
    y_fft = x_fft[0:lenInputHalf] * freqresp_resize.resp
    y_fft[0] = np.real(y_fft[0])    # Change zero-frequency-point real

    # Mirror second-half part
    y_fft_flip = np.flip(np.conj(y_fft[1:len(y_fft)]))
    # Combine two parts
    y_fft_full = np.concatenate([y_fft, y_fft_flip])

    # Calculate time response of output
    y = np.real(np.fft.ifft(y_fft_full))
    t = np.linspace(0, (len(y) - 1) * dt, len(y))
    return t, y


def _floorpow2(x):
    # 2のべき乗に切り下げて丸め込み x>1
    return int(bin(1) + '0' * (len(bin(int(x))) - 3), base=2)


def _ceilpow2(x):
    # 2のべき乗に切り上げて丸め込み x>1
    return int(bin(1) + '0' * (len(bin(int(x))) - 2), base=2)
