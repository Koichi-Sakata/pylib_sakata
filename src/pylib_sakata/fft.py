# Copyright (c) 2021 Koichi Sakata

# class FreqResp(freq, resp, dt=0)
# fft_axis, fft = fft(data, dt)
# fft_axis, fft_mean = fft_ave(data, dt, windivnum=4, overlap=0.5)
# freqresp, coh = tfestimate(x, y, freq, dt, windivnum=4, overlap=0.5)
# freqresp = frdresize(freqresp, freq)
# y = frdsim(freqresp, u, dt)
# makefrdset(path='.', ftype='cpp', dataNum=100)
# deffrdset(frdata, prmSetName, dataNum=100, path='.', ftype='cpp', mode='a')


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
            if self.resp.ndim == 1:
                other = FreqResp(self.freq, np.array([other] * len(self.resp)), self.dt)
            else:
                other = FreqResp(self.freq, other * np.ones(np.shape(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                raise ValueError('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return FreqResp(self.freq, self.resp + other.resp, self.dt)

    def __radd__(self, other):
        """Right add two FRDs (parallel connection)."""
        return self + other

    def __sub__(self, other):
        """Subtract two FRDs (parallel connection)."""
        return self + (-other)

    def __rsub__(self, other):
        """Right subtract two FRDs (parallel connection)."""
        return other + (-self)

    def __mul__(self, other):
        """Multiply two FRDs (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            if self.resp.ndim == 1:
                other = FreqResp(self.freq, np.array([other] * len(self.resp)), self.dt)
            else:
                other = FreqResp(self.freq, other * np.ones(np.shape(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                raise ValueError('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return FreqResp(self.freq, self.resp * other.resp, self.dt)

    def __rmul__(self, other):
        """Right multiply two FRDs (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            if self.resp.ndim == 1:
                other = FreqResp(self.freq, np.array([other] * len(self.resp)), self.dt)
            else:
                other = FreqResp(self.freq, other * np.ones(np.shape(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                raise ValueError('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return other * self

    def __truediv__(self, other):
        """Divide two FRDs (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            if self.resp.ndim == 1:
                other = FreqResp(self.freq, np.array([other] * len(self.resp)), self.dt)
            else:
                other = FreqResp(self.freq, other * np.ones(np.shape(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                raise ValueError('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return FreqResp(self.freq, self.resp / other.resp, self.dt)

    def __rtruediv__(self, other):
        """Right divide two FRDs (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            if self.resp.ndim == 1:
                other = FreqResp(self.freq, np.array([other] * len(self.resp)), self.dt)
            else:
                other = FreqResp(self.freq, other * np.ones(np.shape(self.resp)), self.dt)
        else:
            if not np.array_equal(self.freq, other.freq):
                raise ValueError('Error: frequency range of FRDs is different.')
            if self.dt != other.dt:
                print('Warning: sampling time of FRDs is different.')
        return other / self

    def __pow__(self, other):
        """A FRD to the power of x."""
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            if self.resp.ndim == 1:
                return FreqResp(self.freq, np.array([1.0] * len(self.resp)), self.dt)  # unity
            else:
                return FreqResp(self.freq, np.ones(np.shape(self.resp)), self.dt)
        if other > 0:
            return self * (self ** (other - 1))
        if other < 0:
            return (1.0 / self) * (self ** (other + 1))

    def __matmul__(self, other):
        """Multiply two MIMO FRDs."""
        if not np.array_equal(self.freq, other.freq):
            raise ValueError('Error: frequency range of FRDs is different.')
        if self.dt != other.dt:
            print('Warning: sampling time of FRDs is different.')
        m0, n0, k = np.shape(self.resp)
        m1, n1, k = np.shape(other.resp)
        if n0 != m1:
            print('Error: matrix sizes are wrong.')
        resultresp = np.empty(shape=(m0, n1, len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, :, k] = np.dot(self.resp[:, :, k], other.resp[:, :, k])
        return FreqResp(self.freq, resultresp, self.dt)

    def __mod__(self, other):
        """Divide two MIMO FRDs."""
        if not np.array_equal(self.freq, other.freq):
            raise ValueError('Error: frequency range of FRDs is different.')
        if self.dt != other.dt:
            print('Warning: sampling time of FRDs is different.')
        m0, n0, k = np.shape(self.resp)
        m1, n1, k = np.shape(other.resp)
        if n0 != n1:
            raise ValueError('Error: matrix sizes are wrong.')
        resultresp = np.empty(shape=(m0, m1, len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, :, k] = np.dot(self.resp[:, :, k], np.linalg.pinv(other.resp[:, :, k]))
        return FreqResp(self.freq, resultresp, self.dt)

    def addeye(self):
        """Add identify matrix to a MIMO FRD."""
        m0, n0, k = np.shape(self.resp)
        if m0 != n0:
            raise ValueError('Error: matrix size is wrong.')
        resultresp = np.empty(shape=(m0, m0, len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, :, k] =  self.resp[:, :, k] + np.eye(m0)
        return FreqResp(self.freq, resultresp, self.dt)

    def inv(self):
        """Inverse matrix of a MIMO FRD."""
        m0, n0, k = np.shape(self.resp)
        if m0 != n0:
            raise ValueError('Error: matrix size is wrong.')
        resultresp = np.empty(shape=(m0, m0, len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, :, k] = np.linalg.inv(self.resp[:, :, k])
        return FreqResp(self.freq, resultresp, self.dt)

    def pinv(self):
        """Moore-Penrose pseudo-inverse matrix of a MIMO FRD."""
        m0, n0, k = np.shape(self.resp)
        resultresp = np.empty(shape=(n0, m0, len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, :, k] = np.linalg.pinv(self.resp[:, :, k])
        return FreqResp(self.freq, resultresp, self.dt)

    def rga(self):
        """Relative gain array of a MIMO FRD."""
        m0, n0, k = np.shape(self.resp)
        resultresp = np.empty(shape=(m0, n0, len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, :, k] =  self.resp[:, :, k] * np.linalg.pinv(self.resp[:, :, k]).T
        return FreqResp(self.freq, resultresp, self.dt)

    def det(self):
        """Determinant of a MIMO FRD."""
        m0, n0, k = np.shape(self.resp)
        resultresp = np.empty(shape=(len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[k] =  np.linalg.det(self.resp[:, :, k])
        return FreqResp(self.freq, resultresp, self.dt)

    def eig(self):
        """Eigen value of a MIMO FRD."""
        m0, n0, k = np.shape(self.resp)
        if m0 != n0:
            raise ValueError('Error: matrix size is wrong.')
        resultresp = np.empty(shape=(m0, len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, k] = np.linalg.eig(self.resp[:, :, k]).eigenvalues
        return FreqResp(self.freq, resultresp, self.dt)

    def svd(self):
        """Singular value of a MIMO FRD."""
        m0, n0, k = np.shape(self.resp)
        resultresp = np.empty(shape=(min(m0, n0), len(self.freq)), dtype=complex)
        for k in range(len(self.freq)):
            resultresp[:, k] =  np.linalg.svd(self.resp[:, :, k]).S
        return FreqResp(self.freq, resultresp, self.dt)


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
    f_real = interpolate.interp1d(freqresp.freq, np.real(freqresp.resp), kind='linear', bounds_error=False, fill_value='extrapolate')
    real = f_real(freq)
    f_imag = interpolate.interp1d(freqresp.freq, np.imag(freqresp.resp), kind='linear', bounds_error=False, fill_value='extrapolate')
    imag = f_imag(freq)
    resp = real + imag * 1.j
    return FreqResp(freq, resp, freqresp.dt)


def frdsim(freqresp, x, dt):
    N = len(x)
    N_odd = N % 2
    NHalf = int(np.ceil((N + 1) / 2))
    freq_tmp = np.arange(N) / N / dt
    freq = freq_tmp[0:NHalf]
    freqresp_resize = frdresize(freqresp, freq)

    # Multiplication at frequency domain for first-half part
    x_fft = np.fft.fft(x)
    Yfft = x_fft[0:NHalf] * freqresp_resize.resp

    # Mirror second-half part
    if N_odd:
        Yfft_flip = np.flip(np.conj(Yfft[1:len(Yfft)]))
    else:
        Yfft_flip = np.flip(np.conj(Yfft[1:len(Yfft) - 1]))

    # Combine two parts
    Yfft_full = np.concatenate([Yfft, Yfft_flip])

    # Calculate time response of output
    return np.real(np.fft.ifft(Yfft_full))

def _floorpow2(x):
    # 2のべき乗に切り下げて丸め込み x>1
    return int(bin(1) + '0' * (len(bin(int(x))) - 3), base=2)


def _ceilpow2(x):
    # 2のべき乗に切り上げて丸め込み x>1
    return int(bin(1) + '0' * (len(bin(int(x))) - 2), base=2)


def makefrdset(path='.', ftype='cpp', dataNum=100):
    path_cpp = path + '/gval_frdprm.'+ ftype
    f = open(path_cpp, 'w')
    f.write('/* Automatically generated by Python script */\n\n')
    if ftype == 'cpp':
        f.write('#include "TcPch.h"\n')
        f.write('#pragma hdrstop\n')
        f.write('#include "head_frdprm.h"\n\n')
    elif ftype == 'c':
        f.write('#include "head_common.h"\n\n')
    path_h = path + '/head_frdprm.h'
    f = open(path_h, 'w')
    f.write('/* Automatically generated by Python script */\n\n')
    f.write('#ifndef _HEAD_FRDPRM_\n')
    f.write('#define _HEAD_FRDPRM_\n\n')
    f.write('#define FRDATA_NUM	')
    f.write(str(int(dataNum)))
    f.write('	    // Number of frdata points\n\n')
    f.write('typedef struct {\n')
    f.write('	double	dFreq[FRDATA_NUM];\n')
    f.write('	double	dReal[FRDATA_NUM];\n')
    f.write('	double	dImag[FRDATA_NUM];\n')
    f.write('} FRD_INF;					// FRD information\n\n')
    f.write('\n#endif\n')


def deffrdset(frdata, prmSetName, dataNum=100, path='.', ftype='cpp', mode='a'):
    path_cpp = path + '/gval_frdprm.' + ftype
    if isinstance(frdata, list):
        frdata = np.array(frdata)
    if type(frdata).__module__ != 'numpy':
        if len(frdata.freq) > dataNum:
            print('Warning: The frdata size is bigger than data_num. Please resize the frdata.')
        f = open(path_cpp, mode)
        f.write('\n')
        f.write('FRD_INF	')
        f.write(prmSetName)
        f.write(' = {\n')
        f.write('	{ ')
        for i in range(len(frdata.freq)):
            if i % 10 == 0:
                f.write('\n')
                f.write('	  ')
            f.write(str(frdata.freq[i]))
            if i != len(frdata.freq) - 1:
                f.write(', ')
        f.write('\n	}, \n')
        f.write('	{ ')
        for i in range(len(frdata.freq)):
            if i % 10 == 0:
                f.write('\n')
                f.write('	  ')
            f.write(str(frdata.resp.real[i]))
            if i != len(frdata.freq) - 1:
                f.write(', ')
        f.write('\n	}, \n')
        f.write('	{ ')
        for i in range(len(frdata.freq)):
            if i % 10 == 0:
                f.write('\n')
                f.write('	  ')
            f.write(str(frdata.resp.imag[i]))
            if i != len(frdata.freq) - 1:
                f.write(', ')
        f.write('\n	}\n')
        f.write('};\n')
        f.close()

        path_h = path + '/head_frdprm.h'
        with open(path_h) as reader:
            content = reader.read()
        content = content.replace('\n#endif\n', '')
        with open(path_h, 'w') as writer:
            writer.write(content)
        f = open(path_h, mode)
        f.write('extern FRD_INF	')
        f.write(prmSetName)
        f.write(';\n')
        f.write('\n#endif\n')
        f.close()
    else:
        if type(frdata[0]).__module__ != 'numpy':
            if len(frdata[0].freq) > dataNum:
                print('Warning: The frdata size is bigger than data_num. Please resize the frdata.')
            f = open(path_cpp, mode)
            f.write('\n')
            f.write('FRD_INF	')
            f.write(prmSetName)
            f.write(' = {\n')
            for k in range(len(frdata)):
                f.write('	{\n')
                f.write('		{ ')
                for i in range(len(frdata[k].freq)):
                    if i % 10 == 0:
                        f.write('\n')
                        f.write('		  ')
                    f.write(str(frdata[k].freq[i]))
                    if i != len(frdata[k].freq) - 1:
                        f.write(', ')
                f.write('\n		},\n')
                f.write('		{ ')
                for i in range(len(frdata[k].freq)):
                    if i % 10 == 0:
                        f.write('\n')
                        f.write('		  ')
                    f.write(str(frdata[k].resp.real[i]))
                    if i != len(frdata[k].freq) - 1:
                        f.write(', ')
                f.write('\n		},\n')
                f.write('		{ ')
                for i in range(len(frdata[k].freq)):
                    if i % 10 == 0:
                        f.write('\n')
                        f.write('		  ')
                    f.write(str(frdata[k].resp.imag[i]))
                    if i != len(frdata[k].freq) - 1:
                        f.write(', ')
                f.write('\n		}\n')
                if k == len(frdata) - 1:
                    f.write('	}\n')
                else:
                    f.write('	}, \n')
            f.write('};\n')

            path_h = path + '/head_frdprm.h'
            with open(path_h) as reader:
                content = reader.read()
            content = content.replace('\n#endif\n', '')
            with open(path_h, 'w') as writer:
                writer.write(content)
            f = open(path_h, mode)
            f.write('extern FRD_INF	')
            f.write(prmSetName)
            f.write(';\n')
            f.write('\n#endif\n')
            f.close()
