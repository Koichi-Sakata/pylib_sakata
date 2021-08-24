# Copyright (c) 2021 Koichi Sakata

# class ZpkModel(zeros, poles, gain, dt=0)
# tf = tf(num, den, dt=0)
# ss = ss(A, B, C, D, dt=0)
# zpk = zpk(z, p, k, dt=0)
# ss = tf2ss(tf, form='reachable')
# zpk = tf2zpk(tf)
# tf = ss2tf(ss)
# zpk = ss2zpk(ss)
# tf = zpk2tf(zpk)
# ss = zpk2ss(zpk, form='reachable')
# frd = sys2frd(sys, freq)
# sys = feedback(sysP, sysC, sys='S')
# frdout = frdfeedback(frdP, frdC, sys='S')
# sysD = c2d(sysC, dt, method='tustin')
# sys = pid(freq1, zeta1, freq2, zeta2, M, C, K, dt=None, method='tustin')
# sys = pl1st(freq1, freq2, dt=None, method='tustin')
# sys = pl2nd(freq1, zeta1, freq2, zeta2, dt=None, method='tustin')
# sys = lpf1st(freq, dt=None, method='tustin')
# sys = lpf2nd(freq, zeta, dt=None, method='tustin')
# sys = hpf1st(freq, dt=None, method='tustin')
# sys = hpf2nd(freq, zeta, dt=None, method='tustin')
# sys = nf(freq, zeta, depth, dt=None, method='matched')
# sys = pf(freq, zeta, k, phi, dt=None, method='matched')
# sys = pfopt(freq, zeta, kpdb, sysT, dt=None, method='matched')


import math
import numpy as np
from numpy.polynomial import polynomial
import control
from control import matlab
from .fft import FreqResp


class ZpkModel:
 
    def __init__(self, zeros, poles, gain, dt=0):
        # gain: not system dc gain but coefficient of monic polynomials!
        zeros = list(zeros)
        poles = list(poles)
        
        compz = _list_com(zeros, poles)
        zeros = np.array(_list_dif(zeros, compz))
        poles = np.array(_list_dif(poles, compz))
        self.z = np.array(zeros)
        self.p = np.array(poles)
        if len(compz) > 0:
            print('The common pole-zeros of the zpk model have been deleted.')
        self.k = np.real(gain)
        self.dt = dt

    def __repr__(self):
        var = 's' if self.dt is None or self.dt == 0 else 'z'
        outstr = ""
        # Convert the numerator and denominator polynomials to strings.
        numstr = _zpk_polynomial_to_string(self.z, var=var)
        denstr = _zpk_polynomial_to_string(self.p, var=var)
        gainstr = '%.4g' % self.k + ' * '
        # Figure out the length of the separating line
        dashcount = max(len(numstr), len(denstr))
        dashes = '-' * dashcount

        # Center the numerator or denominator
        if len(numstr) < dashcount:
            numstr = ' ' * ((dashcount - len(numstr)) // 2) + numstr
        if len(denstr) < dashcount:
            denstr = ' ' * ((dashcount - len(denstr)) // 2) + denstr

        outstr += "\n" + ' ' * len(gainstr) + numstr + "\n" + gainstr + dashes + "\n" + ' ' * len(gainstr) + denstr + "\n"

        # See if this is a discrete time system with specific sampling time
        if not (self.dt is None) and type(self.dt) != bool and self.dt > 0:
            # TODO: replace with standard calls to lti functions
            outstr += "\ndt = " + self.dt.__str__() + "\n"
        return outstr
    
    def __neg__(self):
        """Negate a zpk model."""
        return ZpkModel(self.z, self.p, -1.0*self.k, self.dt)
    
    def __add__(self, other):
        """Add two zpk models (parallel connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            other = ZpkModel([], [], other)
        else:
            if self.dt != other.dt:
                print('Warning: sampling time of zpk systems is different.')
        poles = np.concatenate([self.p, other.p])
        zeros1 = np.concatenate([self.z, other.p])
        zeros2 = np.concatenate([other.z, self.p])
        # Reduction of fraction
        comp = _list_com(self.p, other.p)
        poles = np.array(_list_dif(poles, comp))
        zeros1 = np.array(_list_dif(zeros1, comp))
        zeros2 = np.array(_list_dif(zeros2, comp))

        num1 = self.k * polynomial.polyfromroots(zeros1)
        num2 = other.k * polynomial.polyfromroots(zeros2)
        num = polynomial.polyadd(num1, num2)
        zeros = polynomial.polyroots(num)
        gain = num[-1]
        return ZpkModel(zeros, poles, gain, self.dt)

    def __radd__(self, other):
        """Right add two zpk models (parallel connection)."""
        return self + other
    
    def __sub__(self, other):
        """Subtract two zpk models."""
        return self + (-other)

    def __rsub__(self, other):
        """Right subtract two zpk models."""
        return other + (-self)

    def __mul__(self, other):
        """Multiply two zpk models (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            other = ZpkModel([], [], other, self.dt)
        else:
            if self.dt != other.dt:
                print('Warning: sampling time of zpk systems is different.')
        zeros = np.concatenate([self.z, other.z])
        poles = np.concatenate([self.p, other.p])
        gain = self.k * other.k
        zpkout = ZpkModel(zeros, poles, gain, self.dt)
        return zpkout

    def __rmul__(self, other):
        """Right multiply two zpk models (serial connection)."""
        if isinstance(other, (int, float, complex, np.number)):
            other = ZpkModel([], [], other, self.dt)
        else:
            if self.dt != other.dt:
                print('Warning: sampling time of zpk systems is different.')
        return other * self

    def __truediv__(self, other):
        """Divide two zpk models."""
        if isinstance(other, (int, float, complex, np.number)):
            other = ZpkModel([], [], other, self.dt)
        else:
            if self.dt != other.dt:
                print('Warning: sampling time of zpk systems is different.')
        zeros = np.concatenate([self.z, other.p])
        poles = np.concatenate([self.p, other.z])
        gain = self.k / other.k
        return ZpkModel(zeros, poles, gain, self.dt)

    def __rtruediv__(self, other):
        """Right divide two zpk models."""
        if isinstance(other, (int, float, complex, np.number)):
            other = ZpkModel([], [], other, self.dt)
        else:
            if self.dt != other.dt:
                print('Warning: sampling time of zpk systems is different.')
        return other / self
    
    def __pow__(self, other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return ZpkModel([], [], 1.0, self.dt)  # unity
        if other > 0:
            return self * (self**(other - 1))
        if other < 0:
            return (1.0 / self) * (self**(other + 1))

    def feedback(self, other=1, sys='S'):
        """Feedback interconnection between two zpk objects."""
        if sys == 'S':
            return 1/(1+self*other)
        elif sys == 'T':
            return self*other/(1+self*other)
        else: # sys == 'SP':
            return self/(1+self*other)


def _list_com(list1, list2):
    list1 = list(list1)
    list2 = list(list2)
    if len(list2) > len(list1):
        list_tmp = list1
        list1 = list2
        list2 = list_tmp
    out = []
    list1c = list1.copy()
    for value in list2:
        if value in list1c:
            num_pre = len(list1c)
            list1c.remove(value)
            num = len(list1c)
            if num_pre != num:
                out.append(value)
    return out


def _list_dif(list1, list2):
    list1 = list(list1)
    list2 = list(list2)
    if len(list2) > len(list1):
        list_tmp = list1
        list1 = list2
        list2 = list_tmp
    out = list1.copy()
    for value in list2:
        if value in out:
            out.remove(value)
    return out


def _zpk_polynomial_to_string(zp, var='s'):
    """Convert a pole or zero pair to a string"""

    thestr = "1"

    # Compute the number of coefficients
    N = len(zp) - 1

    for k in range(len(zp)):
        if type(zp[k]) == np.complex128:
            if np.imag(zp[k]) == 0:
                zpstr = '%.4g' % abs(np.real(zp[k]))
            elif np.imag(zp[k]) > 0:
                zpstr = '%.4g' % abs(np.real(zp[k])) + '+' + '%.4g' % np.imag(zp[k]) + 'j'
            else:
                zpstr = '%.4g' % abs(np.real(zp[k])) + '%.4g' % np.imag(zp[k]) + 'j'
        else:
            zpstr = '%.4g' % abs(zp[k])

        if np.real(zp[k]) == 0:
            newstr = var
        elif np.real(zp[k]) > 0:
            newstr = '(' + var + '-' + zpstr + ')'
        else:
            newstr = '(' + var + '+' + zpstr + ')'

        if k > 0:
            if newstr != '':
                thestr = "%s%s" % (thestr, newstr)
        else:
            thestr = newstr
    return thestr


def tf(num, den, dt=0):
    return matlab.tf(num, den, dt)


def ss(A, B, C, D, dt=0):
    return matlab.ss(A, B, C, D, dt)


def zpk(z, p, k, dt=0):
    return ZpkModel(z, p, k, dt)


def tf2ss(tf, form='reachable'):
    # form: 'reachable' or 'observable' or 'modal'
    ss, T = control.canonical_form(matlab.tf2ss(tf), form=form)
    return ss


def tf2zpk(tf):
    zeros = tf.zero()
    poles = tf.pole()
    gain = tf.num[0][0][0]/tf.den[0][0][0]
    dt = tf.dt
    zpk = ZpkModel(zeros, poles, gain, dt)
    return zpk


def ss2tf(ss):
    return matlab.ss2tf(ss)


def ss2zpk(ss):
    tf = matlab.ss2tf(ss)
    return tf2zpk(tf)


def zpk2tf(zpk):
    zeros = zpk.z
    poles = zpk.p
    gain = zpk.k
    dt = zpk.dt
    num = np.real(gain*np.flip(polynomial.polyfromroots(zeros)))
    den = np.real(np.flip(polynomial.polyfromroots(poles)))
    return matlab.tf(num, den, dt)


def zpk2ss(zpk, form='reachable'):
    # form: 'reachable' or 'observable' or 'modal'
    tf = zpk2tf(zpk)
    return tf2ss(tf, form)


def sys2frd(sys, freq):
    if type(sys) == ZpkModel:
        sys = zpk2tf(sys)
    if type(freq) == list:
        freq = np.array(freq)
    mag, phase, omega = matlab.freqresp(sys, freq*2.0*np.pi)
    real = mag*np.cos(phase)
    imag = mag*np.sin(phase)
    resp = real+imag*1.j
    return FreqResp(freq, resp, sys.dt)


def feedback(sysP, sysC, sys='S'):
    if type(sysP) == ZpkModel:
        zpkP = sysP
    elif type(sysP) == matlab.TransferFunction:
        zpkP = tf2zpk(sysP)
    else: # type(sysP) == matlab.StateSpace
        zpkP = ss2zpk(sysP)
    if type(sysC) == ZpkModel:
        zpkC = sysC
    elif type(sysC) == matlab.TransferFunction:
        zpkC = tf2zpk(sysC)
    else: # type(sysC) == matlab.StateSpace
        zpkC = ss2zpk(sysC)

    if sys == 'S':
        zpkout =  1/(1+zpkP*zpkC)
    elif sys == 'T':
        zpkout =  zpkP*zpkC/(1+zpkP*zpkC)
    else: # sys == 'SP':
        zpkout =  zpkP/(1+zpkP*zpkC)

    if type(sysP) == ZpkModel:
        return zpkout
    elif type(sysP) == matlab.TransferFunction:
        return zpk2tf(zpkout)
    else: # type(sysP) == matlab.StateSpace
        return zpk2ss(zpkout)


def frdfeedback(frdP, frdC, sys='S'):
    if sys == 'S':
        frdout = 1.0/(1.0 + frdP * frdC)
    elif sys == 'T':
        frdout = 1.0 - 1.0/(1.0 + frdP * frdC)
    else: # sys == 'SP'
        frdout = frdP/(1.0 + frdP * frdC)
    return frdout


def c2d(sysC, dt, method='tustin'):
    if method != 'matched':
        return matlab.c2d(sysC, dt, method=method)
    else:
        # Pole-zero match method of continuous to discrete time conversion
        if type(sysC) == ZpkModel:
            zpk = sysC
        elif type(sysC) == matlab.TransferFunction:
            zpk = tf2zpk(sysC)
        else: # type(sysC) == matlab.StateSpace
            zpk = ss2zpk(sysC)
        szeros = zpk.z
        spoles = zpk.p
        sgain = abs(sys2frd(sysC, 0.01).resp[0])
        zzeros = [0] * len(szeros)
        zpoles = [0] * len(spoles)
        for idx, s in enumerate(szeros):
            z = np.exp(s * dt)
            zzeros[idx] = z
        for idx, s in enumerate(spoles):
            z = np.exp(s * dt)
            zpoles[idx] = z
        sysDpre = ZpkModel(zzeros, zpoles, 1.0, dt)
        zgain = abs(sys2frd(sysDpre, 0.01).resp[0])
        sysD = sysDpre * sgain / zgain
        if type(sysC) == ZpkModel:
            return sysD
        elif type(sysC) == matlab.TransferFunction:
            return zpk2tf(sysD)
        else: # type(sysC) == matlab.StateSpace
            return zpk2ss(sysD)

    
def pid(freq1, zeta1, freq2, zeta2, M, C, K, dt=None, method='tustin'):
    # PID controller
    omega1 = 2.0*np.pi*freq1
    omega2 = 2.0*np.pi*freq2
    ac1 = 2.0*(zeta1*omega1+zeta2*omega2)-C/M
    bc2 = M*(omega1**2+omega2**2+4.0*zeta1*zeta2*omega1*omega2)-C*ac1-K
    bc1 = 2.0*M*(zeta1*omega1*omega2**2+zeta2*omega2*omega1**2)-K*ac1
    bc0 = M*omega1**2 * omega2**2
    num = [bc2, bc1, bc0]
    den = [1, ac1, 0]
    TFs = matlab.tf(num, den)
    if dt == None:
        return TFs
    else:
        TFz = c2d(TFs, dt, method=method)
        return TFz


def pl1st(freq1, freq2, dt=None, method='tustin'):
    # 1st ordeer phase lead filter
    TFs = freq2/freq1 * matlab.tf([1.0, 2.0*np.pi*freq1],[1.0, 2.0*np.pi*freq2])
    if dt == None:
        return TFs
    else:
        TFz = c2d(TFs, dt, method=method)
        return TFz


def pl2nd(freq1, zeta1, freq2, zeta2, dt=None, method='tustin'):
    # 2nd ordeer phase lead filter
    omega1 = 2.0*np.pi*freq1
    omega2 = 2.0*np.pi*freq2
    TFs = (freq2/freq1)**2 * matlab.tf([1.0, 2.0*zeta1*omega1, omega1**2],[1.0, 2.0*zeta2*omega2, omega2**2])
    if dt == None:
        return TFs
    else:
        TFz = c2d(TFs, dt, method=method)
        return TFz


def lpf1st(freq, dt=None, method='tustin'):
    # 1st order low pass filter
    omega = 2.0*np.pi*freq
    TFs = matlab.tf([omega],[1.0, omega])
    if dt == None:
        return TFs
    else:
        TFz = c2d(TFs, dt, method=method)
        return TFz


def lpf2nd(freq, zeta, dt=None, method='tustin'):
    # 2nd order low pass filter
    omega = 2.0*np.pi*freq
    TFs = matlab.tf([omega**2],[1.0, 2.0*zeta*omega, omega**2])
    if dt == None:
        return TFs
    else:
        TFz = c2d(TFs, dt, method=method)
        return TFz


def hpf1st(freq, dt=None, method='tustin'):
    # 1st order high pass filter
    omega = 2.0*np.pi*freq
    TFs = matlab.tf([1.0, 0],[1.0, omega])
    if dt == None:
        return TFs
    else:
        TFz = c2d(TFs, dt, method=method)
        return TFz


def hpf2nd(freq, zeta, dt=None, method='tustin'):
    # 2nd order high pass filter
    omega = 2.0*np.pi*freq
    TFs = matlab.tf([1.0, 2.0*zeta*omega, 0],[1.0, 2.0*zeta*omega, omega**2])
    if dt == None:
        return TFs
    else:
        TFz = c2d(TFs, dt, method=method)
        return TFz


def nf(freq, zeta, depth, dt=None, method='matched'):
    # Notch filter
    if (len(freq)==len(zeta)==len(depth)) == False:
        print('Error: length of notch filter parameters is different!')
    omega = 2.0*np.pi*np.array(freq)
    TFs = np.array([matlab.tf([1.0],[1.0]) for i in range(len(freq))])
    TFz = np.array([matlab.tf([1.0],[1.0], dt) for i in range(len(freq))])
    for i in range(len(freq)):
        TFs[i] = matlab.tf([1.0,2.0*depth[i]*zeta[i]*omega[i],omega[i]**2],[1.0,2.0*zeta[i]*omega[i],omega[i]**2])
    if dt == None:
        return TFs
    else:
        for i in range(len(freq)):
            TFz[i] = c2d(TFs[i], dt, method=method)
        return TFz


def pf(freq, zeta, k, phi, dt=None, method='tustin'):
    # Peak filter
    if (len(freq)==len(zeta)==len(k)==len(phi)) == False:
        print('Error: length of peak filter parameters is different!')
    omega = 2.0*np.pi*np.array(freq)
    TFs = np.array([matlab.tf([0.0],[1.0]) for i in range(len(freq))])
    TFz = np.array([matlab.tf([0.0],[1.0], dt) for i in range(len(freq))])
    for i in range(len(freq)):
        TFs[i] = matlab.tf([k[i], -k[i]*phi[i], 0],[1.0, 2.0*zeta[i]*omega[i], omega[i]**2])
    if dt == None:
        return TFs
    else:
        for i in range(len(freq)):
            TFz[i] = c2d(TFs[i], dt, method=method)
        return TFz


def pfopt(freq, zeta, depth, sysT, dt=None, method='tustin'):
    # Optimized peak filter
    freq, zeta, k, phi = _pfoptparam(freq, zeta, depth, sysT)
    if dt == None:
        TFs = pf(freq, zeta, k, phi)
        return TFs
    else:
        TFz = pf(freq, zeta, k, phi, dt=dt, method=method)
        return TFz


def _pfoptparam(freq, zeta, depth, sysT):
    # Optimized peak filter
    if (len(freq)==len(zeta)==len(depth)) == False:
        print('Error: length of peak filter parameters is different!')
    omega = 2.0*np.pi*np.array(freq)
    invsysT = 1.0 / sysT
    if type(invsysT) != FreqResp:
        invsysT = sys2frd(invsysT, np.array(freq))
    re_invT = np.array([])
    im_invT = np.array([])
    for i in range(len(freq)):
        idx = np.argmin(np.abs(invsysT.freq - freq[i]))
        re_invT = np.append(re_invT, np.real(invsysT.resp[idx]))
        im_invT = np.append(im_invT, np.imag(invsysT.resp[idx]))

    norm_invT = np.sqrt(re_invT**2+im_invT**2)
    phi = -omega*re_invT/im_invT
    k = 2*np.array(zeta)*omega*(1.0/np.array(depth)-1.0)/np.sqrt(phi**2 + omega**2)*norm_invT
    for i in range(len(freq)):
        if (1-2.0*zeta[i]*2 <= 0):
            print('Warning: peak does not exit at the frequency.')
        phaseT = math.atan2(im_invT[i], re_invT[i])
        phaseF = math.atan2(omega[i], -phi[i])
        if phaseT/phaseF < 0:
            k[i] *= -1
    return freq, zeta, k, phi
