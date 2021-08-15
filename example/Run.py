# Copyright (c) 2021 Koichi Sakata


from pylib_sakata import init as init
# init.close_all()
# init.clear_all()

import os
import shutil
import numpy as np
from scipy import signal
from control import matlab
from pylib_sakata import plot
from pylib_sakata import ctrl
from pylib_sakata import fft

print('Start simulation!')

# Common parameters
figurefolderName = 'figure/'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Tu = 1/4000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = matlab.tf([1, 0],[1])
z = matlab.tf([1, 0],[1], Tu)
print('Common parameters were set.')

# Plant model
M1 = 1.0
M2 = 1.0
M = M1+M2
C = 10.0
K = 0.0
Creso = 10.0
Kreso = 100000.0
k1 = M2/(M1*(M1+M2))
k2 = -1.0/(M1+M2)
omegaPreso = np.sqrt(Kreso*(M1+M2)/(M1*M2))
zetaPreso = 0.5*Creso*np.sqrt((M1+M2)/(Kreso*M1*M2))
Pmechs = matlab.tf([1],[M, C, K]) + k1 * matlab.tf([1],[1, 2*zetaPreso*omegaPreso, omegaPreso**2])
numDelay, denDelay = matlab.pade(Tu*4,n=4)
Ds = matlab.tf(numDelay,denDelay)
Dz = z**-4
Pns = Pmechs * Ds
Pnz = matlab.c2d(Pmechs, Tu, method='zoh') * Dz
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PID controller
zeta1 = 1.0
freq1 = 10.0
zeta2 = 1.0
freq2 = 10.0
Cz = ctrl.pid(zeta1, freq1, zeta2, freq2, M, C, K, Tu)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

# Design phase lead filter
zeta1 = 0.7
freq1 = 60
zeta2 = 0.7
freq2 = 90
PLz = ctrl.pl2nd(zeta1, freq1, zeta2, freq2, Tu)
PLz_frd = ctrl.sys2frd(PLz, freq)
print('Phase lead filters were desinged.')

# Design notch filters
#freqNF = [omegaPreso/(2.0*np.pi), 2000]
#zetaNF = [0.25, 0.2]
#depthNF = [0.08, 0.01]
freqNF = [2000]
zetaNF = [0.2]
depthNF = [0.01]
NFz = ctrl.nf(freqNF, zetaNF, depthNF, Tu)
NFz_frd = 1.0
for i in range(len(NFz)):
    NFz_frd *= ctrl.sys2frd(NFz[i], freq)
print('Notch filters were desinged.')

print('System identification simulation is running...')
Snz = ctrl.feedback(Pnz, Cz*PLz, sys='S')
SPnz = ctrl.feedback(Pnz, Cz*PLz, sys='SP')
t = np.linspace(0.0, 50, int(50/Tu))
chirp = signal.chirp(t, f0=0.1, f1=2000, t1=50, method='logarithmic', phi=-90)
u, tout, xout = matlab.lsim(matlab.tf2ss(Snz), chirp, t)
y, tout, xout = matlab.lsim(matlab.tf2ss(SPnz), chirp, t)
u = np.delete(u, len(u)-1)
y = np.delete(y, len(y)-1)

fft_axis, chirp_fft = fft.fft(chirp, Tu)
fft_axis, u_fft = fft.fft(u, Tu)
fft_axis, y_fft = fft.fft(y, Tu)

Pmeas_frd, coh = fft.tfestimate(u, y, freq, Tu)

print('Frequency respose alanysis is running...')
# Model
Gn_frd = Pnz_frd * Cz_frd * PLz_frd**2 * NFz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd
# Measurement
G_frd = Pmeas_frd * Cz_frd * PLz_frd**2 * NFz_frd
S_frd = 1/(1 + G_frd)
T_frd = 1 - S_frd

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, t, chirp, '-', 'm', 0.5, 1.0, [0, 50], [-3.0, 3.0], ylabel='Input [N]', legend=['Chirp'], title='Time response')
plot.plot_xy(ax2, tout, u, '-', 'b', 0.5, 1.0, [0, 50], [-3.0, 3.0], ylabel='Input [N]', legend=['Servo Out'])
plot.plot_xy(ax3, tout, y*1.0e3, '-', 'b', 0.5, 1.0, [0, 50], [-0.3, 0.3], xlabel='Time [s]', ylabel='Output [mm]', legend=['Position'])
plot.savefig(figurefolderName+'/time_inject.png')

fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, fft_axis, chirp_fft, '-', 'm', 1.5, 1.0, freqrange, ylabel='Input [N]', legend=['Chirp'], title='Power spectrum density', xscale='log')
plot.plot_xy(ax2, fft_axis, u_fft, '-', 'b', 1.5, 1.0, freqrange, ylabel='Input [N]', legend=['Servo Out'], xscale='log')
plot.plot_xy(ax3, fft_axis, y_fft*1.0e6, '-', 'b', 1.5, 1.0, freqrange, xlabel='Frequency [Hz]', ylabel='Output [um]', legend=['Position'], xscale='log')
plot.savefig(figurefolderName+'/fft_inject.png')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(311)
ax_phase = fig.add_subplot(312)
ax_coh = fig.add_subplot(313)
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, freq, '-', 'm', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, freq, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement','Model'])
plot.savefig(figurefolderName+'/freq_P.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, freq, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.savefig(figurefolderName+'/freq_C.png')

# Notch and phase lead filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, PLz_frd * NFz_frd, freq, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of filters')
plot.savefig(figurefolderName+'/freq_NF.png')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, G_frd, freq, '-', 'm', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, freq, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement','Model'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, S_frd, freq, '-', 'm', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, freq, '--', 'b', 1.5, 1.0, freqrange, [-60, 10], legend=['Measurement','Model'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, T_frd, freq, '-', 'm', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, freq, '--', 'b', 1.5, 1.0, freqrange, [-60, 10], legend=['Measurement','Model'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, G_frd, '-', 'm', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn_frd, '--', 'b', 1.5, 1.0, legend=['Measurement','Model'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')