# Copyright (c) 2021 Koichi Sakata


from pylib_sakata import init as init
# uncomment the follows when the file is executed in a Python console.
# init.close_all()
# init.clear_all()

import os
import shutil
import numpy as np
from scipy import signal
from control import matlab
from pylib_sakata import ctrl
from pylib_sakata import plot
from pylib_sakata import fft

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_sysid'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Ts = 1/8000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# Plant model
M = 2.0
C = 10.0
K = 0.0
Pmechs = ctrl.tf([1.0], [M, C, K])
Dz = z**-3
Pnz = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PD controller
freq1 = 10.0
freq2 = 10.0
zeta2 = 1.0
Cz = ctrl.pd(freq1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PD controller was designed.')

print('System identification simulation is running...')
Snz = ctrl.feedback(Pnz, Cz, sys='S')
SPnz = ctrl.feedback(Pnz, Cz, sys='SP')
t = np.linspace(0.0, 50, int(50/Ts))
chirp = signal.chirp(t, f0=0.1, f1=2000, t1=50, method='logarithmic', phi=-90)
u, tout, xout = matlab.lsim(ctrl.tf2ss(Snz), chirp)
y, tout, xout = matlab.lsim(ctrl.tf2ss(SPnz), chirp)

fft_axis, chirp_fft = fft.fft(chirp, Ts)
fft_axis, u_fft = fft.fft(u, Ts)
fft_axis, y_fft = fft.fft(y, Ts)

Pmeas_frd, coh = fft.tfestimate(u, y, freq, Ts)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, t, chirp, '-', 'm', 0.5, 1.0, [0, 50], ylabel='Sig [N]', title='Time response')
plot.plot_xy(ax2, tout, u, '-', 'b', 0.5, 1.0, [0, 50], ylabel='In [N]')
plot.plot_xy(ax3, tout, y*1.0e3, '-', 'b', 0.5, 1.0, [0, 50], xlabel='Time [s]', ylabel='Out [mm]')
plot.savefig(figurefolderName+'/time_inject.svg')

fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, fft_axis, chirp_fft, '-', 'm', 1.5, 1.0, freqrange, ylabel='Sig [N]', title='Power spectrum density', xscale='log')
plot.plot_xy(ax2, fft_axis, u_fft, '-', 'b', 1.5, 1.0, freqrange, ylabel='In [N]', xscale='log')
plot.plot_xy(ax3, fft_axis, y_fft*1.0e6, '-', 'b', 1.5, 1.0, freqrange, xlabel='Frequency [Hz]', ylabel='Out [um]', xscale='log')
plot.savefig(figurefolderName+'/fft_inject.svg')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(311)
ax_phase = fig.add_subplot(312)
ax_coh = fig.add_subplot(313)
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, '-', 'm', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_P.svg')

plot.showfig()
print('Finished.')
