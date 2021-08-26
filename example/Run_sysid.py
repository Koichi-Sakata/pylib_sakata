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
Ts = 1/4000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# Plant model
M = 2.0
C = 10.0
K = 0
Pmechs = ctrl.tf([1], [M, C, K])
numDelay, denDelay = matlab.pade(Ts*4, n=4)
Ds = ctrl.tf(numDelay, denDelay)
Dz = z**-4
Pns = Pmechs * Ds
Pnz = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 10.0
zeta1 = 1.0
freq2 = 10.0
zeta2 = 1.0
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

print('System identification simulation is running...')
Snz = ctrl.feedback(Pnz, Cz, sys='S')
SPnz = ctrl.feedback(Pnz, Cz, sys='SP')
t = np.linspace(0.0, 50, int(50/Ts))
chirp = signal.chirp(t, f0=0.1, f1=2000, t1=50, method='logarithmic', phi=-90)
u, tout, xout = matlab.lsim(ctrl.tf2ss(Snz), chirp, t)
y, tout, xout = matlab.lsim(ctrl.tf2ss(SPnz), chirp, t)
u = np.delete(u, len(u)-1)
y = np.delete(y, len(y)-1)

fft_axis, chirp_fft = fft.fft(chirp, Ts)
fft_axis, u_fft = fft.fft(u, Ts)
fft_axis, y_fft = fft.fft(y, Ts)

Pmeas_frd, coh = fft.tfestimate(u, y, freq, Ts)

print('Frequency respose alanysis is running...')
# Model
Gn_frd = Pnz_frd * Cz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd
# Measurement
G_frd = Pmeas_frd * Cz_frd
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
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, '-', 'm', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_P.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.savefig(figurefolderName+'/freq_C.png')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, G_frd, '-', 'm', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, S_frd, '-', 'm', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, '--', 'b', 1.5, 1.0, freqrange, [-60, 10], legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, T_frd, '-', 'm', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, '--', 'b', 1.5, 1.0, freqrange, [-60, 10], legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, G_frd, '-', 'm', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn_frd, '--', 'b', 1.5, 1.0, legend=['Measurement', 'Model'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')