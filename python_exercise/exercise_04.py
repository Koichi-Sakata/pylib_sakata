# Copyright (c) 2021 Koichi Sakata


from pylib_sakata import init as init
# uncomment the follows when the file is executed in a Python console.
# init.close_all()
# init.clear_all()

import os
import shutil
import numpy as np
from control import matlab
from pylib_sakata import ctrl
from pylib_sakata import plot

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_exercise_04'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 10000
freqrange = [10, 10000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
print('Common parameters were set.')

# Plant model
L = 0.1
R = 10.0
Ps = ctrl.tf([1], [L, R])
Ps_frd = ctrl.sys2frd(Ps, freq)
print('Plant model was set.')

# Design PI controller
freqC = 1000.0
zetaC = 1.0
Cs = ctrl.pi(freqC, zetaC, L, R)
Cs_frd = ctrl.sys2frd(Cs, freq)
print('PI controller was designed.')

# Design a peak filter
freqF = 1000.0
zetaF = 1.0
omegaF = 2*np.pi*freqF
Fs = ctrl.tf([1, 2*zetaF*omegaF, omegaF**2], [1, 0, omegaF**2])
Fs_frd = ctrl.sys2frd(Fs, freq)
print('A peak filter was designed.')

print('Frequency response analysis is running...')
Cs = Cs * Fs
Cs_frd = Cs_frd * Fs_frd

Ss = ctrl.feedback(Ps, Cs, sys='S')
Ts = ctrl.feedback(Ps, Cs, sys='T')
SCs = ctrl.feedback(Ss, Cs, sys='G')

Gs_frd = Ps_frd * Cs_frd
Ss_frd = 1/(1 + Gs_frd)
Ts_frd = 1 - Ss_frd

print('Time response analysis is running...')
t = np.linspace(0.0, 5.0e-3, dataNum)
r = np.sin(omegaF * t)
y, tout, xout = matlab.lsim(ctrl.tf2ss(Ts), r, t)
e, tout, xout = matlab.lsim(ctrl.tf2ss(Ss), r, t)
u, tout, xout = matlab.lsim(ctrl.tf2ss(Cs), e, t)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, t, r, '--', 'k', 1.5, 1.0, [0, max(t)], title='Time response')
plot.plot_xy(ax1, t, y, '-', 'b', 1.5, 1.0, [0, max(t)], ylabel='Current [A]', legend=['r', 'y'])
plot.plot_xy(ax2, t, u, '-', 'b', 1.5, 1.0, [0, max(t)], xlabel='Time [s]', ylabel='Voltage [V]', legend=['u'])
plot.savefig(figurefolderName+'/time_resp.png')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Ps_frd, '-', 'b', 1.5, 1.0, title='Frequency response of plant')
plot.savefig(figurefolderName+'/freq_P.png')

# PI controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.savefig(figurefolderName+'/freq_C.png')

# A peak filter
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Fs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of peak filter')
plot.savefig(figurefolderName+'/freq_F.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Ss_frd, '-', 'b', 1.5, 1.0, freqrange, title='Bode diagram')
plot.plot_tffrd(ax_mag, ax_phase, Ts_frd, '-', 'r', 1.5, 1.0, freqrange, magrange=[-50, 20], legend=['S', 'T'])
plot.savefig(figurefolderName+'/freq_ST.png')

plot.showfig()
print('Finished.')
