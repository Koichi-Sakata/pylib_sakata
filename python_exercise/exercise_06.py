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
figurefolderName = 'figure_exercise_06'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 10000
freqrange = [0.1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
print('Common parameters were set.')

# Plant model
M = 2.0
C = 10.0
K = 0
Ps = ctrl.tf([1], [M, C, K])
Ps_frd = ctrl.sys2frd(Ps, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 10.0
zeta1 = 1.0
freq2 = 10.0
zeta2 = 1.0
Cs = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K)
Cs_frd = ctrl.sys2frd(Cs, freq)
print('PID controller was designed.')

print('Frequency response analysis is running...')
Ss = ctrl.feedback(Ps, Cs, sys='S')
Ts = ctrl.feedback(Ps, Cs, sys='T')

Gs_frd = Ps_frd * Cs_frd
Ss_frd = 1/(1 + Gs_frd)
Ts_frd = 1 - Ss_frd

print('Time response analysis is running...')
t = np.linspace(0.0, 0.5, dataNum)
r = np.ones(len(t))
y, tout, xout = matlab.lsim(Ts, r, t)
e, tout, xout = matlab.lsim(Ss, r, t)
u, tout, xout = matlab.lsim(Cs, e, t)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, t, y, '-', 'b', 1.5, 1.0, [0, max(t)], ylabel='Position [m]', legend=['y'], title='Time response')
plot.plot_xy(ax2, t, u, '-', 'b', 1.5, 1.0, [0, max(t)], xlabel='Time [s]', ylabel='Force [N]', legend=['u'])
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

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Ss_frd, '-', 'b', 1.5, 1.0, freqrange, title='Bode diagram')
plot.plot_tffrd(ax_mag, ax_phase, Ts_frd, '-', 'r', 1.5, 1.0, freqrange, magrange=[-50, 10], legend=['S', 'T'])
plot.savefig(figurefolderName+'/freq_ST.png')

print('Finished.')
