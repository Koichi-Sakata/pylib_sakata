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
figurefolderName = 'figure_exercise_02-03-advanced'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 10000
freqrange = [10, 10000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
print('Common parameters were set.')

# Plant model
L = 0.1
R = 10
Ps = ctrl.tf([1], [L, R])
Ps_frd = ctrl.sys2frd(Ps, freq)
print('Plant model was set.')

# Design PI controller
freqC = 1000.0
zetaC = 1.0
Cs = ctrl.pi(freqC, zetaC, L, R)
Cs_frd = ctrl.sys2frd(Cs, freq)

freqC = 2000.0
Cs_pzcancel = 2.0 * np.pi * freqC * ctrl.tf([L, R], [1, 0])
Cs_pzcancel_frd = ctrl.sys2frd(Cs_pzcancel, freq)
print('PI controller was designed.')

print('Frequency response analysis is running...')
Ss = ctrl.feedback(Ps, Cs, sys='S')
Ts = ctrl.feedback(Ps, Cs, sys='T')
SPs = ctrl.feedback(Ps, Cs, sys='SP')

Ss_pzcancel = ctrl.feedback(Ps, Cs_pzcancel, sys='S')
Ts_pzcancel = ctrl.feedback(Ps, Cs_pzcancel, sys='T')
SPs_pzcancel = ctrl.feedback(Ps, Cs_pzcancel, sys='SP')

Gs_frd = Ps_frd * Cs_frd
Ss_frd = 1/(1 + Gs_frd)
Ts_frd = 1 - Ss_frd

Gs_pzcancel_frd = Ps_frd * Cs_pzcancel_frd
Ss_pzcancel_frd = 1/(1 + Gs_pzcancel_frd)
Ts_pzcancel_frd = 1 - Ss_pzcancel_frd

print('Time response analysis is running...')
t = np.linspace(0.0, 5.0e-3, dataNum)
r = np.ones(len(t))
y1, tout, xout = matlab.lsim(ctrl.tf2ss(Ts), r, t)
e1, tout, xout = matlab.lsim(ctrl.tf2ss(Ss), r, t)
u1, tout, xout = matlab.lsim(ctrl.tf2ss(Cs), e1, t)
y2, tout, xout = matlab.lsim(ctrl.tf2ss(Ts_pzcancel), r, t)
e2, tout, xout = matlab.lsim(ctrl.tf2ss(Ss_pzcancel), r, t)
u2, tout, xout = matlab.lsim(ctrl.tf2ss(Cs_pzcancel), e2, t)

d = np.concatenate([np.zeros(1000), 10 * np.ones(len(t)-1000)])
y3, tout, xout = matlab.lsim(ctrl.tf2ss(SPs), d, t)
y4, tout, xout = matlab.lsim(ctrl.tf2ss(SPs_pzcancel), d, t)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, t, y1, '-', 'b', 1.5, 1.0, [0, max(t)], title='Time response')
plot.plot_xy(ax1, t, y2, '-', 'r', 1.5, 1.0, [0, max(t)], ylabel='Current [A]', legend=['y: 2nd order poleplacement', 'y: Pole-zero cancellation'])
plot.plot_xy(ax2, t, u1, '-', 'b', 1.5, 1.0, [0, max(t)])
plot.plot_xy(ax2, t, u2, '-', 'r', 1.5, 1.0, [0, max(t)], xlabel='Time [s]', ylabel='Voltage [V]', legend=['u: 2nd order poleplacement', 'u: Pole-zero cancellation'])
plot.savefig(figurefolderName+'/time_resp.png')

fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, t, d, '-', 'b', 1.5, 1.0, ylabel='Disturbance [V]', legend=['d'], title='Time response')
plot.plot_xy(ax2, t, y3*1.0e3, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax2, t, y4*1.0e3, '-', 'r', 1.5, 1.0, [0, max(t)], xlabel='Time [s]', ylabel='Current [mA]', legend=['y: 2nd order poleplacement', 'y: Pole-zero cancellation'])
plot.savefig(figurefolderName+'/time_dist.png')

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
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PI controller')
plot.plot_tffrd(ax_mag, ax_phase, Cs_pzcancel_frd, '-', 'r', 1.5, 1.0, freqrange, legend=['2nd order poleplacement', 'Pole-zero cancellation'])
plot.savefig(figurefolderName+'/freq_C.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Ss_frd, '-', 'b', 1.5, 1.0, title='Bode diagram')
plot.plot_tffrd(ax_mag, ax_phase, Ts_frd, '-', 'c', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Ss_pzcancel_frd, '-', 'r', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Ts_pzcancel_frd, '-', 'm', 1.5, 1.0, freqrange, magrange=[-50, 10], legend=['S: 2nd order poleplacement', 'T: 2nd order poleplacement', 'S: Pole-zero cancellation', 'T: Pole-zero cancellation'])
plot.savefig(figurefolderName+'/freq_ST.png')

plot.showfig()
print('Finished.')
