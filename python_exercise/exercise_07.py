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
figurefolderName = 'figure_exercise_07'
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
M = 2.0
C = 10.0
K = 0
Kt = 1.0
Pi = ctrl.tf([1], [L, R])
Ps = ctrl.tf([1], [M, C, K])
Pi_frd = ctrl.sys2frd(Pi, freq)
Ps_frd = ctrl.sys2frd(Ps, freq)
print('Plant model was set.')

# Design current PI controller
freqC = 500.0
zetaC = 1.0
Ci = ctrl.pi(freqC, zetaC, L, R)
Ci_frd = ctrl.sys2frd(Ci, freq)
print('Current PI controller was designed.')

# Design position PID controller
freq1 = 100.0
zeta1 = 1.0
freq2 = 100.0
zeta2 = 1.0
Cs = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K)
Cs_frd = ctrl.sys2frd(Cs, freq)
print('Position PID controller was designed.')

print('Frequency response analysis is running...')
Si = ctrl.feedback(Pi, Ci, sys='S')
Ti = ctrl.feedback(Pi, Ci, sys='T')

Ss = ctrl.feedback(Ps, Cs*Ti*Kt, sys='S')
Ts = ctrl.feedback(Ps, Cs*Ti*Kt, sys='T')

Gi_frd = Pi_frd * Ci_frd
Si_frd = 1/(1 + Gi_frd)
Ti_frd = 1 - Si_frd

Gs_frd = Ps_frd * Cs_frd * Ti_frd
Ss_frd = 1/(1 + Gs_frd)
Ts_frd = 1 - Ss_frd

print('Time response analysis is running...')
ti = np.linspace(0.0, 5.0e-3, dataNum)
ri = np.ones(len(ti))
yi, touti, xout = matlab.lsim(Ti, ri, ti)
ei, touti, xout = matlab.lsim(Si, ri, ti)
ui, touti, xout = matlab.lsim(Ci, ei, ti)

tp = np.linspace(0.0, 0.05, dataNum)
rp = np.ones(len(tp))
yp, toutp, xout = matlab.lsim(Ts, rp, tp)
ep, toutp, xout = matlab.lsim(Ss, rp, tp)
up, toutp, xout = matlab.lsim(Cs, ep, tp)

print('Plotting figures...')
# Time response of current
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, ti, yi, '-', 'b', 1.5, 1.0, [0, max(ti)], ylabel='Current [A]', legend=['y'], title='Time response of current')
plot.plot_xy(ax2, ti, ui, '-', 'b', 1.5, 1.0, [0, max(ti)], xlabel='Time [s]', ylabel='Voltage [V]', legend=['u'])
plot.savefig(figurefolderName+'/time_resp_i.png')

# Time response of position
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, tp, yp, '-', 'b', 1.5, 1.0, [0, max(tp)], ylabel='Position [m]', legend=['y'], title='Time response of position')
plot.plot_xy(ax2, tp, up, '-', 'b', 1.5, 1.0, [0, max(tp)], xlabel='Time [s]', ylabel='Force [N]', legend=['u'])
plot.savefig(figurefolderName+'/time_resp_p.png')

# Sensitivity function of current
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Si_frd, '-', 'b', 1.5, 1.0, freqrange, title='Bode diagram of current loop')
plot.plot_tffrd(ax_mag, ax_phase, Ti_frd, '-', 'r', 1.5, 1.0, freqrange, magrange=[-50, 10], legend=['S', 'T'])
plot.savefig(figurefolderName+'/freq_ST_i.png')

# Sensitivity function of position
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Ss_frd, '-', 'b', 1.5, 1.0, freqrange, title='Bode diagram of position loop')
plot.plot_tffrd(ax_mag, ax_phase, Ts_frd, '-', 'r', 1.5, 1.0, freqrange, magrange=[-50, 10], legend=['S', 'T'])
plot.savefig(figurefolderName+'/freq_ST_p.png')

# Nyquist of current
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gi_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram of current loop')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist_i.png')

# Nyquist of position
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gs_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram of position loop')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist_p.png')

print('Finished.')
