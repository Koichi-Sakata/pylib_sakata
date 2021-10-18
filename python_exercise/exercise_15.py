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
from pylib_sakata import traj
from pylib_sakata import plot

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_exercise_15'
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
Dz = z**-4
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

# Design ZPETC
Czpetc, Nzpetc = ctrl.zpetc(Pnz, Ts)
Czpetc_frd = ctrl.sys2frd(Czpetc, freq)
lead_frd = ctrl.sys2frd(z**Nzpetc, freq)
print('ZPETC was designed.')

print('Frequency response analysis is running...')
Gn_frd = Pnz_frd * Cz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd

print('Time response analysis is running...')
Snz = ctrl.feedback(Pnz, Cz, sys='S')
traj = traj.traj4th(0, 1.0, 0.5, 1.0, 0.0005, 0.5)
t_4th = traj.time
r_4th = traj.pos
e1, tout, xout = matlab.lsim(Snz, r_4th, t_4th)
e2, tout, xout = matlab.lsim((z**-Nzpetc-Czpetc*Pnz), r_4th, t_4th)
e2, tout, xout = matlab.lsim(Snz, e2, t_4th)
u1, tout, xout = matlab.lsim(Cz, e1, t_4th)
u2, tout, xout = matlab.lsim(Cz, e2, t_4th)


print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, t_4th, r_4th, '-', 'b', 1.5, 1.0, ylabel='Ref Pos [m]', title='Time response')
plot.plot_xy(ax2, t_4th, e1*1.0e3, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax2, t_4th, e2*1.0e3, '-', 'r', 1.5, 1.0, yrange=[-0.3, 0.3], xlabel='Time [s]', ylabel='Error Pos [mm]', legend=['w/o ZPETC', 'with ZPETC'])
plot.plot_xy(ax3, t_4th, u1, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax3, t_4th, u2, '-', 'r', 1.5, 1.0, yrange=[-4.0, 8.0], xlabel='Time [s]', ylabel='FB Out [N]', legend=['w/o ZPETC', 'with ZPETC'])
plot.savefig(figurefolderName+'/time_resp_4th.png')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '-', 'b', 1.5, 1.0, title='Frequency response of plant')
plot.savefig(figurefolderName+'/freq_P.png')

# ZPETC
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd*Czpetc_frd*lead_frd, '-', 'b', 1.5, 1.0, title='Frequency response of P[z] * Czpetc[z] * z^Nzpetc')
plot.savefig(figurefolderName+'/freq_ZPETC.png')

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
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of open loop transfer function')
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of sensitivity function')
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of complementary sensitivity function')
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')
