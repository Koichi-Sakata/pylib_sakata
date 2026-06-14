# Copyright (c) 2026 Koichi Sakata


import warnings
warnings.filterwarnings('ignore')
from pylib_sakata import init as init
init.close_all()
# uncomment the follows when the file is NOT executed in a Python console.
# init.clear_all()

import os
import shutil
import numpy as np
from control import matlab
from pylib_sakata import ctrl
from pylib_sakata import traj
from pylib_sakata import plot
from scipy import signal

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_ilc'
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
K = 0
Pns = ctrl.tf([1], [M, C, K])
Ndelay = 4
Dz = z**-Ndelay
Pnz = ctrl.c2d(Pns, Ts, method='zoh') * Dz
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 20.0
zeta1 = 1.0
freq2 = 20.0
zeta2 = 1.0
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

print('Frequency response analysis is running...')
Gn_frd = Pnz_frd * Cz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd

# Design Q-filter
freqQ = 100
zetaQ = 0.7
Qz = ctrl.lpf2nd(freqQ, zetaQ, Ts)
numQz = Qz.num[0][0]
denQz = Qz.den[0][0]
Qzflip = ctrl.tf(np.flip(numQz), np.flip(denQz), Ts)
Q = Qz * Qzflip
Q_frd = ctrl.sys2frd(Q, freq)
print('Q-filter was designed.')

# Design learning-filter
Snz = ctrl.feedback(Pnz, Cz, sys='S')
SPnz = ctrl.feedback(Pnz, Cz, sys='SP')
SPnz_frd = ctrl.sys2frd(SPnz, freq)
Pinvz, N_L = ctrl.zpetc(Pnz, Ts)
Lz1 = Pinvz
Lz2 = Cz
Lz = Lz1 + Lz2
Lz1_frd = ctrl.sys2frd(Lz1, freq)
Lz2_frd = ctrl.sys2frd(Lz2, freq)
L_frd = Lz1_frd + Lz2_frd
print('Learning-filter was designed.')

print('Time response analysis is running...')
def ilc(Sz, SPz, Lz, Qz, r, e, f, t):
    Le1, tout, xout = matlab.lsim(ctrl.tf2ss(Lz1), e, t)
    Le2, tout, xout = matlab.lsim(ctrl.tf2ss(Lz2), e, t)
    Le = Le1 + Le2
    f_new = signal.filtfilt(Qz.num[0][0], Qz.den[0][0], Le + f)
    e_new, u, y = ctrl.trdsim(r, t, Pnz, Cz, uff=f_new, Ndelay=Ndelay)
    return e_new, f_new

posStep = 0.5
velMax = 0.5
accAve = 1.0
traj = traj.traj4th(0, posStep, velMax, accAve, Ts, 0.5)
r = traj.pos
t = traj.time
# 1st trial
n = 0
e, u, y = ctrl.trdsim(r, t, Pnz, Cz, Ndelay=Ndelay)
f = np.zeros(len(t))

col = ['b', 'g', 'r', 'c', 'm', 'y']
legend = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6']
fig = plot.makefig(dpi=150, figsize=(6,6))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
plot.plot_xy(ax1, traj.time, traj.pos * 1.0e3, '-', 'b', 1.5, 1.0, ylabel='Ref Pos [mm]', title='Time response')
plot.plot_xy(ax2, traj.time, traj.vel * 1.0e3, '-', 'b', 1.5, 1.0, ylabel='Ref Vel [mm/s]')
plot.plot_xy(ax3, t, e * 1.0e6, '-', col[np.mod(n, len(col))], 1.5, 1.0, ylabel='Error Pos [$\mu$m]', legend=legend[np.mod(n, len(legend))], loc='upper left', bbox_to_anchor=(1, 1))
plot.plot_xy(ax4, t, f, '-', col[np.mod(n, len(col))], 1.5, 1.0, xlabel='Time [s]', ylabel='ILC [N]')

# Nth trial
n_max = 5
for n in range(1, n_max):
    # Update ILC
    e, f = ilc(Snz, SPnz, Lz, Qz, r, e, f, t)
    # Plot
    plot.plot_xy(ax3, t, e*1.0e6, '-', col[np.mod(n, len(col))], 1.5, 1.0, yrange=[-0.1, 0.1], legend=legend[np.mod(n, len(legend))], loc='upper left', bbox_to_anchor=(1, 1))
    plot.plot_xy(ax4, t, f, '-', col[np.mod(n, len(col))], 1.5, 1.0, xlabel='Time [s]')
    if n == n_max-1:
        plot.savefig(figurefolderName+'/time_resp.png')

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '-', 'b', 1.5, 1.0, title='Frequency response of plant')
plot.savefig(figurefolderName+'/freq_P.png')

# Q-filter
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Q_frd, '-', 'b', 1.5, 1.0, phasebase=180, title='Frequency response of Q')
plot.savefig(figurefolderName+'/freq_Q.png')

# Learning-filter
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, L_frd, '-', 'b', 1.5, 1.0, phasebase=180, title='Frequency response of L')
plot.savefig(figurefolderName+'/freq_L.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, '-', 'b', 1.5, 1.0, freqrange, phasebase=180, title='Frequency response of PID controller')
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
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, '-', 'b', 1.5, 1.0, freqrange, phasebase=180, title='Frequency response of complementary sensitivity function')
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

plot.showfig()
print('Finished.')
