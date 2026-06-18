# Copyright (c) 2021 Koichi Sakata


import warnings
warnings.filterwarnings('ignore')
from pylib_sakata import init as init
init.close_all()
# uncomment the follows when the file is NOT executed in a Python console.
# init.clear_all()

import os
import shutil
import numpy as np
import control as ct
from control import matlab
import scipy.signal as signal
from pylib_sakata import ctrl
from pylib_sakata import traj
from pylib_sakata import plot

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_ptc'
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
Np = 2
Ac = np.array([[0, 1], [-K/M, -C/M]])
Bc = np.array([[0], [1/M]])
Cc = np.array([[1, 0]])
Dc = np.array([[0]])
Pns = ct.ss(Ac, Bc, Cc, Dc)
Pnz = ct.c2d(Pns, Ts, method='zoh')
As, Bs, Cs, Ds = ct.ssdata(Pnz)
Ndelay = 4
Dz = z**-Ndelay
Pnz_frd = ctrl.sys2frd(Pnz * Dz, freq)
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
Czpetc, Nzpetc = ctrl.zpetc(ct.ss2tf(Pnz) * Dz, Ts)
Czpetc_frd = ctrl.sys2frd(Czpetc, freq)
lead_frd = ctrl.sys2frd(z**Nzpetc, freq)
print('ZPETC was designed.')

# Design PTC
A = As @ As
B = np.append(As @ Bs, Bs, 1)
C = np.append(Cs, Cs @ As, 0)
D = np.append(np.zeros([1, 2]), np.append(Cs @ Bs, np.array([[0]]), 1), 0)
PTC = signal.StateSpace(np.zeros([2, 2]), np.eye(2), -np.linalg.inv(B) @ A, np.linalg.inv(B), dt=Np*Ts)
print('PTC was designed.')

print('Frequency response analysis is running...')
Gn_frd = Pnz_frd * Cz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd

print('Time response analysis is running...')
# Target trajectory
Snz = ctrl.feedback(Pnz, Cz, sys='S')
traj_4th = traj.traj4th(0, 1.0, 0.5, 1.0, Ts, 0.5)
t_4th = traj_4th.time
r_4th = traj_4th.pos
v_4th = traj_4th.vel
a_4th = traj_4th.acc
n = Nzpetc
r_4th_pre = np.roll(r_4th, -n)
r_4th_pre[-n:] = r_4th[-1]
traj_4th = traj.traj4th(0, 1.0, 0.5, 1.0, Np * Ts, 0.5)
n = 1 + int(Ndelay / Np)
r_4th_ptc = np.roll(traj_4th.pos, -n)
r_4th_ptc[-n:] = traj_4th.pos[-n]
v_4th_ptc = np.roll(traj_4th.vel, -n)
v_4th_ptc[-n:] = traj_4th.vel[-n]
# ZPETC
u_zpetc, tout, xout = matlab.lsim(ctrl.tf2ss(Czpetc), r_4th_pre)
# PTC
xd = np.array([r_4th_ptc, v_4th_ptc]).T
t, u, x = signal.dlsim(PTC, xd)
u_ptc = u.flatten()
u_ptc = u_ptc[:-1]

e1, u1, y1 = ctrl.trdsim(r_4th, t_4th, Pnz, Cz, uff=u_zpetc, Ndelay=Ndelay)
e2, u2, y2 = ctrl.trdsim(r_4th, t_4th, Pnz, Cz, uff=u_ptc, Ndelay=Ndelay)

print('Plotting figures...')
# Time response
fig = plot.makefig(dpi=150, figsize=(6,6))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, t_4th, r_4th, '-', 'b', 1.5, 1.0, ylabel='Ref Pos [m]', title='Time response')
plot.plot_xy(ax2, t_4th, e1*1.0e9, '-', 'b', 1.5, 1.0, legend='with ZPETC')
plot.plot_xy(ax2, t_4th, e2*1.0e9, '-', 'r', 1.5, 1.0, yrange=[-10, 10], xlabel='Time [s]', ylabel='Error Pos [nm]', legend='with PTC')
plot.plot_xy(ax3, t_4th, u1 - u_zpetc, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax3, t_4th, u2 - u_ptc, '-', 'r', 1.5, 1.0, yrange=[-0.01, 0.01], xlabel='Time [s]', ylabel='FB Out [N]')
plot.savefig(figurefolderName+'/time_resp_4th.png')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '-', 'b', 1.5, 1.0, title='Frequency response of plant')
plot.savefig(figurefolderName+'/freq_P.png')

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
