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

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_trdsim'
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
M1 = 1.0
M2 = 1.0
M = M1 + M2
C = 10.0
K = 0.0
Creso = 10.0
Kreso = 20000.0
k1 = M2/(M1 * (M1 + M2))
k2 = -1.0/(M1 + M2)
omegaPreso = np.sqrt(Kreso * (M1 + M2)/(M1 * M2))
zetaPreso = 0.5 * Creso*np.sqrt((M1 + M2)/(Kreso * M1 * M2))
Pmechs1 = ctrl.tf([1], [M, C, K]) + k1 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Pmechs2 = ctrl.tf([1], [M, C, K]) + k2 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Ndelay = 4
Dz = z**-Ndelay
Pnz1 = ctrl.c2d(Pmechs1, Ts, method='zoh')
Pnz2 = ctrl.c2d(Pmechs2, Ts, method='zoh')
Pnz1_frd = ctrl.sys2frd(Pnz1 * Dz, freq)
Pnz2_frd = ctrl.sys2frd(Pnz2 * Dz, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 20.0
zeta1 = 0.7
freq3 = 20.0
freq4 = 100.0
freq2 = np.sqrt(freq3 * freq4)
zeta2 = 0.5*(freq3 + freq4)/freq2
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

print('Frequency response analysis is running...')
# Motor side
Gn1_frd = Pnz1_frd * Cz_frd
Sn1_frd = 1/(1 + Gn1_frd)
Tn1_frd = 1 - Sn1_frd

# Load side
Gn2_frd = Pnz2_frd * Cz_frd
Sn2_frd = 1/(1 + Gn2_frd)
Tn2_frd = 1 - Sn2_frd

print('Time response analysis is running...')
posStep = 0.4
velMax = 1
accAve = 10
traj = traj.traj4th(0, posStep, velMax, accAve, Ts, 0.5)
r = traj.pos
t = traj.time
e, u, y = ctrl.trdsim(r, t, Pnz1, Cz, Ndelay=Ndelay)

print('Plotting figures...')
# Time response
fig = plot.makefig(dpi=150, figsize=(6,6))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, t, r, '-', 'b', 1.5, 1.0, [0, max(t)], ylabel='Position [m]', legend='r', title='Time response')
plot.plot_xy(ax1, t, y, '-', 'r', 1.5, 1.0, [0, max(t)], ylabel='Position [m]', legend='y')
plot.plot_xy(ax2, t, e*1e6, '-', 'b', 1.5, 1.0, [0, max(t)], ylabel='Position [$\mu$m]', legend='e')
plot.plot_xy(ax3, t, u, '-', 'b', 1.5, 1.0, [0, max(t)], xlabel='Time [s]', ylabel='Force [N]', legend='u')
plot.savefig(figurefolderName+'/time_resp.png')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pnz1_frd, '-', 'b', 1.5, 1.0, legend='Motor side', title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz2_frd, '-', 'r', 1.5, 1.0, freqrange, legend='Load side')
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
plot.plot_tffrd(ax_mag, ax_phase, Gn1_frd, '-', 'b', 1.5, 1.0, legend='Motor side', title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, Gn2_frd, '-', 'r', 1.5, 1.0, freqrange, legend='Load side')
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn1_frd, '-', 'b', 1.5, 1.0, legend='Motor side', title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Sn2_frd, '-', 'r', 1.5, 1.0, freqrange, [-60, 20], legend='Load side')
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn1_frd, '-', 'b', 1.5, 1.0, phasebase=180, legend='Motor side', title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Tn2_frd, '-', 'r', 1.5, 1.0, freqrange, [-60, 20], phasebase=180, legend='Load side')
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn1_frd, '-', 'b', 1.5, 1.0, xrange=[-20, 10], yrange=[-15, 15], legend='Motor side', title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn2_frd, '-', 'r', 1.5, 1.0, xrange=[-20, 10], yrange=[-15, 15], legend='Load side')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn1_frd, '-', 'b', 1.5, 1.0, legend='Motor side', title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn2_frd, '-', 'r', 1.5, 1.0, legend='Load side')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist_zoom.png')

print('Finished.')
