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
figurefolderName = 'figure_2mass_pl'
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
M1 = 1.0
M2 = 1.0
M = M1 + M2
C = 10.0
K = 0.0
Creso = 10.0
Kreso = 50000.0
k1 = M2/(M1 * (M1 + M2))
k2 = -1.0/(M1 + M2)
omegaPreso = np.sqrt(Kreso * (M1 + M2)/(M1 * M2))
zetaPreso = 0.5 * Creso*np.sqrt((M1 + M2)/(Kreso * M1 * M2))
Pmechs1 = ctrl.tf([1], [M, C, K]) + k1 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Pmechs2 = ctrl.tf([1], [M, C, K]) + k2 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
numDelay, denDelay = matlab.pade(Ts*4, n=4)
Ds = ctrl.tf(numDelay, denDelay)
Dz = z**-4
Pns1 = Pmechs1 * Ds
Pns2 = Pmechs2 * Ds
Pnz1 = ctrl.c2d(Pmechs1, Ts, method='zoh') * Dz
Pnz2 = ctrl.c2d(Pmechs2, Ts, method='zoh') * Dz
Pnz1_frd = ctrl.sys2frd(Pnz1, freq)
Pnz2_frd = ctrl.sys2frd(Pnz2, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 10.0
zeta1 = 1.0
freq2 = 10.0
zeta2 = 1.0
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

# Design phase lead filter
zeta1 = 0.7
freq1 = 40
zeta2 = 0.7
freq2 = 60
PLz1 = ctrl.pl2nd(freq1, zeta1, freq2, zeta2, Ts)
PLz1_frd = ctrl.sys2frd(PLz1, freq)
PLz2 = ctrl.pl2nd(freq2, zeta2, freq1, zeta1, Ts)
PLz2_frd = ctrl.sys2frd(PLz2, freq)
print('Phase lead filters were desinged.')

print('Frequency respose alanysis is running...')
# Motor side
Gn1_frd = Pnz1_frd * Cz_frd
Sn1_frd = 1/(1 + Gn1_frd)
Tn1_frd = 1 - Sn1_frd

Gn1_pl_frd = Pnz1_frd * Cz_frd * PLz1_frd
Sn1_pl_frd = 1/(1 + Gn1_pl_frd)
Tn1_pl_frd = 1 - Sn1_pl_frd

# Load side
Gn2_frd = Pnz2_frd * Cz_frd
Sn2_frd = 1/(1 + Gn2_frd)
Tn2_frd = 1 - Sn2_frd

Gn2_pl_frd = Pnz2_frd * Cz_frd * PLz2_frd
Sn2_pl_frd = 1/(1 + Gn2_pl_frd)
Tn2_pl_frd = 1 - Sn2_pl_frd

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pnz1_frd, '-', 'b', 1.5, 1.0, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz2_frd, '-', 'r', 1.5, 1.0, freqrange, legend=['Motor side', 'Load side'])
plot.savefig(figurefolderName+'/freq_P.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.savefig(figurefolderName+'/freq_C.png')

# Notch and phase lead filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, PLz1_frd, '-', 'b', 1.5, 1.0, title='Frequency response of filters')
plot.plot_tffrd(ax_mag, ax_phase, PLz2_frd, '-', 'r', 1.5, 1.0, freqrange, [-10, 10], legend=['PL for motor side', 'PL for load side'])
plot.savefig(figurefolderName+'/freq_PL.png')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Gn1_frd, '-', 'b', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, Gn2_frd, '-', 'r', 1.5, 1.0, freqrange, legend=['Motor side', 'Load side'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn1_frd, '-', 'b', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Sn2_frd, '-', 'r', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Sn1_pl_frd, '-', 'c', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Sn2_pl_frd, '-', 'm', 1.5, 1.0, freqrange, [-60, 20], legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn1_frd, '-', 'b', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Tn2_frd, '-', 'r', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Tn1_pl_frd, '-', 'c', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Tn2_pl_frd, '-', 'm', 1.5, 1.0, freqrange, [-60, 20], legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn1_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn2_frd, '-', 'r', 1.5, 1.0)
plot.plot_nyquist(ax, Gn1_pl_frd, '-', 'c', 1.5, 1.0)
plot.plot_nyquist(ax, Gn2_pl_frd, '-', 'm', 1.5, 1.0, legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn1_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn2_frd, '-', 'r', 1.5, 1.0)
plot.plot_nyquist(ax, Gn1_pl_frd, '-', 'c', 1.5, 1.0)
plot.plot_nyquist(ax, Gn2_pl_frd, '-', 'm', 1.5, 1.0, xrange=[-5, 5], yrange=[-5, 5], legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist_.png')

print('Finished.')
