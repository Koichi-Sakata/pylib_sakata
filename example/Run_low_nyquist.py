# Copyright (c) 2023 Koichi Sakata


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
figurefolderName = 'figure_low_nyquist'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Ts = 1/400
dataNum = 100000
freqrange = [10, 2000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
freqrange2 = [1, 2000]
freq2 = np.logspace(np.log10(freqrange2[0]), np.log10(freqrange2[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# Plant model
M = 2.0
C = 10.0
K = 0.0
Pmechs = ctrl.tf([1], [M, C, K])
numDelay, denDelay = matlab.pade(Ts*4, n=4)
Ds = ctrl.tf(numDelay, denDelay)
Dz = z**-3
Pns = Pmechs * Ds
Pnz = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
Pns_frd = ctrl.sys2frd(Pns, freq)
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PID controller
freq_pid1 = 20
zeta_pid1 = 0.7
freq_pid2 = 20
zeta_pid2 = 0.7
Cs = ctrl.pid(freq_pid1, zeta_pid1, freq_pid2, zeta_pid2, M, C, K)
Cs_frd = ctrl.sys2frd(Cs, freq2)
Cz = ctrl.pid(freq_pid1, zeta_pid1, freq_pid2, zeta_pid2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq2)
print('PID controller was designed.')

# Design notch filters
freqNF = [100]
zetaNF = [0.2]
depthNF = [0.01]
NFs = ctrl.nf(freqNF, zetaNF, depthNF)
NFs_frd = ctrl.sys2frd(NFs[0], freq)
NFz_matched = ctrl.nf(freqNF, zetaNF, depthNF, Ts, 'matched')
NFz_matched_frd = ctrl.sys2frd(NFz_matched[0], freq)
print('Notch filters were designed.')

print('Frequency response analysis is running...')
# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, '-', 'b', 1.5, 1.0, freqrange2, title='Frequency response of PID controller')
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, '--', 'r', 1.5, 1.0, freqrange2, [70, 110], legend=['Continuous', 'Discrete'], loc='lower left')
plot.savefig(figurefolderName+'/freq_C.png')

# Notch filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, NFs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of notch filter')
plot.plot_tffrd(ax_mag, ax_phase, NFz_matched_frd, '--', 'r', 1.5, 1.0, freqrange, [-50, 10], legend=['Continuous', 'Discrete'], loc='lower left')
plot.savefig(figurefolderName+'/freq_NF.png')

print('Finished.')
