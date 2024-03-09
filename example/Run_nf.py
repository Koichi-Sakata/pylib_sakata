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
figurefolderName = 'figure_nf'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Ts = 1/2000
dataNum = 10000
freqrange = [50, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# Design notch filters
freqNF = [300]
zetaNF = [0.2]
depthNF = [0.0]
NFs = ctrl.nf(freqNF, zetaNF, depthNF)
NFz_tustin = ctrl.nf(freqNF, zetaNF, depthNF, Ts, 'tustin')
NFz_matched = ctrl.nf(freqNF, zetaNF, depthNF, Ts, 'matched')
NFs_frd = ctrl.sys2frd(NFs[0], freq)
NFz_tustin_frd = ctrl.sys2frd(NFz_tustin[0], freq)
NFz_matched_frd = ctrl.sys2frd(NFz_matched[0], freq)
print('Notch filters were designed.')

freqNF = [300, 300, 300, 300]
zetaNF = [0.2, 0.2, 0.2, 0.2]
depthNF = [0.0, 0.02, 0.1, 0.2]
NFz = ctrl.nf(freqNF, zetaNF, depthNF, Ts)
NFz_d_frd = []
for i in range(len(NFz)):
    NFz_d_frd.append(ctrl.sys2frd(NFz[i], freq))
print('Notch filters were designed.')

freqNF = [300, 300, 300, 300]
zetaNF = [0.01, 0.1, 0.2, 0.7]
depthNF = [0.0, 0.0, 0.0, 0.0]
NFz = ctrl.nf(freqNF, zetaNF, depthNF, Ts)
NFz_zeta_frd = []
for i in range(len(NFz)):
    NFz_zeta_frd.append(ctrl.sys2frd(NFz[i], freq))
print('Notch filters were designed.')


print('Frequency response analysis is running...')
# Notch filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, NFs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of notch filter')
plot.savefig(figurefolderName+'/freq_NF.png')

# Notch filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, NFz_d_frd[0], '-', 'b', 1.5, 1.0, title='Frequency response of notch filter')
plot.plot_tffrd(ax_mag, ax_phase, NFz_d_frd[1], '-', 'g', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, NFz_d_frd[2], '-', 'r', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, NFz_d_frd[3], '-', 'm', 1.5, 1.0, freqrange, [-50, 10], legend=['d: 0.0', 'd: 0.02', 'd: 0.1', 'd: 0.2'])
plot.savefig(figurefolderName+'/freq_NF_d.png')

# Notch filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, NFz_zeta_frd[0], '-', 'b', 1.5, 1.0, title='Frequency response of notch filter')
plot.plot_tffrd(ax_mag, ax_phase, NFz_zeta_frd[1], '-', 'g', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, NFz_zeta_frd[2], '-', 'r', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, NFz_zeta_frd[3], '-', 'm', 1.5, 1.0, freqrange, [-50, 10], legend=['zeta: 0.01', 'zeta: 0.1', 'zeta: 0.2', 'zeta: 0.7'])
plot.savefig(figurefolderName+'/freq_NF_zeta.png')

# Notch filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, NFs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of notch filter')
plot.plot_tffrd(ax_mag, ax_phase, NFz_tustin_frd, '-', 'g', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, NFz_matched_frd, '--', 'r', 1.5, 1.0, legend=['Continuous', 'Discrete: Tustin', 'Discrete: Matched'])
plot.savefig(figurefolderName+'/freq_NF_discrete.png')

plot.showfig()
print('Finished.')
