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
figurefolderName = 'figure_exercise_12'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 100000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
print('Common parameters were set.')

# Plant model
M = 2.0
C = 10.0
K = 0
Pmechs = ctrl.tf([1], [M, C, K])
omegaDelay = 2*np.pi*500.0
Ds = ctrl.tf([omegaDelay], [1, omegaDelay])**2
Pns = Pmechs * Ds
Pns_frd = ctrl.sys2frd(Pns, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 10
zeta1 = 1.0
freq2 = 10.0
zeta2 = 1.0
Cs = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K)
Cs_frd = ctrl.sys2frd(Cs, freq)
print('PID controller was designed.')

# Design resonant filters
freqRF = [2, 3, 5, 10, 20, 30, 50, 100]
zetaRF = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
depthRF = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

RFs = ctrl.rfopt(freqRF, zetaRF, depthRF, ctrl.feedback(Pns, Cs, sys='T'))
RFs_frd = 0.0
for i in range(len(RFs)):
    RFs_frd += ctrl.sys2frd(RFs[i], freq)
print('Resonant filters were designed.')

print('Frequency response analysis is running...')
# Model
Gs_frd = Pns_frd * Cs_frd
Ss_frd = 1/(1 + Gs_frd)
Ts_frd = 1 - Ss_frd
# Model with resonant filters
Gs_rf_frd = Pns_frd * Cs_frd * (1.0+RFs_frd)
Ss_rf_frd = 1/(1 + Gs_rf_frd)
Ts_rf_frd = 1 - Ss_rf_frd

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pns_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of plant')
plot.savefig(figurefolderName+'/freq_P.png')

# Resonant filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, RFs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of resonant filters')
plot.savefig(figurefolderName+'/freq_RF.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd * (1.0+RFs_frd), '-', 'r', 1.5, 1.0, freqrange, legend=['w/o RF', 'with RF'])
plot.savefig(figurefolderName+'/freq_C.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Ss_frd, '-', 'b', 1.5, 1.0, freqrange, title='Bode diagram')
plot.plot_tffrd(ax_mag, ax_phase, Ts_frd, '-', 'r', 1.5, 1.0, freqrange, magrange=[-50, 10], legend=['S', 'T'])
plot.savefig(figurefolderName+'/freq_ST.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Ss_frd, '-', 'b', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Ss_rf_frd, '-', 'r', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Ss_frd*0.1, '--', 'k', 0.5, 1.0, freqrange, [-60, 10], legend=['w/o RF', 'with RF'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Ts_frd, '-', 'b', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Ts_rf_frd, '-', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['w/o RF', 'with RF'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gs_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gs_rf_frd, '-', 'r', 1.5, 1.0, [-20, 10], [-15, 15], legend=['w/o RF', 'with RF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gs_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gs_rf_frd, '-', 'r', 1.5, 1.0, legend=['w/o RF', 'with RF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist_zoom.png')

plot.showfig()
print('Finished.')
