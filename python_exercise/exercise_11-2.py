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
figurefolderName = 'figure_exercise_11-2'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
print('Common parameters were set.')

# Plant model
M1 = 300.0
M2 = 300.0
M = M1 + M2
C = 0.0
K = 0.0
Creso = 3.0e3
Kreso = 4.0e7
k1 = M2/(M1 * (M1 + M2))
k2 = -1.0/(M1 + M2)
omegaPreso = np.sqrt(Kreso * (M1 + M2)/(M1 * M2))
zetaPreso = 0.5 * Creso*np.sqrt((M1 + M2)/(Kreso * M1 * M2))
Pmechs0 = ctrl.tf([1], [M, C, K])
Pmechs1 = ctrl.tf([1], [M, C, K]) + k1 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Pmechs2 = ctrl.tf([1], [M, C, K]) + k2 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
omegaDelay = 2*np.pi*500.0
Ds = ctrl.tf([omegaDelay], [1, omegaDelay])**2
Pns0 = Pmechs0 * Ds
Pns1 = Pmechs1 * Ds
Pns2 = Pmechs2 * Ds
Pns0_frd = ctrl.sys2frd(Pns0, freq)
Pns1_frd = ctrl.sys2frd(Pns1, freq)
Pns2_frd = ctrl.sys2frd(Pns2, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 52.0
zeta1 = 8.7
freq2 = 12.0
zeta2 = 0.7
Cs = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K)
Cs_frd = ctrl.sys2frd(Cs, freq)
print('PID controller was designed.')

# Design notch filters
freqNF = [omegaPreso/(2.0*np.pi)]
zetaNF = [0.4]
depthNF = [0.04]
NFs = ctrl.nf(freqNF, zetaNF, depthNF)
NFs_frd = 1.0
NFs_all = 1.0
for i in range(len(NFs)):
    NFs_frd *= ctrl.sys2frd(NFs[i], freq)
    NFs_all *= NFs[i]
print('Notch filters were desinged.')

print('Frequency respose alanysis is running...')
# Motor side
Gn1_frd = Pns1_frd * Cs_frd
Sn1_frd = 1/(1 + Gn1_frd)
Tn1_frd = 1 - Sn1_frd

Gn1_nf_frd = Pns1_frd * Cs_frd * NFs_frd
Sn1_nf_frd = 1/(1 + Gn1_nf_frd)
Tn1_nf_frd = 1 - Sn1_nf_frd

# Load side
Gn2_frd = Pns2_frd * Cs_frd
Sn2_frd = 1/(1 + Gn2_frd)
Tn2_frd = 1 - Sn2_frd

Gn2_nf_frd = Pns2_frd * Cs_frd * NFs_frd
Sn2_nf_frd = 1/(1 + Gn2_nf_frd)
Tn2_nf_frd = 1 - Sn2_nf_frd

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pns0_frd, '-', 'k', 1.5, 1.0, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pns1_frd, '-', 'b', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Pns2_frd, '-', 'r', 1.5, 1.0, freqrange, legend=['Rigid', 'Motor side', 'Load side'])
plot.savefig(figurefolderName+'/freq_P.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.savefig(figurefolderName+'/freq_C.png')

# Notch filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, NFs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of filters')
plot.savefig(figurefolderName+'/freq_NF.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd * NFs_frd, '-', 'r', 1.5, 1.0, freqrange, legend=['w/o NF', 'with NF'])
plot.savefig(figurefolderName+'/freq_C_nf.png')

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
plot.plot_tffrd(ax_mag, ax_phase, Sn1_nf_frd, '-', 'c', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Sn2_nf_frd, '-', 'm', 1.5, 1.0, freqrange, [-60, 20], legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn1_frd, '-', 'b', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Tn2_frd, '-', 'r', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Tn1_nf_frd, '-', 'c', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Tn2_nf_frd, '-', 'm', 1.5, 1.0, freqrange, [-60, 20], legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn1_frd, '-', 'b', 1.5, 1.0, [-20, 10], [-15, 15], title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn2_frd, '-', 'r', 1.5, 1.0, [-20, 10], [-15, 15])
plot.plot_nyquist(ax, Gn1_nf_frd, '-', 'c', 1.5, 1.0, [-20, 10], [-15, 15])
plot.plot_nyquist(ax, Gn2_nf_frd, '-', 'm', 1.5, 1.0, [-20, 10], [-15, 15], legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn1_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn2_frd, '-', 'r', 1.5, 1.0)
plot.plot_nyquist(ax, Gn1_nf_frd, '-', 'c', 1.5, 1.0)
plot.plot_nyquist(ax, Gn2_nf_frd, '-', 'm', 1.5, 1.0, legend=['Motor side', 'Load side', 'Motor side with NF', 'Load side with NF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist_zoom.png')

print('Finished.')
