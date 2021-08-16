# Copyright (c) 2021 Koichi Sakata


from pylib_sakata import init as init
# init.close_all()
# init.clear_all()

import os
import shutil
import numpy as np
from control import matlab
from pylib_sakata import plot
from pylib_sakata import ctrl

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_pf'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Ts = 1/4000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0],[1])
z = ctrl.tf([1, 0],[1], Ts)
print('Common parameters were set.')

# Plant model
M = 2.0
zetaP = 0.7
omegaP = 2*np.pi*10
C = M*2*zetaP*omegaP
K = M*omegaP**2
C = 0
K = 0
Pmechs = ctrl.tf([1],[M, C, K])
numDelay, denDelay = matlab.pade(Ts*4,n=4)
omegaD = 2.0*np.pi*500
#Ds = ctrl.tf([omegaD],[1, omegaD])**2
Ds = ctrl.tf(numDelay,denDelay)
Dz = z**-4
Pns = Pmechs * Ds
Pnz = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
Pns_frd = ctrl.sys2frd(Pns, freq)
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PID controller
fd = 200
fi = 30
freq1 = 10
zeta1 = 1.0
freq2 = 10.0
zeta2 = 1.0
Cs = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K)
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cs_frd = ctrl.sys2frd(Cs, freq)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

# Design peak filters

freqPF = [2, 3, 5, 10, 20, 30, 50, 100]
zetaPF = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
depthPF = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

PFs = ctrl.pfopt(freqPF, zetaPF, depthPF, ctrl.feedback(Pnz, Cz, sys='T'))
PFz = ctrl.pfopt(freqPF, zetaPF, depthPF, ctrl.feedback(Pnz, Cz, sys='T'), Ts)
PFs_frd = 0.0
PFz_frd = 0.0
for i in range(len(PFz)):
    PFs_frd += ctrl.sys2frd(PFs[i], freq)
    PFz_frd += ctrl.sys2frd(PFz[i], freq)
print('Peak filters were desinged.')

# Design notch filters
freqNF = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
zetaNF = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
depthNF = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
NFs = ctrl.nf(freqNF, zetaNF, depthNF)
NFz = ctrl.nf(freqNF, zetaNF, depthNF, Ts)
NFs_frd = 1.0
NFz_frd = 1.0
for i in range(len(NFz)):
    NFs_frd *= ctrl.sys2frd(NFs[i], freq)
    NFz_frd *= ctrl.sys2frd(NFz[i], freq)
print('Notch filters were desinged.')

print('Frequency respose alanysis is running...')
# Model
Gn_frd = Pnz_frd * Cz_frd * NFz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd
# Model with peak filters
G_frd = Pnz_frd * Cz_frd * NFz_frd * (1.0+PFz_frd)
S_frd = 1/(1 + G_frd)
T_frd = 1 - S_frd

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, freq, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pns_frd, freq, '--', 'r', 1.5, 1.0, freqrange, legend=['Continuous','Discrete'])
plot.savefig(figurefolderName+'/freq_P.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, freq, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, freq, '--', 'r', 1.5, 1.0, freqrange, legend=['Continuous','Discrete'])
plot.savefig(figurefolderName+'/freq_C.png')

# Peak filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, PFs_frd, freq, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of peak filters')
plot.plot_tffrd(ax_mag, ax_phase, PFz_frd, freq, '--', 'r', 1.5, 1.0, freqrange, legend=['Continuous','Discrete'])
plot.savefig(figurefolderName+'/freq_PF.png')

# Peak filters
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, NFs_frd, freq, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of notch filters')
plot.plot_tffrd(ax_mag, ax_phase, NFz_frd, freq, '--', 'r', 1.5, 1.0, freqrange, legend=['Continuous','Discrete'])
plot.savefig(figurefolderName+'/freq_NF.png')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, freq, '-', 'b', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, G_frd, freq, '-', 'm', 1.5, 1.0, freqrange, legend=['w/o PF','with PF'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, freq, '-', 'b', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, S_frd, freq, '-', 'm', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd*0.1, freq, '--', 'k', 0.5, 1.0, freqrange, [-60, 10], legend=['w/o PF','with PF'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, freq, '-', 'b', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, T_frd, freq, '-', 'm', 1.5, 1.0, freqrange, [-60, 10], legend=['w/o PF','with PF'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, G_frd, '-', 'm', 1.5, 1.0, [-20, 10], [-15, 15], legend=['w/o PF','with PF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')