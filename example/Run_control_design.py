# Copyright (c) 2022 Koichi Sakata


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
figurefolderName = 'figure_control_design'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
srcpathName = 'src'
if os.path.exists(srcpathName):
    shutil.rmtree(srcpathName)
os.makedirs(srcpathName)
Ts = 1/8000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# Plant model
M = 0.022
C = 1.0
K = 0
Pmechs = ctrl.tf([1], [M, C, K])
numDelay, denDelay = matlab.pade(Ts*4, n=4)
Ds = ctrl.tf(numDelay, denDelay)
Dz = z**-3
Pns = Pmechs * Ds
Pnz = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PD controller
freq1 = 10.0
freq2 = 10.0
zeta2 = 1.0
Cz_PD = ctrl.pd(freq1, freq2, zeta2, M, C, K, Ts)
Cz_PD_frd = ctrl.sys2frd(Cz_PD, freq)
print('PD controller was designed.')

# Design PID controller
freq1 = 30.0
zeta1 = 1.0
freq2 = 20.0
zeta2 = 0.7
Cz_PID = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_PID_frd = ctrl.sys2frd(Cz_PID, freq)
print('PID controller was designed.')

# Design notch filters
freqNF = [2000]
zetaNF = [0.3]
depthNF = [0.01]
NFz = ctrl.nf(freqNF, zetaNF, depthNF, Ts)
NFz_frd = 0.0
for i in range(len(freqNF)):
    NFz_frd += ctrl.sys2frd(NFz[i], freq)
print('Notch filters were designed.')

# Design peak filters
freqPF = [10.0, 20.0, 30.0, 31.0, 50.0, 60.0, 70.0, 90.0, 180.0]
zetaPF = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
depthPF = [0.02, 0.1, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PFz = ctrl.pfopt(freqPF, zetaPF, depthPF, ctrl.feedback(Pnz, Cz_PID, sys='T'), Ts)
PFz_frd = 0.0
for i in range(len(freqPF)):
    PFz_frd += ctrl.sys2frd(PFz[i], freq)
print('Peak filters were designed.')

print('Frequency response analysis is running...')
G_PD_frd = Pnz_frd * Cz_PD_frd
S_PD_frd = 1/(1 + G_PD_frd)
T_PD_frd = 1 - S_PD_frd

G_PID_frd = Pnz_frd * Cz_PID_frd
S_PID_frd = 1/(1 + G_PID_frd)
T_PID_frd = 1 - S_PID_frd

G_PIDwPF_frd = Pnz_frd * Cz_PID_frd * (1.0+PFz_frd)
S_PIDwPF_frd = 1/(1 + G_PIDwPF_frd)
T_PIDwPF_frd = 1 - S_PIDwPF_frd

print('Creating parameter set Cpp and header files...')
axis_num = 6
Cz_PID_axes = np.array([ctrl.tf([1.0], [1.0], Ts) for i in range(axis_num)])
Cz_PD_axes = np.array([ctrl.tf([1.0], [1.0], Ts) for i in range(axis_num)])
NFz_axes = np.array([[ctrl.tf([1.0], [1.0], Ts) for j in range(len(NFz))] for i in range(axis_num)])
PFz_axes = np.array([[ctrl.tf([0.0], [1.0], Ts) for j in range(len(PFz))] for i in range(axis_num)])

for i in range(axis_num):
    Cz_PID_axes[i] = Cz_PID
    Cz_PD_axes[i] = Cz_PD

for i in range(axis_num):
    for j in range(len(NFz)):
        NFz_axes[i][j] = NFz[j]

for i in range(axis_num):
    for j in range(len(PFz)):
        PFz_axes[i][j] = PFz[j]

path = 'src'
ctrl.makeprmset(path)
ctrl.defprmset(Cz_PID_axes, 'gstPIDInf['+str(axis_num)+']', path)
ctrl.defprmset(Cz_PD_axes, 'gstPDInf['+str(axis_num)+']', path)
ctrl.defprmset(NFz_axes, 'gstNFInf['+str(axis_num)+']['+str(len(NFz))+']', path)
ctrl.defprmset(PFz_axes, 'gstPFInf['+str(axis_num)+']['+str(len(PFz))+']', path)

print('Plotting figures...')
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
plot.plot_tffrd(ax_mag, ax_phase, Cz_PD_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.plot_tffrd(ax_mag, ax_phase, Cz_PID_frd, '--', 'r', 1.5, 1.0, freqrange, magrange=[30, 75], legend=['PD', 'PID'])
plot.savefig(figurefolderName+'/freq_C.png')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, G_PD_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, G_PID_frd, '-', 'm', 1.5, 1.0, freqrange)
plot.plot_tffrd(ax_mag, ax_phase, G_PIDwPF_frd, '--', 'r', 1.5, 1.0, freqrange, legend=['PD', 'PID', 'PID with PF'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, S_PD_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, S_PID_frd, '-', 'm', 1.5, 1.0, freqrange)
plot.plot_tffrd(ax_mag, ax_phase, S_PIDwPF_frd, '--', 'r', 1.5, 1.0, freqrange, legend=['PD', 'PID', 'PID with PF'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, T_PD_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, T_PID_frd, '-', 'm', 1.5, 1.0, freqrange)
plot.plot_tffrd(ax_mag, ax_phase, T_PIDwPF_frd, '--', 'r', 1.5, 1.0, freqrange, legend=['PD', 'PID', 'PID with PF'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, G_PD_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, G_PID_frd, '-', 'm', 1.5, 1.0)
plot.plot_nyquist(ax, G_PIDwPF_frd, '--', 'r', 1.5, 1.0, legend=['PD', 'PID', 'PID with PF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')
