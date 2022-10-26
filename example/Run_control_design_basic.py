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
Pmechs = ctrl.tf([1.0], [M, C, K])
Pmechz = ctrl.c2d(Pmechs, Ts, method='zoh')
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

print('Frequency response analysis is running...')
G_PD_frd = Pnz_frd * Cz_PD_frd
S_PD_frd = 1/(1 + G_PD_frd)
T_PD_frd = 1 - S_PD_frd

G_PID_frd = Pnz_frd * Cz_PID_frd
S_PID_frd = 1/(1 + G_PID_frd)
T_PID_frd = 1 - S_PID_frd

print('Creating parameter set Cpp and header files...')
axis_num = 6
Cz_PID_axes = np.array([ctrl.tf([1.0], [1.0], Ts) for i in range(axis_num)])
Cz_PD_axes = np.array([ctrl.tf([1.0], [1.0], Ts) for i in range(axis_num)])

for i in range(axis_num):
    Cz_PID_axes[i] = Cz_PID
    Cz_PD_axes[i] = Cz_PD

path = 'src'
ctrl.makeprmset(path)
ctrl.defprmset(Cz_PID_axes, 'gstPIDInf['+str(axis_num)+']', path)
ctrl.defprmset(Cz_PD_axes, 'gstPDInf['+str(axis_num)+']', path)

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

print('Finished.')
