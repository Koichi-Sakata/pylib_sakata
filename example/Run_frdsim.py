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
from pylib_sakata import plot
from pylib_sakata import fft

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_frdsim'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Ts = 1/1000
dataNum = 10001
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# System model
freqG = 1.0
omegaG = 2 * np.pi * freqG
Gs = omegaG/(s + omegaG)
Gfrd = ctrl.sys2frd(Gs, freq)

t = np.arange(dataNum) * Ts
x = np.sin(2.0 * np.pi * t) + np.sin(2.0 * np.pi * 10.0 * t)

y_lsim, tout, xout = matlab.lsim(ctrl.tf2ss(Gs), x, t)
y_fft = fft.frdsim(Gfrd, x, Ts)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, t, x, '-', 'b', 0.5, 1.0, [0, t[-1]], ylabel='In', title='Time response')
plot.plot_xy(ax2, t, y_lsim, '-', 'b', 0.5, 1.0, [0, t[-1]], legend='lsim')
plot.plot_xy(ax2, t, y_fft, '--', 'r', 0.5, 1.0, [0, t[-1]], xlabel='Time [s]', ylabel='Out', legend='frd', loc='upper right')
plot.savefig(figurefolderName+'/time_frdsim.png')

print('Finished.')
