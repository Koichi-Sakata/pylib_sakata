# Copyright (c) 2021 Koichi Sakata


import warnings
warnings.filterwarnings('ignore')
from pylib_sakata import init as init
init.close_all()
# uncomment the follows when the file is NOT executed in a Python console.
# init.clear_all()

import os
import shutil
import numpy as np
from pylib_sakata import ctrl
from pylib_sakata import plot

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_derivative'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 10000
freqrange = [1, 10000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
print('Common parameters were set.')

# Plant model
tau = 1/(2.0 * np.pi * 1000)
Exact_differential  = ctrl.tf([1, 0], [1])
Quasi_derivative = ctrl.tf([1, 0], [tau, 1])
Exact_differential_frd = ctrl.sys2frd(Exact_differential, freq)
Quasi_derivative_frd = ctrl.sys2frd(Quasi_derivative, freq)
print('Plant model was set.')

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Exact_differential_frd, '-', 'b', 1.5, 1.0, phasebase=180, legend='Exact differential', title='Frequency response of derivative')
plot.plot_tffrd(ax_mag, ax_phase, Quasi_derivative_frd, '--', 'm', 1.5, 1.0, magrange=[0, 100], phasebase=180, legend='Quasi-derivative')
plot.savefig(figurefolderName+'/freq_derivative.png')

plot.showfig()
print('Finished.')
