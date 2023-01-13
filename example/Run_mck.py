# Copyright (c) 2021 Koichi Sakata


from pylib_sakata import init as init
# uncomment the follows when the file is executed in a Python console.
# init.close_all()
# init.clear_all()

import os
import shutil
import numpy as np
from pylib_sakata import ctrl
from pylib_sakata import plot

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_mck'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
print('Common parameters were set.')

# Plant model
M = 2.0
C = 10.0
K = M * (2.0 * np.pi * 10.0)**2
Pns = ctrl.tf([1], [M, C, K])
Pns_frd = ctrl.sys2frd(Pns, freq)
print('Plant model was set.')

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pns_frd, '-', 'b', 1.5, 1.0, title='Frequency response of plant')
plot.savefig(figurefolderName+'/freq_P.png')

print('Finished.')
