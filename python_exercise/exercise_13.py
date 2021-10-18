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
figurefolderName = 'figure_exercise_13'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
dataNum = 10000
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

# Design DOB
freq_dob = 8.0
zeta_dob = 0.7
omega_dob = 2*np.pi*freq_dob
DOBu = omega_dob**2 * ctrl.tf([1], [1, 2*zeta_dob*omega_dob, omega_dob**2])
DOBy = omega_dob**2 * ctrl.tf([M, C, K], [1, 2*zeta_dob*omega_dob, omega_dob**2])
DOBu_frd = ctrl.sys2frd(DOBu, freq)
DOBy_frd = ctrl.sys2frd(DOBy, freq)
print('DOB was designed.')

print('Frequency response analysis is running...')
# w/o DOB
Gn_frd = Pns_frd * Cs_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd

# with DOB
Gdob_frd = Pns_frd * (Cs_frd + DOBy_frd)/(1.0 - DOBu_frd)
Sdob_frd = 1/(1 + Gdob_frd)
Tdob_frd = 1 - Sdob_frd

print('Plotting figures...')
# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Pns_frd, '-', 'b', 1.5, 1.0, title='Frequency response of plant')
plot.savefig(figurefolderName+'/freq_P.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cs_frd, '-', 'b', 1.5, 1.0, title='Frequency response of FB controller')
plot.plot_tffrd(ax_mag, ax_phase, (Cs_frd + DOBy_frd)/(1.0 - DOBu_frd), '-', 'r', 1.5, 1.0, freqrange, legend=['PID', 'PID + DOB'])
plot.savefig(figurefolderName+'/freq_C.png')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, '-', 'b', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, Gdob_frd, '-', 'r', 1.5, 1.0, freqrange, legend=['w/o DOB', 'with DOB'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, '-', 'b', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Sdob_frd, '-', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['w/o DOB', 'with DOB'])
plot.savefig(figurefolderName+'/freq_S.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn_frd, '-', 'b', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gdob_frd, '-', 'r', 1.5, 1.0, legend=['w/o DOB', 'with DOB'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')