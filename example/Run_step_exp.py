# Copyright (c) 2021 Koichi Sakata


from pylib_sakata import init as init
# uncomment the follows when the file is executed in a Python console.
# init.close_all()
# init.clear_all()

import os
import shutil
import numpy as np
from scipy import signal
from control import matlab
from pylib_sakata import ctrl
from pylib_sakata import plot
from pylib_sakata import fft
from pylib_sakata import meas

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_step_exp'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
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
K = 0.0
Pmechs = ctrl.tf([1], [M, C, K])
numDelay, denDelay = matlab.pade(Ts*4, n=4)
Ds = ctrl.tf(numDelay, denDelay)
Dz = z**-3
Pns = Pmechs * Ds
Pnz = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PID controller
freq1 = 30.0
zeta1 = 1
freq2 = 20.0
zeta2 = 0.7
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

print('Getting measurement data...')
# SysId
Pmeas_frd, coh = meas.measdata2frd('data/freq_resp.csv', 'ServoOutN[0]', 'ActPosUm[0]', 'FlagInject', freq, 1., 1.e-6, 8, 0.8)

# Time response
measdata = meas.getcsvdata('data/time_resp.csv')
time = measdata.time
RefPosUm = measdata.value[meas.getdataindex(measdata, 'RefPosUm[0]')]
ErrPosUm = measdata.value[meas.getdataindex(measdata, 'ErrPosUm[0]')]
ServoOutN = measdata.value[meas.getdataindex(measdata, 'ServoOutN[0]')]
# FFT
freq_fft, ErrPosUm_fft = fft.fft(ErrPosUm[8000:72000], Ts)

print('Frequency response analysis is running...')
# Measurement
Gn_frd = Pmeas_frd * Cz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, time, RefPosUm*1.0e-3, '-', 'b', 1.5, 1.0, ylabel='Ref Pos [mm]', title='Time response')
plot.plot_xy(ax2, time, ErrPosUm, '-', 'b', 1.5, 1.0, ylabel='Error Pos [um]')
plot.plot_xy(ax3, time, ServoOutN, '-', 'b', 1.5, 1.0, xlabel='Time [s]', ylabel='ServoOut [N]')
plot.savefig(figurefolderName+'/time_resp.png')

# FFT
fig = plot.makefig()
ax1 = fig.add_subplot(111)
plot.plot_xy(ax1, freq_fft, ErrPosUm_fft, '-', 'b', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 1.6], xlabel='Frequency [Hz]', ylabel='Error Pos [um]', title='Power spectrum density')
plot.savefig(figurefolderName+'/time_fft.png')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(311)
ax_phase = fig.add_subplot(312)
ax_coh = fig.add_subplot(313)
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, '-', 'm', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_P.png')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.savefig(figurefolderName+'/freq_C.png')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of open loop transfer function')
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, '-', 'b', 1.5, 1.0, freqrange, [-60, 10], title='Frequency response of sensitivity function')
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, '-', 'b', 1.5, 1.0, freqrange, [-60, 10], title='Frequency response of complementary sensitivity function')
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn_frd, '-', 'b', 1.5, 1.0, title='Nyquist diagram')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')