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
figurefolderName = 'figure_sysid_2mass_exp'
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
# M1 = 0.035
M1 = 0.0343
# M2 = 0.075
M2 = 0.0757
M = M1 + M2
C = 0.7
K = 0.0
# Creso = 4.0
Creso = 2.5
# Kreso = 55000.0
Kreso = 65460.0
k1 = M2/(M1 * (M1 + M2))
k2 = -1.0/(M1 + M2)
omegaPreso = np.sqrt(Kreso * (M1 + M2)/(M1 * M2))
zetaPreso = 0.5 * Creso*np.sqrt((M1 + M2)/(Kreso * M1 * M2))
Pmechs1 = ctrl.tf([1], [M, C, K]) + k1 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Pmechs2 = ctrl.tf([1], [M, C, K]) + k2 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
numDelay, denDelay = matlab.pade(Ts*4, n=4)
Ds = ctrl.tf(numDelay, denDelay)
Dz = z**-3
Pns1 = Pmechs1 * Ds
Pns2 = Pmechs2 * Ds
Pnz1 = ctrl.c2d(Pmechs1, Ts, method='zoh') * Dz
Pnz2 = ctrl.c2d(Pmechs2, Ts, method='zoh') * Dz
Pnz1_frd = ctrl.sys2frd(Pnz1, freq)
Pnz2_frd = ctrl.sys2frd(Pnz2, freq)
Pnz = Pnz1
Pnz_frd = Pnz1_frd
print('Plant model was set.')

print('Getting measurement data...')
measfileName = 'data/freq_resp_2mass_20230720.csv'
# Frequency response
Pmeas_frd, coh = meas.measdata2frd(measfileName, 'ServoOutN[0]', 'ActPosUm[0]', 'FlagInject', freq, 1., 1.e-6, 8, 0.8)

# Time response
measdata = meas.getdata(measfileName)
time = measdata.time
NoiseOut = measdata.value[meas.getdataindex(measdata, 'NoiseOut[0]')]
ServoOutN = measdata.value[meas.getdataindex(measdata, 'ServoOutN[0]')]
ActPosUm = measdata.value[meas.getdataindex(measdata, 'ActPosUm[0]')]
FlagInject = measdata.value[meas.getdataindex(measdata, 'FlagInject')]

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
plot.plot_xy(ax1, time, NoiseOut, '-', 'b', 0.5, 1.0, ylabel='[Nm]', legend=['Sig'], loc='upper right', title='Time response')
plot.plot_xy(ax2, time, ServoOutN, '-', 'b', 0.5, 1.0, ylabel='[Nm]', legend=['In'], loc='upper right')
plot.plot_xy(ax3, time, ActPosUm, '-', 'b', 0.5, 1.0, ylabel='[um]', legend=['Out'], loc='upper right')
plot.plot_xy(ax4, time, FlagInject, '-', 'b', 0.5, 1.0, xlabel='Time [s]', ylabel='[.]', legend=['Flag'], loc='upper right')
plot.savefig(figurefolderName+'/time_resp.svg')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(311)
ax_phase = fig.add_subplot(312)
ax_coh = fig.add_subplot(313)
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, '-', 'm', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_P.svg')

print('Finished.')