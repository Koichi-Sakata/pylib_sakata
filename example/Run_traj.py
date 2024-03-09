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
from pylib_sakata import traj
from pylib_sakata import fft
from pylib_sakata import plot

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_traj'
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

print('Time response analysis is running...')
posStep = 1
velMax = 1
accAve = 2

traj0 = traj.traj3rd(0, posStep, velMax, accAve, Ts)
traj1 = traj.traj4th(0, posStep, velMax, accAve, Ts)
traj2 = traj.traj4th2(0, posStep, velMax, accAve, Ts)
traj3 = traj.trajSinStep(0, posStep, velMax, accAve, Ts)
traj4 = traj.trajSinStep2(0, posStep, velMax, accAve, Ts)
traj5 = traj.trajSinStep3(0, posStep, velMax, accAve, Ts)

freq_fft0, jerk0_fft = fft.fft_ave(traj0.jerk, Ts)
freq_fft1, jerk1_fft = fft.fft_ave(traj1.jerk, Ts)
freq_fft2, jerk2_fft = fft.fft_ave(traj2.jerk, Ts)
freq_fft3, jerk3_fft = fft.fft_ave(traj3.jerk, Ts)
freq_fft4, jerk4_fft = fft.fft_ave(traj4.jerk, Ts)
freq_fft5, jerk5_fft = fft.fft_ave(traj5.jerk, Ts)

print('Plotting figures...')
# Time response
fig = plot.makefig(dpi=400, figsize=(12, 12), popwin=True)
ax1 = fig.add_subplot(511)
ax2 = fig.add_subplot(512)
ax3 = fig.add_subplot(513)
ax4 = fig.add_subplot(514)
ax5 = fig.add_subplot(515)
plot.plot_xy(ax1, traj0.time, traj0.pos, '--', 'k', 1.5, 1.0)
plot.plot_xy(ax1, traj1.time, traj1.pos, '--', 'b', 1.5, 1.0)
plot.plot_xy(ax1, traj2.time, traj2.pos, '--', 'c', 1.5, 1.0)
plot.plot_xy(ax1, traj3.time, traj3.pos, '-', 'r', 1.0, 1.0)
plot.plot_xy(ax1, traj4.time, traj3.pos, '-', 'm', 1.0, 1.0)
plot.plot_xy(ax1, traj5.time, traj5.pos, '-', 'g', 1.0, 1.0, ylabel='Pos [m]', title='Time response', legend=['3rd', '4th', '4th v2', 'sin', 'sin v2', 'sin v3'])
plot.plot_xy(ax2, traj0.time, traj0.vel, '--', 'k', 1.5, 1.0)
plot.plot_xy(ax2, traj1.time, traj1.vel, '--', 'b', 1.5, 1.0)
plot.plot_xy(ax2, traj2.time, traj2.vel, '--', 'c', 1.5, 1.0)
plot.plot_xy(ax2, traj3.time, traj3.vel, '-', 'r', 1.0, 1.0)
plot.plot_xy(ax2, traj4.time, traj4.vel, '-', 'm', 1.0, 1.0)
plot.plot_xy(ax2, traj5.time, traj5.vel, '-', 'g', 1.0, 1.0, ylabel='Vel [m/s]')
plot.plot_xy(ax3, traj0.time, traj0.acc, '--', 'k', 1.5, 1.0)
plot.plot_xy(ax3, traj1.time, traj1.acc, '--', 'b', 1.5, 1.0)
plot.plot_xy(ax3, traj2.time, traj2.acc, '--', 'c', 1.5, 1.0)
plot.plot_xy(ax3, traj3.time, traj3.acc, '-', 'r', 1.0, 1.0)
plot.plot_xy(ax3, traj4.time, traj4.acc, '-', 'm', 1.0, 1.0)
plot.plot_xy(ax3, traj5.time, traj5.acc, '-', 'g', 1.0, 1.0, ylabel='Acc [m/s2]')
plot.plot_xy(ax4, traj0.time, traj0.jerk, '--', 'k', 1.5, 1.0)
plot.plot_xy(ax4, traj1.time, traj1.jerk, '--', 'b', 1.5, 1.0)
plot.plot_xy(ax4, traj2.time, traj2.jerk, '--', 'c', 1.5, 1.0)
plot.plot_xy(ax4, traj3.time, traj3.jerk, '-', 'r', 1.0, 1.0)
plot.plot_xy(ax4, traj4.time, traj4.jerk, '-', 'm', 1.0, 1.0)
plot.plot_xy(ax4, traj5.time, traj5.jerk, '-', 'g', 1.0, 1.0, xlabel='Time [s]', ylabel='Jerk [m/s3]')
plot.plot_xy(ax5, traj1.time, traj1.snap, '--', 'b', 1.5, 1.0)
plot.plot_xy(ax5, traj2.time, traj2.snap, '--', 'c', 1.5, 1.0)
plot.plot_xy(ax5, traj3.time, traj3.snap, '-', 'r', 1.0, 1.0)
plot.plot_xy(ax5, traj4.time, traj4.snap, '-', 'm', 1.0, 1.0)
plot.plot_xy(ax5, traj5.time, traj5.snap, '-', 'g', 1.0, 1.0, xlabel='Time [s]', ylabel='Snap [m/s4]')
plot.savefig(figurefolderName+'/time_traj.png')

# FFT
fig = plot.makefig(dpi=200)
ax1 = fig.add_subplot(111)
plot.plot_xy(ax1, freq_fft0, jerk0_fft, '--', 'k', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft1, jerk1_fft, '--', 'b', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft2, jerk2_fft, '--', 'c', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft3, jerk3_fft, '-', 'r', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft3, jerk4_fft, '-', 'm', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft4, jerk5_fft, '-', 'g', 1.5, 1.0, xscale='log', yscale='log', xrange=[1.0, 100.0], yrange=[0.001, 100], xlabel='Frequency [Hz]', ylabel='Ref Jerk [mm/s3]', legend=['3rd', '4th', '4th v2', 'sin', 'sin v2', 'sin v3'])
plot.savefig(figurefolderName+'/time_fft.png')

plot.showfig()
print('Finished.')
