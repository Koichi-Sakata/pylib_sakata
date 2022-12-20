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
from pylib_sakata import fft
from pylib_sakata import meas

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_pf'
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

# Design PD controller
freq1 = 10.0
freq2 = 10.0
zeta2 = 1.0
Cz_PD = ctrl.pd(freq1, freq2, zeta2, M, C, K, Ts)
Cz_PD_frd = ctrl.sys2frd(Cz_PD, freq)
print('PD controller was designed.')

# Design PID controller
freq1 = 30.0
zeta1 = 1
freq2 = 20.0
zeta2 = 0.7
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

print('Frequency response analysis is running...')
# Measurement w/o PF
Gn_frd = Pnz_frd * Cz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd

# Design peak filters
freqPF = [10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 90.0]
zetaPF = [0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001]
depthPF = [0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1]

PFs = ctrl.pfopt(freqPF, zetaPF, depthPF, Tn_frd)
PFz = ctrl.pfopt(freqPF, zetaPF, depthPF, Tn_frd , Ts)
PFs_frd = 0.0
PFz_frd = 0.0
for i in range(len(PFz)):
    PFs_frd += ctrl.sys2frd(PFs[i], freq)
    PFz_frd += ctrl.sys2frd(PFz[i], freq)

freqPF, zetaPF, kPF, phiPF = ctrl.pfoptparam(freqPF, zetaPF, depthPF, ctrl.feedback(Pnz, Cz, sys='T'))
print('Peak filters were designed.')

print('Creating parameter set Cpp and header files...')
axis_num = 6
Cz_PID_axes = np.array([ctrl.tf([1.0], [1.0], Ts) for i in range(axis_num)])
Cz_PD_axes = np.array([ctrl.tf([1.0], [1.0], Ts) for i in range(axis_num)])
PFz_axes = np.array([[ctrl.tf([0.0], [1.0], Ts) for j in range(len(PFz))] for i in range(axis_num)])

for i in range(axis_num):
    Cz_PID_axes[i] = Cz
    Cz_PD_axes[i] = Cz_PD
    for j in range(len(PFz)):
        PFz_axes[i][j] = PFz[j]

path = 'src'
ctrl.makeprmset(path)
ctrl.defprmset(Cz_PID_axes, 'gstPIDInf['+str(axis_num)+']', path)
ctrl.defprmset(Cz_PD_axes, 'gstPDInf['+str(axis_num)+']', path)
ctrl.defprmset(PFz_axes, 'gstPFInf['+str(axis_num)+']['+str(len(PFz))+']', path)

print('Frequency response analysis is running...')
# Measurement with PF
G_frd = Pmeas_frd * Cz_frd * (1.0+PFz_frd)
S_frd = 1/(1 + G_frd)
T_frd = 1 - S_frd

print('Simulating time response with peak filters')
# Time response
time_sim, ErrPosUm_pf_sim = fft.frdsim(S_frd/Sn_frd, ErrPosUm, Ts)
# FFT
freq_fft_sim, ErrPosUm_fft_pf_sim = fft.fft(ErrPosUm_pf_sim[8000:72000], Ts)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(111)
plot.plot_xy(ax1, time, ErrPosUm, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax1, time_sim, ErrPosUm_pf_sim, '--', 'r', 1.5, 1.0, yrange=[-10.0, 10.0], xlabel='Time [s]', ylabel='Error Pos [um]', legend=['w/o PF (Exp)', 'with PF (Sim)'], loc='upper right')
plot.savefig(figurefolderName+'/time_resp_sim.png')

# FFT
fig = plot.makefig()
ax1 = fig.add_subplot(111)
plot.plot_xy(ax1, freq_fft, ErrPosUm_fft, '-', 'b', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft_sim, ErrPosUm_fft_pf_sim, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 1.6], xlabel='Frequency [Hz]', ylabel='Error Pos [um]', legend=['w/o PF (Exp)', 'with PF (Sim)'])
plot.savefig(figurefolderName+'/time_fft_sim.png')

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
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, '-', 'b', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, G_frd, '--', 'r', 1.5, 1.0, freqrange, legend=['w/o PF', 'with PF'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, '-', 'b', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, S_frd, '--', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['w/o PF', 'with PF'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, '-', 'b', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, T_frd, '--', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['w/o PF', 'with PF'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn_frd, '-', 'b', 1.5, 1.0, title='Nyquist diagram')
plot.plot_nyquist(ax, G_frd, '--', 'r', 1.5, 1.0, legend=['w/o PF', 'with PF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

print('Finished.')
