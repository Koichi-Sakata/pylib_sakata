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
figurefolderName = 'figure_nf_pf_2mass_exp'
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
M1 = 0.0343
M2 = 0.0757
M = M1 + M2
C = 0.7
K = 0.0
Creso = 2.5
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

# Design PID controller
freq1 = 25.0
zeta1 = 0.7
freq2 = 25.0
zeta2 = 0.7
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
print('PID controller was designed.')

print('Getting measurement data...')
measfileName = 'data/freq_resp_2mass_20230720.csv'
# Frequency response
Pmeas_frd, coh = meas.measdata2frd(measfileName, 'ServoOutN[0]', 'ActPosUm[0]', 'FlagInject', freq, 1., 1.e-6, 8, 0.8)

# Time response
measdata = meas.getdata('data/time_resp_2mass_20230720.csv')
time = measdata.time
RefPosUm = measdata.value[meas.getdataindex(measdata, 'RefPosUm[0]')]
ErrPosUm = measdata.value[meas.getdataindex(measdata, 'ErrPosUm[0]')]
ServoOutN = measdata.value[meas.getdataindex(measdata, 'ServoOutN[0]')]
# FFT
freq_fft, ErrPosUm_fft = fft.fft(ErrPosUm[8000:72000], Ts)
freq_fft, ServoOutN_fft = fft.fft(ServoOutN[8000:72000], Ts)

measdata_nf = meas.getdata('data/time_resp_2mass_nf_20230720.csv')
time_nf = measdata_nf.time
RefPosUm_nf = measdata_nf.value[meas.getdataindex(measdata_nf, 'RefPosUm[0]')]
ErrPosUm_nf = measdata_nf.value[meas.getdataindex(measdata_nf, 'ErrPosUm[0]')]
ServoOutN_nf = measdata_nf.value[meas.getdataindex(measdata_nf, 'ServoOutN[0]')]
# FFT
freq_fft_nf, ErrPosUm_fft_nf = fft.fft(ErrPosUm_nf[8000:72000], Ts)
freq_fft_nf, ServoOutN_fft_nf = fft.fft(ServoOutN_nf[8000:72000], Ts)

measdata_nf_pf = meas.getdata('data/time_resp_2mass_nf_pf_20230720.csv')
time_nf_pf = measdata_nf_pf.time
RefPosUm_nf_pf = measdata_nf_pf.value[meas.getdataindex(measdata_nf_pf, 'RefPosUm[0]')]
ErrPosUm_nf_pf = measdata_nf_pf.value[meas.getdataindex(measdata_nf_pf, 'ErrPosUm[0]')]
ServoOutN_nf_pf = measdata_nf_pf.value[meas.getdataindex(measdata_nf_pf, 'ServoOutN[0]')]
# FFT
freq_fft_nf_pf, ErrPosUm_fft_nf_pf = fft.fft(ErrPosUm_nf_pf[8000:72000], Ts)
freq_fft_nf_pf, ServoOutN_fft_nf_pf = fft.fft(ServoOutN_nf_pf[8000:72000], Ts)

print('Frequency response analysis is running...')
# Measurement w/o Filters
G_frd = Pmeas_frd * Cz_frd
S_frd = 1/(1 + G_frd)
T_frd = 1 - S_frd

# Design notch filters
freqNF = [265]
zetaNF = [0.2]
depthNF = [0.02]
NFz = ctrl.nf(freqNF, zetaNF, depthNF, Ts)
NFz_all = 1.0
NFz_frd = 1.0
for i in range(len(freqNF)):
    NFz_all *= NFz[i]
    NFz_frd *= ctrl.sys2frd(NFz[i], freq)
print('Notch filters were designed.')

print('Frequency response analysis is running...')
# Measurement with NF
G_frd_nf = Pmeas_frd * Cz_frd * NFz_frd
S_frd_nf = 1/(1 + G_frd_nf)
T_frd_nf = 1 - S_frd_nf

# Design peak filters
freqPF = [10.0, 20.0, 30.0, 60.0, 70.0, 90.0]
zetaPF = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
depthPF = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PFz = ctrl.pfopt(freqPF, zetaPF, depthPF, ctrl.feedback(Pnz, Cz * NFz_all, sys='T'), Ts)
PFz_frd = 0.0
for i in range(len(freqPF)):
    PFz_frd += ctrl.sys2frd(PFz[i], freq)
print('Peak filters were designed.')

print('Frequency response analysis is running...')
# Measurement with NF and PF
G_frd_nf_pf = Pmeas_frd * Cz_frd * NFz_frd * (1.0+PFz_frd)
S_frd_nf_pf = 1/(1 + G_frd_nf_pf)
T_frd_nf_pf = 1 - S_frd_nf_pf

print('Simulating time response with peak filters')
# Time response
time_sim_nf, ErrPosUm_sim_nf = fft.frdsim(S_frd_nf/S_frd, ErrPosUm, Ts)
time_sim_nf_pf, ErrPosUm_sim_nf_pf = fft.frdsim(S_frd_nf_pf/S_frd, ErrPosUm, Ts)
# FFT
freq_fft_sim_nf, ErrPosUm_fft_sim_nf = fft.fft(ErrPosUm_sim_nf[8000:72000], Ts)
freq_fft_sim_nf_pf, ErrPosUm_fft_sim_nf_pf = fft.fft(ErrPosUm_sim_nf_pf[8000:72000], Ts)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, time, RefPosUm*1.0e-3, '-', 'b', 1.5, 1.0, ylabel='Ref Pos [mm]', title='Time response')
plot.plot_xy(ax2, time, ErrPosUm, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax2, time_nf, ErrPosUm_nf, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax2, time_nf_pf, ErrPosUm_nf_pf, '-', 'r', 1.5, 1.0, yrange=[-2.5, 2.5], ylabel='Error Pos [um]')
plot.plot_xy(ax3, time, ServoOutN, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax3, time_nf, ServoOutN_nf, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax3, time_nf_pf, ServoOutN_nf_pf, '-', 'r', 1.5, 1.0, xlabel='Time [s]', ylabel='ServoOut [N]', legend=['w/o Filters', 'with NF', 'with NF+PF'])
plot.savefig(figurefolderName+'/time_resp.svg')

# FFT
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, freq_fft, ErrPosUm_fft, '-', 'b', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft_nf, ErrPosUm_fft_nf, '--', 'g', 1.5, 1.0)
plot.plot_xy(ax1, freq_fft_nf_pf, ErrPosUm_fft_nf_pf, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.4], ylabel='Error Pos [um]', legend=['w/o Filters', 'with NF', 'with NF+PF'])
plot.plot_xy(ax2, freq_fft, ServoOutN_fft, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax2, freq_fft_nf, ServoOutN_fft_nf, '--', 'g', 1.5, 1.0)
plot.plot_xy(ax2, freq_fft_nf_pf, ServoOutN_fft_nf_pf, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.002], xlabel='Frequency [Hz]', ylabel='ServoOut [N]', legend=['w/o Filters', 'with NF', 'with NF+PF'])
plot.savefig(figurefolderName+'/time_fft.svg')

# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, time, ErrPosUm, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax1, time_nf, ErrPosUm_nf, '--', 'g', 1.5, 1.0)
plot.plot_xy(ax1, time_nf_pf, ErrPosUm_nf_pf, '--', 'r', 1.5, 1.0, yrange=[-2.5, 2.5], ylabel='Error Pos [um]', legend=['w/o Filters (Exp)', 'with NF (Exp)', 'with NF+PF (Exp)'], loc='upper right', title='Time response')
plot.plot_xy(ax2, time, ErrPosUm, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax2, time_sim_nf, ErrPosUm_sim_nf, '--', 'g', 1.5, 1.0)
plot.plot_xy(ax2, time_sim_nf_pf, ErrPosUm_sim_nf_pf, '--', 'r', 1.5, 1.0, yrange=[-1.0, 1.0], xlabel='Time [s]', ylabel='Error Pos [um]', legend=['w/o Filters (Exp)', 'with NF (Sim)', 'with NF+PF (Sim)'], loc='upper right')
plot.savefig(figurefolderName+'/time_resp_vs_sim.svg')

# FFT
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, freq_fft, ErrPosUm_fft, '-', 'b', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft_nf, ErrPosUm_fft_nf, '--', 'g', 1.5, 1.0)
plot.plot_xy(ax1, freq_fft_nf_pf, ErrPosUm_fft_nf_pf, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.4], ylabel='Error Pos [um]', legend=['w/o Filters (Exp)', 'with NF (Exp)', 'with NF+PF (Exp)'])
plot.plot_xy(ax2, freq_fft, ErrPosUm_fft, '-', 'b', 1.5, 1.0)
plot.plot_xy(ax2, freq_fft_sim_nf, ErrPosUm_fft_sim_nf, '--', 'g', 1.5, 1.0)
plot.plot_xy(ax2, freq_fft_sim_nf_pf, ErrPosUm_fft_sim_nf_pf, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.4], xlabel='Frequency [Hz]', ylabel='Error Pos [um]', legend=['w/o Filters (Exp)', 'with NF (Sim)', 'with NF+PF (Sim)'])
plot.savefig(figurefolderName+'/time_fft_vs_sim.svg')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(311)
ax_phase = fig.add_subplot(312)
ax_coh = fig.add_subplot(313)
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, '-', 'm', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement', 'Model'])
plot.savefig(figurefolderName+'/freq_P.svg')

# PID controller
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Cz_frd, '-', 'b', 1.5, 1.0, freqrange, title='Frequency response of PID controller')
plot.savefig(figurefolderName+'/freq_C.svg')

# Open loop function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, G_frd, '-', 'b', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, G_frd_nf, '-', 'g', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, G_frd_nf_pf, '-', 'r', 1.5, 1.0, freqrange, legend=['w/o Filters', 'with NF', 'with NF+PF'])
plot.savefig(figurefolderName+'/freq_G.svg')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, S_frd, '-', 'b', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, S_frd_nf, '-', 'g', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, S_frd_nf_pf, '-', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['w/o Filters', 'with NF', 'with NF+PF'])
plot.savefig(figurefolderName+'/freq_S.svg')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, T_frd, '-', 'b', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, T_frd_nf, '-', 'g', 1.5, 1.0)
plot.plot_tffrd(ax_mag, ax_phase, T_frd_nf_pf, '-', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['w/o Filters', 'with NF', 'with NF+PF'])
plot.savefig(figurefolderName+'/freq_T.svg')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, G_frd, '-', 'b', 1.5, 1.0, title='Nyquist diagram')
plot.plot_nyquist(ax, G_frd_nf, '-', 'g', 1.5, 1.0)
plot.plot_nyquist(ax, G_frd_nf_pf, '-', 'r', 1.5, 1.0, legend=['w/o Filters', 'with NF', 'with NF+PF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.svg')

print('Finished.')