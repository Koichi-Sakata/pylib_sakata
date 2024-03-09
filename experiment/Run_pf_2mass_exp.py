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
figurefolderName = 'figure_pf_2mass_exp'
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
measdata = meas.getdata('data/time_resp_2mass_nf_20230720.csv')
time = measdata.time
RefPosUm = measdata.value[meas.getdataindex(measdata, 'RefPosUm[0]')]
ErrPosUm = measdata.value[meas.getdataindex(measdata, 'ErrPosUm[0]')]
ServoOutN = measdata.value[meas.getdataindex(measdata, 'ServoOutN[0]')]
# FFT
freq_fft, ErrPosUm_fft = fft.fft(ErrPosUm[8000:72000], Ts)
freq_fft, ServoOutN_fft = fft.fft(ServoOutN[8000:72000], Ts)

measdata_pf = meas.getdata('data/time_resp_2mass_nf_pf_20230720.csv')
time_pf = measdata_pf.time
RefPosUm_pf = measdata_pf.value[meas.getdataindex(measdata_pf, 'RefPosUm[0]')]
ErrPosUm_pf = measdata_pf.value[meas.getdataindex(measdata_pf, 'ErrPosUm[0]')]
ServoOutN_pf = measdata_pf.value[meas.getdataindex(measdata_pf, 'ServoOutN[0]')]
# FFT
freq_fft_pf, ErrPosUm_fft_pf = fft.fft(ErrPosUm_pf[8000:72000], Ts)
freq_fft_pf, ServoOutN_fft_pf = fft.fft(ServoOutN_pf[8000:72000], Ts)

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
# Measurement w/o PF
Gn_frd = Pmeas_frd * Cz_frd * NFz_frd
Sn_frd = 1/(1 + Gn_frd)
Tn_frd = 1 - Sn_frd

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
# Measurement with PF
G_frd = Pmeas_frd * Cz_frd * NFz_frd * (1.0+PFz_frd)
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
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, time, RefPosUm*1.0e-3, '-', 'b', 1.5, 1.0, ylabel='Ref Pos [mm]', title='Time response')
plot.plot_xy(ax2, time, ErrPosUm, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax2, time_pf, ErrPosUm_pf, '--', 'r', 1.5, 1.0, yrange=[-1.0, 1.0], ylabel='Error Pos [um]')
plot.plot_xy(ax3, time, ServoOutN, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax3, time_pf, ServoOutN_pf, '--', 'r', 1.5, 1.0, xlabel='Time [s]', ylabel='ServoOut [N]', legend=['with NF only', 'with NF+PF'])
plot.savefig(figurefolderName+'/time_resp.png')

# FFT
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, freq_fft, ErrPosUm_fft, '-', 'g', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft_pf, ErrPosUm_fft_pf, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.4], ylabel='Error Pos [um]', legend=['with NF only', 'with NF+PF'])
plot.plot_xy(ax2, freq_fft, ServoOutN_fft, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax2, freq_fft_pf, ServoOutN_fft_pf, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.002], xlabel='Frequency [Hz]', ylabel='ServoOut [N]', legend=['with NF only', 'with NF+PF'])
plot.savefig(figurefolderName+'/time_fft.png')

# Time response
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, time, ErrPosUm, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax1, time_pf, ErrPosUm_pf, '--', 'r', 1.5, 1.0, yrange=[-1.0, 1.0], ylabel='Error Pos [um]', legend=['w/o PF (Exp)', 'with PF (Exp)'], loc='upper right', title='Time response')
plot.plot_xy(ax2, time, ErrPosUm, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax2, time_sim, ErrPosUm_pf_sim, '--', 'r', 1.5, 1.0, yrange=[-1.0, 1.0], xlabel='Time [s]', ylabel='Error Pos [um]', legend=['w/o PF (Exp)', 'with PF (Sim)'], loc='upper right')
plot.savefig(figurefolderName+'/time_resp_vs_sim.png')

# FFT
fig = plot.makefig()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plot.plot_xy(ax1, freq_fft, ErrPosUm_fft, '-', 'g', 1.5, 1.0, title='Power spectrum density')
plot.plot_xy(ax1, freq_fft_pf, ErrPosUm_fft_pf, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.4], ylabel='Error Pos [um]', legend=['w/o PF (Exp)', 'with PF (Exp)'])
plot.plot_xy(ax2, freq_fft, ErrPosUm_fft, '-', 'g', 1.5, 1.0)
plot.plot_xy(ax2, freq_fft_sim, ErrPosUm_fft_pf_sim, '--', 'r', 1.5, 1.0, xscale='log', xrange=[1.0, 1000.0], yrange=[0.0, 0.4], xlabel='Frequency [Hz]', ylabel='Error Pos [um]', legend=['w/o PF (Exp)', 'with PF (Sim)'])
plot.savefig(figurefolderName+'/time_fft_vs_sim.png')

# Plant
fig = plot.makefig()
ax_mag = fig.add_subplot(311)
ax_phase = fig.add_subplot(312)
ax_coh = fig.add_subplot(313)
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, '-', 'r', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
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
plot.plot_tffrd(ax_mag, ax_phase, Gn_frd, '-', 'g', 1.5, 1.0, title='Frequency response of open loop transfer function')
plot.plot_tffrd(ax_mag, ax_phase, G_frd, '--', 'r', 1.5, 1.0, freqrange, legend=['with NF only', 'with NF+PF'])
plot.savefig(figurefolderName+'/freq_G.png')

# Sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, '-', 'g', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, S_frd, '--', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['with NF only', 'with NF+PF'])
plot.savefig(figurefolderName+'/freq_S.png')

# Complementary sensitivity function
fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, '-', 'g', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, T_frd, '--', 'r', 1.5, 1.0, freqrange, [-60, 10], legend=['with NF only', 'with NF+PF'])
plot.savefig(figurefolderName+'/freq_T.png')

# Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, Gn_frd, '-', 'g', 1.5, 1.0, title='Nyquist diagram')
plot.plot_nyquist(ax, G_frd, '--', 'r', 1.5, 1.0, legend=['with NF only', 'with NF+PF'])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquist.png')

plot.showfig()
print('Finished.')