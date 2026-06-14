# Copyright (c) 2025 Koichi Sakata


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
from pylib_sakata import fft

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_mimo'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Ts = 1/8000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
colorlist = ['b', 'g', 'r', 'c', 'm', 'y']
print('Common parameters were set.')

# Plant model
M = 0.1
C = 0.7
K = 0.0
fanti = 100
freso = 200
Creso = 2.5
M1 = (fanti / freso) ** 2 * M
M2 = M - M1
Kreso = (2.0 * np.pi * fanti) ** 2 * M2
k1 = M2/(M1 * (M1 + M2))
k2 = -1.0/(M1 + M2)
omegaPreso = np.sqrt(Kreso * (M1 + M2)/(M1 * M2))
zetaPreso = 0.5 * Creso*np.sqrt((M1 + M2)/(Kreso * M1 * M2))
Pmechs1 = ctrl.tf([1], [M, C, K]) + k1 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Pmechs2 = ctrl.tf([1], [M, C, K]) + k2 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Preso1 = k1 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Preso2 = k2 * ctrl.tf([1], [1, 2*zetaPreso*omegaPreso, omegaPreso**2])
Dz = z**-3
Pns00 = Pmechs1
Pns01 = Preso1
Pns10 = Preso2
Pns11 = Pmechs2
Pnz00 = ctrl.c2d(Pns00, Ts, method='zoh') * Dz
Pnz01 = ctrl.c2d(Pns01, Ts, method='zoh') * Dz
Pnz10 = ctrl.c2d(Pns10, Ts, method='zoh') * Dz
Pnz11 = ctrl.c2d(Pns11, Ts, method='zoh') * Dz
Pnz00_frd = ctrl.sys2frd(Pns00, freq)
Pnz01_frd = ctrl.sys2frd(Pns01, freq)
Pnz10_frd = ctrl.sys2frd(Pns10, freq)
Pnz11_frd = ctrl.sys2frd(Pns11, freq)

mimoresp = np.empty(shape=(2, 2, len(freq)), dtype=complex)
for k in range(len(freq)):
    mimoresp[0, 0, k] = Pnz00_frd.resp[k]
    mimoresp[0, 1, k] = Pnz01_frd.resp[k]
    mimoresp[1, 0, k] = Pnz10_frd.resp[k]
    mimoresp[1, 1, k] = Pnz11_frd.resp[k]
P_mimo_frd = fft.FreqResp(freq, mimoresp, Ts)

fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, fft.FreqResp(freq, P_mimo_frd.resp[0, 0, :]), '-', 'b', 1.5, 1.0, title='Frequency response of plant')

# Design PID controller
freq1 = 20.0
zeta1 = 0.7
freq3 = 10
freq4 = 50
freq2 = np.sqrt(freq3 * freq4)
zeta2 = 0.5*(freq3 + freq4)/freq2
Cz = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_frd = ctrl.sys2frd(Cz, freq)
mimoresp = np.empty(shape=(2, 2, len(freq)), dtype=complex)
for k in range(len(freq)):
    mimoresp[0, 0, k] = Cz_frd.resp[k]
    mimoresp[0, 1, k] = 0.0
    mimoresp[1, 0, k] = 0.0
    mimoresp[1, 1, k] = Cz_frd.resp[k]
C_mimo_frd = fft.FreqResp(freq, mimoresp, Ts)
print('PID controller was designed.')

print('Frequency response analysis is running...')
G_mimo_frd = P_mimo_frd * C_mimo_frd
L_mimo_frd = P_mimo_frd @ C_mimo_frd
S_mimo_frd = L_mimo_frd.addeye().pinv()
T_mimo_frd = (-S_mimo_frd).addeye()
rga_frd = P_mimo_frd.rga()
det_frd = L_mimo_frd.addeye().det()
eig_frd = L_mimo_frd.eig()
sv_frd = L_mimo_frd.svd()

print('Plotting figures...')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

nrows = 2
ncols = 2
magrange = [-150, 50]
loc = 'upper left'
title='P'
fig = plot.makefig(figsize=(10, 8))
plt.suptitle('Frequency response of plant')
gs_main = gridspec.GridSpec(nrows, ncols, figure=fig)
for m in range(nrows):
    for n in range(ncols):
        gs_nested = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[m, n])
        ax_mag = fig.add_subplot(gs_nested[0, 0])
        ax_phase = fig.add_subplot(gs_nested[1, 0])
        plot.plot_tffrd(ax_mag, ax_phase, fft.FreqResp(P_mimo_frd.freq, P_mimo_frd.resp[m, n,:], P_mimo_frd.dt), styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=magrange,
                legend='P', loc=loc, title=None, labelouter=True)
        plot.plot_tffrd(ax_mag, ax_phase, fft.FreqResp(rga_frd.freq, rga_frd.resp[m, n,:], rga_frd.dt), styl='-', col='r', width=1.5, alpha=1.0, freqrange=None, magrange=magrange,
                legend='RGA', loc=loc, title=title+f'({m}, {n})')
plt.tight_layout()
plot.savefig(figurefolderName+'/freq_plant.png')

loc = 'upper left'
title='Sys'
fig = plot.makefig(figsize=(10, 8))
plt.suptitle('Frequency response of closed loop')
gs_main = gridspec.GridSpec(nrows, ncols, figure=fig)
for m in range(nrows):
    for n in range(ncols):
        gs_nested = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[m, n])
        ax_mag = fig.add_subplot(gs_nested[0, 0])
        ax_phase = fig.add_subplot(gs_nested[1, 0])
        plot.plot_tffrd(ax_mag, ax_phase, fft.FreqResp(S_mimo_frd.freq, S_mimo_frd.resp[m, n,:], S_mimo_frd.dt), styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=magrange, phasebase=180,
                legend='S', loc=loc, title=None, labelouter=True)
        plot.plot_tffrd(ax_mag, ax_phase, fft.FreqResp(T_mimo_frd.freq, T_mimo_frd.resp[m, n,:], T_mimo_frd.dt), styl='-', col='r', width=1.5, alpha=1.0, freqrange=None, magrange=magrange, phasebase=180,
                legend='T', loc=loc, title=title+f'({m}, {n})')
plt.tight_layout()
plot.savefig(figurefolderName+'/freq_closed.png')

# MIMO Nyquist
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, det_frd, '-', 'b', 1.5, 1.0, title='MIMO Nyquist diagram')
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquistmimo.png')

# SISO Nyquist
legend=['L0', 'L1']
fig = plot.makefig()
ax = fig.add_subplot(111)
for m in range(nrows):
    plot.plot_nyquist(ax, fft.FreqResp(G_mimo_frd.freq, G_mimo_frd.resp[m, m, :], G_mimo_frd.dt), '-', colorlist[m], 1.5, 1.0, title='SISO Nyquist diagram', legend=legend[m])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/nyquistsiso.png')

# Eigenvalue loci
legend=['Eig0', 'Eig1']
fig = plot.makefig()
ax = fig.add_subplot(111)
for m in range(nrows):
    plot.plot_nyquist(ax, fft.FreqResp(eig_frd.freq, eig_frd.resp[m, :], eig_frd.dt), '-', colorlist[m], 1.5, 1.0, title='Eigenvalue loci', legend=legend[m])
plot.plot_nyquist_assistline(ax)
plot.savefig(figurefolderName+'/eigloci.png')

print('Finished.')
