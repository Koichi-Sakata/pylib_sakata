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
import matplotlib.animation as animation
import matplotlib.pyplot as plt

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_2nd_order_sys'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)
Ts = 1/8000
dataNum = 10000
freqrange = [0.1, 10]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# 2nd order system
zeta = [1e-6, 0.1, 0.3, 0.7, 1.0, 2.0]
omega = 2.0 * np.pi * 1.0
Gs = []
Gs_frd = []
Gs_pole = []
for k in range(len(zeta)):
    sys = ctrl.tf([omega**2], [1, 2.0 * zeta[k] * omega, omega**2])
    Gs.append(sys)
    Gs_frd.append(ctrl.sys2frd(sys, freq))
    Gs_pole.append(ctrl.tf2zpk(Gs[k]).p)

print('Time response analysis is running...')
t = np.linspace(0.0, 5.0, 200)
r = np.ones(len(t))
y = [[] for k in range(len(Gs))]
for k in range(len(Gs)):
    y[k], tout, xout = matlab.lsim(ctrl.tf2ss(Gs[k]), r, t)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax = fig.add_subplot(111)
legend = ['zeta = 0.0', 'zeta = 0.1', 'zeta = 0.3', 'zeta = 0.7', 'zeta = 1.0', 'zeta = 2.0']
color = ['k', 'b', 'c', 'g', 'r', 'm']
for k in range(len(y)):
    plot.plot_xy(ax, tout, y[k], '-', color[k], 1.0, 1.0, yrange=[0.0, 2.0], xlabel='Time [s]', ylabel='y', legend=legend, loc='upper right', title='Time response')
plot.savefig(figurefolderName+'/time_response1.png')

fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
for k in range(len(Gs_frd)):
    plot.plot_tffrd(ax_mag, ax_phase, Gs_frd[k], '-', color[k], 1.5, 1.0, freqrange, magrange=[-40, 40], legend=legend, loc='upper right', title='Frequency response')
plot.savefig(figurefolderName+'/freq_Gs1.png')

def axes_set_linewidth(axes, t=1, b=1, r=1, l=1):
    axes.spines['top'].set_linewidth(t)
    axes.spines['bottom'].set_linewidth(b)
    axes.spines['right'].set_linewidth(r)
    axes.spines['left'].set_linewidth(l)


fig = plot.makefig()
ax = fig.add_subplot()

axes_set_linewidth(ax, t=0, r=0, b=2, l=2)
ax.set_xlabel('zeta')
ax.set_ylabel('y')
ax.grid(visible=True, which='both', axis='both')
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2)
ax.plot([0, 2], [1, 1], color='k')

ims = []
for i in range(len(y[0])):
    p1 = ax.scatter(zeta[0], y[0][i], color=color[0], marker='s', s=500)
    p2 = ax.scatter(zeta[1], y[1][i], color=color[1], marker='s', s=500)
    p3 = ax.scatter(zeta[2], y[2][i], color=color[2], marker='s', s=500)
    p4 = ax.scatter(zeta[3], y[3][i], color=color[3], marker='s', s=500)
    p5 = ax.scatter(zeta[4], y[4][i], color=color[4], marker='s', s=500)
    p6 = ax.scatter(zeta[5], y[5][i], color=color[5], marker='s', s=500)
    ax.label_outer()
    ims.append([p1, p2, p3, p4, p5, p6])

ani = animation.ArtistAnimation(fig, ims, interval=25, repeat=None)
ani.save(figurefolderName+'/animation1.gif', writer='pillow')

fig = plot.makefig()
ax = fig.add_subplot()
ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')
ax.set_aspect('equal', adjustable='box')
ax.grid(visible=True, which='both', axis='both')
for k in range(len(Gs_pole)):
    ax.scatter(Gs_pole[k].real, Gs_pole[k].imag, color=color[k], label=legend[k], marker='x', s=100)
ax.legend()
cir = np.linspace(-np.pi, 0.0)
cx = np.sin(cir)
cy = np.cos(cir)
ax.plot(omega*cx, omega*cy, linestyle='-', color='gray', linewidth=0.5)
plt.title('Pole placement')
plot.savefig(figurefolderName+'/freq_pole1.png')

# 2nd order system
zeta = 0.7
fn = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
omega = 2.0 * np.pi * fn
Gs = []
Gs_frd = []
Gs_pole = []
for k in range(len(omega)):
    sys = ctrl.tf([omega[k]**2], [1, 2.0 * zeta * omega[k], omega[k]**2])
    Gs.append(sys)
    Gs_frd.append(ctrl.sys2frd(sys, freq))
    Gs_pole.append(ctrl.tf2zpk(Gs[k]).p)

print('Time response analysis is running...')
t = np.linspace(0.0, 5.0, 200)
r = np.ones(len(t))
y = [[] for k in range(len(Gs))]
for k in range(len(Gs)):
    y[k], tout, xout = matlab.lsim(ctrl.tf2ss(Gs[k]), r, t)

print('Plotting figures...')
# Time response
fig = plot.makefig()
ax = fig.add_subplot(111)
legend = ['fn = 0.1', 'fn = 0.2', 'fn = 0.4', 'fn = 0.8', 'fn = 1.6', 'fn = 3.2']
color = ['k', 'b', 'c', 'g', 'r', 'm']
for k in range(len(y)):
    plot.plot_xy(ax, tout, y[k], '-', color[k], 1.0, 1.0, yrange=[0.0, 2.0], xlabel='Time [s]', ylabel='y', legend=legend, loc='upper right', title='Time response')
plot.savefig(figurefolderName+'/time_response2.png')

fig = plot.makefig()
ax_mag = fig.add_subplot(211)
ax_phase = fig.add_subplot(212)
for k in range(len(Gs_frd)):
    plot.plot_tffrd(ax_mag, ax_phase, Gs_frd[k], '-', color[k], 1.5, 1.0, freqrange, magrange=[-40, 40], legend=legend, loc='upper right', title='Frequency response')
plot.savefig(figurefolderName+'/freq_Gs2.png')

def axes_set_linewidth(axes, t=1, b=1, r=1, l=1):
    axes.spines['top'].set_linewidth(t)
    axes.spines['bottom'].set_linewidth(b)
    axes.spines['right'].set_linewidth(r)
    axes.spines['left'].set_linewidth(l)


fig = plot.makefig()
ax = fig.add_subplot()

axes_set_linewidth(ax, t=0, r=0, b=2, l=2)
ax.set_xlabel('fn')
ax.set_ylabel('y')
ax.grid(visible=True, which='both', axis='both')
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2)
ax.plot([0, 3.2], [1, 1], color='k')

ims = []
for i in range(len(y[0])):
    p1 = ax.scatter(fn[0], y[0][i], color=color[0], marker='s', s=500)
    p2 = ax.scatter(fn[1], y[1][i], color=color[1], marker='s', s=500)
    p3 = ax.scatter(fn[2], y[2][i], color=color[2], marker='s', s=500)
    p4 = ax.scatter(fn[3], y[3][i], color=color[3], marker='s', s=500)
    p5 = ax.scatter(fn[4], y[4][i], color=color[4], marker='s', s=500)
    p6 = ax.scatter(fn[5], y[5][i], color=color[5], marker='s', s=500)
    ax.label_outer()
    ims.append([p1, p2, p3, p4, p5, p6])

ani = animation.ArtistAnimation(fig, ims, interval=25, repeat=None)
ani.save(figurefolderName+'/animation2.gif', writer='pillow')

fig = plot.makefig()
ax = fig.add_subplot()
ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')
ax.set_aspect('equal', adjustable='box')
ax.grid(visible=True, which='both', axis='both')
for k in range(len(Gs_pole)):
    ax.scatter(Gs_pole[k].real, Gs_pole[k].imag, color=color[k], label=legend[k], marker='x', s=100)
ax.legend()
plt.title('Pole placement')
plot.savefig(figurefolderName+'/freq_pole2.png')

plot.showfig()
print('Finished.')
