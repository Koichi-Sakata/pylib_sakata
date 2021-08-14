# Copyright (c) 2021 Koichi Sakata

# plot_xy(ax, x, y, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, xlabel=None, ylabel=None, legend=None, title=None, xscale='linear', yscale='linear', labelouter=True)
# plot_tf(ax_mag, ax_phase, sys, freq, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, title=None, labelouter=True)
# plot_tffrd(ax_mag, ax_phase, freqresp, freq, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, title=None, labelouter=True, ax_coh=None, coh=None)
# plot_nyquist(ax, freqresp, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, legend=None, title=None, labelouter=True)
# plot_nyquist_assistline(ax)
# makefig()
# savefig(figName)


import numpy as np
import matplotlib
from control import matlab
from matplotlib import pyplot as plt

def plot_xy(ax, x, y, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, xlabel=None, ylabel=None, legend=None, title=None, xscale='linear', yscale='linear', labelouter=True):
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xrange == None:
        xmin = min(x)
        xmax = max(x)
        xrange = [xmin, xmax]
    if yrange == None:
        ymin = min(y)
        ymax = max(y)
        yrange = [ymin - 0.2*(ymax-ymin), ymax + 0.2*(ymax-ymin)]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    ax.grid(b=True, which='both', axis='both')
    # plot
    ax.plot(x, y, linestyle=styl, color=col, linewidth=width, alpha=alpha)
    # legend and title
    if legend != None:
        ax.legend(legend, loc='best')
    if title != None:
        ax.set_title(title)
    if labelouter == True:
        ax.label_outer()


def plot_tf(ax_mag, ax_phase, sys, freq, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, title=None, labelouter=True):
    mag, phase, omega = matlab.freqresp(sys, freq*2.0*np.pi)
    magdb = 20.0*np.log10(mag)
    phasedeg = phase*180.0/np.pi

    ax_mag.set_xscale('log')
    if freqrange == None:
        freqmin = min(freq)
        freqmax = max(freq)
        freqrange = [freqmin, freqmax]
    if magrange == None:
        magmin = min(magdb)
        magmax = max(magdb)
        magrange = [magmin - 0.2*(magmax-magmin), magmax + 0.2*(magmax-magmin)]
    ax_mag.set_xlim(freqrange)
    ax_mag.set_ylim(magrange)
    if ax_phase == None:
            ax_mag.set_xlabel('Frequency [Hz]')
    ax_mag.set_ylabel('Magnitude [dB]')
    ax_mag.grid(b=True, which='both', axis='both')
    # mag plot
    ax_mag.plot(freq, magdb, linestyle=styl, color=col, linewidth=width, alpha=alpha)
    # legend and title
    if legend != None:
        ax_mag.legend(legend, loc='best')
    if title != None:
        ax_mag.set_title(title)
    if labelouter == True:
        ax_mag.label_outer()
    
    if ax_phase != None:
        ax_phase.set_xscale('log')
        ax_phase.set_xlim(freqrange)
        ax_phase.set_ylim(-200, 200)
        ax_phase.set_xlabel('Frequency [Hz]')
        ax_phase.set_ylabel('Phase [deg]')
        ax_phase.set_yticks([-180, -90, 0, 90, 180])
        ax_phase.grid(b=True, which='both', axis='both')
        # phase plot
        ax_phase.plot(freq, phasedeg, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        if labelouter == True:
            ax_phase.label_outer()  


def plot_tffrd(ax_mag, ax_phase, freqresp, freq, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, title=None, labelouter=True, ax_coh=None, coh=None):
    mag = np.absolute(freqresp)
    phase = np.angle(freqresp)
    magdb = 20.0*np.log10(mag)
    phasedeg = phase*180.0/np.pi

    ax_mag.set_xscale('log')
    if freqrange == None:
        freqmin = min(freq)
        freqmax = max(freq)
        freqrange = [freqmin, freqmax]
    if magrange == None:
        magmin = min(magdb)
        magmax = max(magdb)
        magrange = [magmin - 0.2*(magmax-magmin), magmax + 0.2*(magmax-magmin)]
    ax_mag.set_xlim(freqrange)
    ax_mag.set_ylim(magrange)
    if ax_phase == None and ax_coh == None:
            ax_mag.set_xlabel('Frequency [Hz]')
    ax_mag.set_ylabel('Magnitude [dB]')
    ax_mag.grid(b=True, which='both', axis='both')
    # mag plot
    ax_mag.plot(freq, magdb, linestyle=styl, color=col, linewidth=width, alpha=alpha)
    # legend and title
    if legend != None:
        ax_mag.legend(legend, loc='best')
    if title != None:
        ax_mag.set_title(title)
    if labelouter == True:
        ax_mag.label_outer()
    
    if ax_phase != None:
        ax_phase.set_xscale('log')
        ax_phase.set_xlim(freqrange)
        ax_phase.set_ylim(-200, 200)
        if ax_coh == None:
            ax_phase.set_xlabel('Frequency [Hz]')
        ax_phase.set_ylabel('Phase [deg]')
        ax_phase.set_yticks([-180, -90, 0, 90, 180])
        ax_phase.grid(b=True, which='both', axis='both')
        # phase plot
        ax_phase.plot(freq, phasedeg, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        if labelouter == True:
            ax_phase.label_outer()
        
    if ax_coh != None:
        ax_coh.set_xscale('log')
        ax_coh.set_xlim(freqrange)
        ax_coh.set_ylim(0, 1.2)
        ax_coh.set_xlabel('Frequency [Hz]')
        ax_coh.set_ylabel('Coherence [.]')
        ax_coh.grid(b=True, which='both', axis='both')
        # coherence plot
        ax_coh.plot(freq, coh, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        if labelouter == True:
            ax_phase.label_outer()


def plot_nyquist(ax, freqresp, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, legend=None, title=None, labelouter=True):
    x = np.real(freqresp)
    y = np.imag(freqresp)

    if xrange == None:
        xrange = [-2, 1]
    if yrange == None:
        yrange = [-1.5, 1.5]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(b=True, which='both', axis='both')
    # plot
    ax.plot(x, y, linestyle=styl, color=col, linewidth=width, alpha=alpha)
    # legend and title
    if legend != None:
        ax.legend(legend, loc='best')
    if title != None:
        ax.set_title(title)
    if labelouter == True:
        ax.label_outer()


def plot_nyquist_assistline(ax):
    cir = np.linspace(-np.pi, np.pi)
    cx = np.sin(cir)
    cy = np.cos(cir)
    # plot
    ax.plot(cx, cy, linestyle='-', color='gray', linewidth=0.5)
    ax.plot(0.5*cx-1, 0.5*cy, linestyle='-', color='gray', linewidth=0.5)
    ax.plot(-1.0, 0.0, marker='x', color='r')
    
    
def makefig(dpi=100, popwin=False):
    fig = plt.figure(dpi=dpi)
    if popwin != False:
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(50,250,640, 545)
    return fig


def savefig(figName):
    plt.savefig(figName)

