# Copyright (c) 2021 Koichi Sakata

# plot_xy(ax, x, y, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, xlabel=None, ylabel=None, legend=None, loc='best', title=None, xscale='linear', yscale='linear', labelouter=True)
# plot_tf(ax_mag, ax_phase, sys, freq, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, loc='best', title=None, labelouter=True)
# plot_tffrd(ax_mag, ax_phase, freqresp, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, loc='best', title=None, labelouter=True, ax_coh=None, coh=None)
# plot_nyquist(ax, freqresp, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, legend=None, loc='best', title=None, labelouter=True)
# plot_nyquist_assistline(ax)
# makefig()
# savefig(figName)
# showfig()


import numpy as np
import matplotlib
from control import matlab
from matplotlib import pyplot as plt
from .fft import FreqResp


def plot_xy(ax, x, y, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, xlabel=None, ylabel=None,
            legend=None, loc='best', title=None, xscale='linear', yscale='linear', labelouter=True):
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xrange == None:
        xmin = min(x)
        xmax = max(x)
        xrange = [xmin, xmax]
    if yrange == None:
        ymin = min(y)
        ymax = max(y)
        yrange = [ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin)]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    ax.grid(visible=True, which='both', axis='both')
    # plot
    ax.plot(x, y, linestyle=styl, color=col, linewidth=width, alpha=alpha)
    # legend and title
    if legend != None:
        ax.legend(legend, loc=loc)
    if title != None:
        ax.set_title(title)
    if labelouter == True:
        ax.label_outer()


def plot_tf(ax_mag, ax_phase, sys, freq, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None,
            legend=None, loc='best', title=None, labelouter=True):
    if type(freq) == list:
        freq = np.array(freq)
    mag, phase, omega = matlab.freqresp(sys, freq * 2.0 * np.pi)
    magdb = 20.0 * np.log10(mag)
    phasedeg = phase * 180.0 / np.pi

    if freqrange == None:
        freqmin = min(freqresp.freq)
        freqmax = max(freqresp.freq)
        freqrange = [freqmin, freqmax]

    if ax_mag != None:
        ax_mag.set_xscale('log')
        if magrange == None:
            magmin = min(magdb)
            magmax = max(magdb)
            magrange = [magmin - 0.2 * (magmax - magmin), magmax + 0.2 * (magmax - magmin)]
        ax_mag.set_xlim(freqrange)
        ax_mag.set_ylim(magrange)
        if ax_phase == None:
            ax_mag.set_xlabel('Frequency [Hz]')
        ax_mag.set_ylabel('Magnitude [dB]')
        ax_mag.grid(visible=True, which='both', axis='both')
        # mag plot
        ax_mag.plot(freq, magdb, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        # legend and title
        if legend != None:
            ax_mag.legend(legend, loc=loc)
        if title != None:
            ax_mag.set_title(title)
        if labelouter == True:
            ax_mag.label_outer()

    if ax_phase != None:
        ax_phase.set_xscale('log')
        ax_phase.set_xlim(freqrange)
        # ax_phase.set_ylim(-200, 200)
        ax_phase.set_ylim(-400, 40)
        ax_phase.set_xlabel('Frequency [Hz]')
        ax_phase.set_ylabel('Phase [deg]')
        # ax_phase.set_yticks([-180, -90, 0, 90, 180])
        ax_phase.set_yticks([-360, -270, -180, -90, 0])
        ax_phase.grid(visible=True, which='both', axis='both')
        # phase plot
        for k in range(len(phasedeg)):
            if phasedeg[k] > 0:
                phasedeg[k] -= 360
        ax_phase.plot(freq, phasedeg, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        # legend and title
        if ax_mag == None:
            if legend != None:
                ax_phase.legend(legend, loc=loc)
            if title != None:
                ax_phase.set_title(title)
        if labelouter == True:
            ax_phase.label_outer()


def plot_tffrd(ax_mag, ax_phase, freqresp, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None,
               legend=None, loc='best', title=None, labelouter=True, ax_coh=None, coh=None):
    mag = np.absolute(freqresp.resp)
    phase = np.angle(freqresp.resp)
    magdb = 20.0 * np.log10(mag)
    phasedeg = phase * 180.0 / np.pi

    if freqrange == None:
        freqmin = min(freqresp.freq)
        freqmax = max(freqresp.freq)
        freqrange = [freqmin, freqmax]

    if ax_mag != None:
        ax_mag.set_xscale('log')
        if magrange == None:
            magmin = min(magdb)
            magmax = max(magdb)
            magrange = [magmin - 0.2 * (magmax - magmin), magmax + 0.2 * (magmax - magmin)]
        ax_mag.set_xlim(freqrange)
        ax_mag.set_ylim(magrange)
        if ax_phase == None and ax_coh == None:
            ax_mag.set_xlabel('Frequency [Hz]')
        ax_mag.set_ylabel('Magnitude [dB]')
        ax_mag.grid(visible=True, which='both', axis='both')
        # mag plot
        ax_mag.plot(freqresp.freq, magdb, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        # legend and title
        if legend != None:
            ax_mag.legend(legend, loc=loc)
        if title != None:
            ax_mag.set_title(title)
        if labelouter == True:
            ax_mag.label_outer()

    if ax_phase != None:
        ax_phase.set_xscale('log')
        ax_phase.set_xlim(freqrange)
        # ax_phase.set_ylim(-200, 200)
        ax_phase.set_ylim(-400, 40)
        if ax_coh == None:
            ax_phase.set_xlabel('Frequency [Hz]')
        ax_phase.set_ylabel('Phase [deg]')
        # ax_phase.set_yticks([-180, -90, 0, 90, 180])
        ax_phase.set_yticks([-360, -270, -180, -90, 0])
        ax_phase.grid(visible=True, which='both', axis='both')
        # phase plot
        for k in range(len(phasedeg)):
            if phasedeg[k] > 0:
                phasedeg[k] -= 360
        ax_phase.plot(freqresp.freq, phasedeg, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        # legend and title
        if ax_mag == None:
            if legend != None:
                ax_phase.legend(legend, loc=loc)
            if title != None:
                ax_phase.set_title(title)
        if labelouter == True:
            ax_phase.label_outer()

    if ax_coh != None:
        ax_coh.set_xscale('log')
        ax_coh.set_xlim(freqrange)
        ax_coh.set_ylim(0, 1.2)
        ax_coh.set_xlabel('Frequency [Hz]')
        ax_coh.set_ylabel('Coherence [.]')
        ax_coh.grid(visible=True, which='both', axis='both')
        # coherence plot
        ax_coh.plot(freqresp.freq, coh, linestyle=styl, color=col, linewidth=width, alpha=alpha)
        if labelouter == True:
            ax_phase.label_outer()


def plot_nyquist(ax, freqresp, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, legend=None,
                 loc='best', title=None, labelouter=True):
    x = np.real(freqresp.resp)
    y = np.imag(freqresp.resp)

    if xrange == None:
        xrange = [-2, 1]
    if yrange == None:
        yrange = [-1.5, 1.5]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(visible=True, which='both', axis='both')
    # plot
    ax.plot(x, y, linestyle=styl, color=col, linewidth=width, alpha=alpha)
    # legend and title
    if legend != None:
        ax.legend(legend, loc=loc)
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
    ax.plot(0.5 * cx - 1, 0.5 * cy, linestyle='-', color='gray', linewidth=0.5)
    ax.plot(-1.0, 0.0, marker='x', color='r')


def makefig(dpi=100, figsize=(6, 4), popwin=False):
    fig = plt.figure(dpi=dpi, figsize=figsize)
    if popwin != False:
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.geometry(str(figsize[0]) + '00x' + str(figsize[1]) + '00+0+0')
    return fig


def savefig(figName):
    plt.savefig(figName, bbox_inches='tight')


def showfig():
    plt.show()
