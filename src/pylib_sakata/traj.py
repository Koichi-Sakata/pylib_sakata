# Copyright (c) 2021 Koichi Sakata

# class TrajInf(time, pos, vel, acc, T, dt)
# TrajInf = traj3rd(posStart, posStep, velMax, accAve, dt, Tstay=0)
# TrajInf = traj4th(posStart, posStep, velMax, accAve, dt, Tstay=0)
# TrajInf = traj4th2(posStart, posStep, velMax, accAve, dt, Tstay=0)
# TrajInf = trajSinStep(posStart, posStep, velMax, accAve, dt, Tstay=0)
# TrajInf = trajSinStep2(posStart, posStep, velMax, accAve, dt, Tstay=0)
# TrajInf = trajSinStep3(posStart, posStep, velMax, accAve, dt, Tstay=0)

import numpy as np


class TrajInf():

    def __init__(self, time, pos, vel, acc, jerk, snap, T, dt):
        self.time = time
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.T = T
        self.dt = dt


def traj3rd(posStart, posStep, velMax, accAve, dt, Tstay=0):
    if velMax <= 0:
        print('Error: velMax should be more than 0.0')

    if accAve <= 0:
        print('Error: accAve should be more than 0.0')

    if posStep < 0:
        direction = False
        posStep *= -1
    else:
        direction = True

    velCon = velMax
    Tacc = velMax / accAve
    posCon = velMax * Tacc
    Tcon = (posStep - posCon) / velMax

    if posStep <= posCon:
        accAve = posStep / Tacc ** 2
        Tcon = 0.0
        velCon = accAve * Tacc

    if Tacc == 0.0:
        A = 0.0
    else:
        A = 4.0 * velCon / (Tacc ** 2)
    Tmove = 2.0 * Tacc + Tcon

    datalength = int((Tstay + Tmove) / dt + 1)

    time = np.array(range(datalength)) * dt
    pos = np.array([0.0] * datalength)
    vel = np.array([0.0] * datalength)
    acc = np.array([0.0] * datalength)
    jerk = np.array([0.0] * datalength)
    snap = np.array([0.0] * datalength)
    for i, t in enumerate(time):
        if t <= 0.0:
            pos[i] = 0.0
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif 0.0 < t <= 0.5 * Tacc:
            pos[i] = 1.0/6.0 * A * t ** 3
            vel[i] = 0.5 * A * t ** 2
            acc[i] = A * t
            jerk[i] = A
            snap[i] = 0.0
        elif 0.5 * Tacc < t <= Tacc:
            pos[i] = -1.0/6.0 * A * (t - Tacc) ** 3 + velCon * (t - Tacc) + 0.5 * velCon * Tacc
            vel[i] = -0.5 * A * (t - Tacc) ** 2 + velCon
            acc[i] = -A * (t - Tacc)
            jerk[i] = -A
            snap[i] = 0.0
        elif Tcon > 0.0 and Tacc < t <= Tacc + Tcon:
            t = t - Tacc
            pos[i] = velCon * t + 0.5 * velCon * Tacc
            vel[i] = velCon
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif Tacc + Tcon < t <= Tmove - 0.5 * Tacc:
            t = t - Tacc - Tcon
            pos[i] = -1.0/6.0 * A * t ** 3 + velCon * t + velCon * (Tcon + 0.5 * Tacc)
            vel[i] = -0.5 * A * t ** 2 + velCon
            acc[i] = -A * t
            jerk[i] = -A
            snap[i] = 0.0
        elif Tmove - 0.5 * Tacc < t <= Tmove:
            t = t - Tacc - Tcon
            pos[i] = 1.0/6.0 * A * (t - Tacc) ** 3 + posStep
            vel[i] = 0.5 * A * (t - Tacc) ** 2
            acc[i] = A * (t - Tacc)
            jerk[i] = A
            snap[i] = 0.0
        else:
            pos[i] = posStep
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0

        if not direction:
            # Move to minus direction
            pos[i] *= -1
            vel[i] *= -1
            acc[i] *= -1
            jerk[i] *= -1
            snap[i] *= -1
    pos += posStart
    return TrajInf(time, pos, vel, acc, jerk, snap, Tmove, dt)


def traj4th(posStart, posStep, velMax, accAve, dt, Tstay=0):
    if velMax <= 0:
        print('Error: velMax should be more than 0.0')

    if accAve <= 0:
        print('Error: accAve should be more than 0.0')

    if posStep < 0:
        direction = False
        posStep *= -1
    else:
        direction = True

    velCon = velMax
    Tacc = velMax / accAve
    posCon = velMax * Tacc
    Tcon = (posStep - posCon) / velMax

    if posStep <= posCon:
        accAve = posStep / Tacc ** 2
        Tcon = 0.0
        velCon = accAve * Tacc

    if Tacc == 0.0:
        A = 0.0
    else:
        A = 6.0 * velCon / (Tacc ** 3)
    Tmove = 2.0 * Tacc + Tcon

    datalength = int((Tstay + Tmove) / dt + 1)

    time = np.array(range(datalength)) * dt
    pos = np.array([0.0] * datalength)
    vel = np.array([0.0] * datalength)
    acc = np.array([0.0] * datalength)
    jerk = np.array([0.0] * datalength)
    snap = np.array([0.0] * datalength)
    for i, t in enumerate(time):
        if t <= 0.0:
            pos[i] = 0.0
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif 0.0 < t <= Tacc:
            pos[i] = -(1.0 / 12.0) * A * t ** 4 + (1.0 / 6.0) * A * Tacc * t ** 3
            vel[i] = -(1.0 / 3.0) * A * t ** 3 + 0.5 * A * Tacc * t ** 2
            acc[i] = -A * t ** 2 + A * Tacc * t
            jerk[i] = -2.0 * A * t + A * Tacc
            snap[i] = -2.0 * A
        elif Tcon > 0.0 and Tacc < t <= Tacc + Tcon:
            t = t - Tacc
            pos[i] = 0.5 * Tacc * velCon + velCon * t
            vel[i] = velCon
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif Tacc + Tcon < t < Tmove:
            t = t - Tacc - Tcon
            pos[i] = (1.0 / 12.0) * A * t ** 4 - (1.0 / 6.0) * A * Tacc * t ** 3 + velCon * t + velCon * (Tcon + 0.5 * Tacc)
            vel[i] = (1.0 / 3.0) * A * t ** 3 - (1.0 / 2.0) * A * Tacc * t ** 2 + velCon
            acc[i] = A * t ** 2 - A * Tacc * t
            jerk[i] = 2.0 * A * t - A * Tacc
            snap[i] = 2.0 * A
        else:
            pos[i] = posStep
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0

        if not direction:
            # Move to minus direction
            pos[i] *= -1
            vel[i] *= -1
            acc[i] *= -1
            jerk[i] *= -1
            snap[i] *= -1
    pos += posStart
    return TrajInf(time, pos, vel, acc, jerk, snap, Tmove, dt)


def traj4th2(posStart, posStep, velMax, accAve, dt, Tstay=0):
    if velMax <= 0:
        print('Error: velMax should be more than 0.0')

    if accAve <= 0:
        print('Error: accAve should be more than 0.0')

    if posStep < 0:
        direction = False
        posStep *= -1
    else:
        direction = True

    velCon = velMax
    Tacc = velMax / accAve
    posCon = velMax * Tacc
    Tcon = (posStep - posCon) / velMax

    if posStep <= posCon:
        accAve = posStep / Tacc ** 2
        Tcon = 0.0
        velCon = accAve * Tacc

    if Tacc == 0.0:
        A = 0.0
    else:
        A = accAve / Tacc ** 2
    Tmove = 2.0 * Tacc + Tcon

    datalength = int((Tstay + Tmove) / dt + 1)

    time = np.array(range(datalength)) * dt
    pos = np.array([0.0] * datalength)
    vel = np.array([0.0] * datalength)
    acc = np.array([0.0] * datalength)
    jerk = np.array([0.0] * datalength)
    snap = np.array([0.0] * datalength)
    for i, t in enumerate(time):
        if t <= 0.0:
            pos[i] = 0.0
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif 0.0 < t <= 0.25 * Tacc:
            pos[i] = (4.0 / 3.0) * A * t ** 4
            vel[i] = (16.0 / 3.0) * A * t ** 3
            acc[i] = 16.0 * A * t ** 2
            jerk[i] = 32.0 * A * t
            snap[i] = 32.0 * A
        elif 0.25 * Tacc < t <= 0.75 * Tacc:
            t = t - 0.25 * Tacc
            pos[i] = -(4.0 / 3.0) * A * (t - Tacc / 4.0) ** 4 + A * Tacc ** 2 * t ** 2 + (1.0 / 96.0) * A * Tacc ** 4
            vel[i] = -(16.0 / 3.0) * A * (t - Tacc / 4.0) ** 3 + 2.0 * A * Tacc ** 2 * t
            acc[i] = -16.0 * A * (t - Tacc / 4.0) ** 2 + 2.0 * A * Tacc ** 2
            jerk[i] = -32.0 * A * (t - Tacc / 4.0)
            snap[i] = -32.0 * A
        elif 0.75 * Tacc < t <= Tacc:
            t = t - 0.75 * Tacc
            pos[i] = (4.0 / 3.0) * A * (t - Tacc / 4.0) ** 4 + A * Tacc ** 3 * t + A * Tacc ** 4 / 4.0
            vel[i] = (16.0 / 3.0) * A * (t - Tacc / 4.0) ** 3 + A * Tacc ** 3
            acc[i] = 16.0 * A * (t - Tacc / 4.0) ** 2
            jerk[i] = 32.0 * A * (t - Tacc / 4.0)
            snap[i] = 32.0 * A
        elif Tcon > 0.0 and Tacc < t <= Tacc + Tcon:
            t = t - Tacc
            pos[i] = 0.5 * Tacc * velCon + velCon * t
            vel[i] = velCon
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif Tacc + Tcon < t <= 1.25 * Tacc + Tcon:
            t = t - (Tacc + Tcon)
            pos[i] = -(4.0 / 3.0) * A * t ** 4 + velCon * t + velCon * (Tcon + 0.5 * Tacc)
            vel[i] = -(16.0 / 3.0) * A * t ** 3 + velCon
            acc[i] = -16.0 * A * t ** 2
            jerk[i] = -32.0 * A * t
            snap[i] = -32.0 * A
        elif 1.25 * Tacc + Tcon < t <= 1.75 * Tacc + Tcon:
            t = t - (1.25 * Tacc + Tcon)
            pos[i] = (4.0 / 3.0) * A * (t - Tacc / 4.0) ** 4 - A * Tacc ** 2 * t ** 2 + velCon * t - (1.0 / 96.0) * A * Tacc ** 4 + velCon * (Tcon + 0.75 * Tacc)
            vel[i] = (16.0 / 3.0) * A * (t - Tacc / 4.0) ** 3 - 2.0 * A * Tacc ** 2 * t + velCon
            acc[i] = 16.0 * A * (t - Tacc / 4.0) ** 2 - 2.0 * A * Tacc ** 2
            jerk[i] = 32.0 * A * (t - Tacc / 4.0)
            snap[i] = 32.0 * A
        elif 1.75 * Tacc + Tcon < t < Tmove:
            t = t - (1.75 * Tacc + Tcon)
            pos[i] = -(4.0 / 3.0) * A * (t - Tacc / 4.0) ** 4 + posStep
            vel[i] = -(16.0 / 3.0) * A * (t - Tacc / 4.0) ** 3
            acc[i] = -16.0 * A * (t - Tacc / 4.0) ** 2
            jerk[i] = -32.0 * A * (t - Tacc / 4.0)
            snap[i] = -32.0 * A
        else:
            pos[i] = posStep
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0

        if not direction:
            # Move to minus direction
            pos[i] *= -1
            vel[i] *= -1
            acc[i] *= -1
            jerk[i] *= -1
            snap[i] *= -1
    pos += posStart
    return TrajInf(time, pos, vel, acc, jerk, snap, Tmove, dt)


def trajSinStep(posStart, posStep, velMax, accAve, dt, Tstay=0):
    if velMax <= 0:
        print('Error: velMax should be more than 0.0')

    if accAve <= 0:
        print('Error: accAve should be more than 0.0')

    if posStep < 0:
        direction = False
        posStep *= -1
    else:
        direction = True

    velCon = velMax
    Tacc = velMax / accAve
    posCon = velMax * Tacc
    Tcon = (posStep - posCon) / velMax

    if posStep <= posCon:
        accAve = posStep / Tacc ** 2
        Tcon = 0.0
        velCon = accAve * Tacc

    if Tacc == 0.0:
        A = 0.0
    else:
        A = 0.5 * velMax
    w = np.pi * accAve / velMax
    Tmove = 2.0 * Tacc + Tcon

    datalength = int((Tstay + Tmove) / dt + 1)

    time = np.array(range(datalength)) * dt
    pos = np.array([0.0] * datalength)
    vel = np.array([0.0] * datalength)
    acc = np.array([0.0] * datalength)
    jerk = np.array([0.0] * datalength)
    snap = np.array([0.0] * datalength)
    for i, t in enumerate(time):
        if t <= 0.0:
            pos[i] = 0.0
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif 0.0 < t <= Tacc:
            pos[i] = A * (-1 / w * np.sin(w * t) + t)
            vel[i] = A * (-np.cos(w * t) + 1)
            acc[i] = A * w * np.sin(w * t)
            jerk[i] = A * w ** 2 * np.cos(w * t)
            snap[i] = -A * w ** 3 * np.sin(w * t)
        elif Tcon > 0.0 and Tacc < t <= Tacc + Tcon:
            t = t - Tacc
            pos[i] = 0.5 * Tacc * velCon + velCon * t
            vel[i] = velCon
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif Tacc + Tcon < t < Tmove:
            t = t - Tacc - Tcon
            pos[i] = -A * (-1 / w * np.sin(w * t) + t) + velCon * t + velCon * (Tcon + 0.5 * Tacc)
            vel[i] = -A * (-np.cos(w * t) + 1) + velCon
            acc[i] = -A * w * np.sin(w * t)
            jerk[i] = -A * w ** 2 * np.cos(w * t)
            snap[i] = A * w ** 3 * np.sin(w * t)
        else:
            pos[i] = posStep
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0

        if not direction:
            # Move to minus direction
            pos[i] *= -1
            vel[i] *= -1
            acc[i] *= -1
            jerk[i] *= -1
            snap[i] *= -1
    pos += posStart
    return TrajInf(time, pos, vel, acc, jerk, snap, Tmove, dt)

def trajSinStep2(posStart, posStep, velMax, accAve, dt, Tstay=0):
    if velMax <= 0:
        print('Error: velMax should be more than 0.0')

    if accAve <= 0:
        print('Error: accAve should be more than 0.0')

    if posStep < 0:
        direction = False
        posStep *= -1
    else:
        direction = True

    velCon = velMax
    Tacc = velMax / accAve
    posCon = velMax * Tacc
    Tcon = (posStep - posCon) / velMax

    if posStep <= posCon:
        accAve = posStep / Tacc ** 2
        Tcon = 0.0
        velCon = accAve * Tacc

    if Tacc == 0.0:
        A = 0.0
    else:
        A = accAve
    w = 2 * np.pi / Tacc
    Tmove = 2.0 * Tacc + Tcon

    datalength = int((Tstay + Tmove) / dt + 1)

    time = np.array(range(datalength)) * dt
    pos = np.array([0.0] * datalength)
    vel = np.array([0.0] * datalength)
    acc = np.array([0.0] * datalength)
    jerk = np.array([0.0] * datalength)
    snap = np.array([0.0] * datalength)
    for i, t in enumerate(time):
        if t <= 0.0:
            pos[i] = 0.0
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif 0.0 < t <= Tacc:
            pos[i] = A * (1 / w ** 2 * (np.cos(w * t) - 1) + 0.5 * t ** 2)
            vel[i] = A * (-1 / w * np.sin(w * t) + t)
            acc[i] = A * (-np.cos(w * t) + 1)
            jerk[i] = A * w * np.sin(w * t)
            snap[i] = A * w ** 2 * np.cos(w * t)
        elif Tcon > 0.0 and Tacc < t <= Tacc + Tcon:
            t = t - Tacc
            pos[i] = 0.5 * Tacc * velCon + velCon * t
            vel[i] = velCon
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif Tacc + Tcon < t < Tmove:
            t = t - Tacc - Tcon
            pos[i] = -A * (1 / w ** 2 * (np.cos(w * t) - 1) + 0.5 * (t - Tacc) ** 2) + posStep
            vel[i] = -A * (-1 / w * np.sin(w * t) + t) + velCon
            acc[i] = -A * (-np.cos(w * t) + 1)
            jerk[i] = -A * w * np.sin(w * t)
            snap[i] = -A * w ** 2 * np.cos(w * t)
        else:
            pos[i] = posStep
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0

        if not direction:
            # Move to minus direction
            pos[i] *= -1
            vel[i] *= -1
            acc[i] *= -1
            jerk[i] *= -1
            snap[i] *= -1
    pos += posStart
    return TrajInf(time, pos, vel, acc, jerk, snap, Tmove, dt)


def trajSinStep3(posStart, posStep, velMax, accAve, dt, Tstay=0):
    if velMax <= 0:
        print('Error: velMax should be more than 0.0')

    if accAve <= 0:
        print('Error: accAve should be more than 0.0')

    if posStep < 0:
        direction = False
        posStep *= -1
    else:
        direction = True

    velCon = velMax
    Tacc = velMax / accAve
    posCon = velMax * Tacc
    Tcon = (posStep - posCon) / velMax

    if posStep <= posCon:
        accAve = posStep / Tacc ** 2
        Tcon = 0.0
        velCon = accAve * Tacc

    if Tacc == 0.0:
        A = 0.0
    else:
        A = 4 * accAve ** 2 / velCon

    w = 4 * np.pi / Tacc
    Tmove = 2.0 * Tacc + Tcon

    datalength = int((Tstay + Tmove) / dt + 1)

    time = np.array(range(datalength)) * dt
    pos = np.array([0.0] * datalength)
    vel = np.array([0.0] * datalength)
    acc = np.array([0.0] * datalength)
    jerk = np.array([0.0] * datalength)
    snap = np.array([0.0] * datalength)
    for i, t in enumerate(time):
        if t <= 0.0:
            pos[i] = 0.0
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif 0.0 < t <= 0.5 * Tacc:
            pos[i] = A * (1 / w ** 2 * (1 / w * np.sin(w * t) - t) + 1.0/6.0 * t ** 3)
            vel[i] = A * (1 / w ** 2 * (np.cos(w * t) - 1) + 0.5 * t ** 2)
            acc[i] = A * (-1 / w * np.sin(w * t) + t)
            jerk[i] = A * (-np.cos(w * t) + 1)
            snap[i] = A * w * np.sin(w * t)
        elif 0.5 * Tacc < t <= Tacc:
            pos[i] = -A * (1 / w ** 2 * (1 / w * np.sin(w * t) - (t - Tacc)) + 1.0 / 6.0 * (t - Tacc) ** 3) + velCon * t - 0.5 * velCon * Tacc
            vel[i] = -A * (1 / w ** 2 * (np.cos(w * t) - 1) + 0.5 * (t - Tacc) ** 2) + velCon
            acc[i] = -A * (-1 / w * np.sin(w * t) + t) + 4 * accAve
            jerk[i] = -A * (-np.cos(w * t) + 1)
            snap[i] = -A * w * np.sin(w * t)
        elif Tcon > 0.0 and Tacc < t <= Tacc + Tcon:
            t = t - Tacc
            pos[i] = velCon * t + 0.5 * velCon * Tacc
            vel[i] = velCon
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0
        elif Tacc + Tcon < t <= Tmove - 0.5 * Tacc:
            t = t - Tacc - Tcon
            pos[i] = -A * (1 / w ** 2 * (1 / w * np.sin(w * t) - t) + 1.0/6.0 * t ** 3) + velCon * t + velCon * (Tcon + 0.5 * Tacc)
            vel[i] = -A * (1 / w ** 2 * (np.cos(w * t) - 1) + 0.5 * t ** 2) + velCon
            acc[i] = -A * (-1 / w * np.sin(w * t) + t)
            jerk[i] = -A * (-np.cos(w * t) + 1)
            snap[i] = -A * w * np.sin(w * t)
        elif Tmove - 0.5 * Tacc < t <= Tmove:
            t = t - Tacc - Tcon
            pos[i] = A * (1 / w ** 2 * (1 / w * np.sin(w * t) - (t - Tacc)) + 1.0 / 6.0 * (t - Tacc) ** 3) + posStep
            vel[i] = A * (1 / w ** 2 * (np.cos(w * t) - 1) + 0.5 * (t - Tacc) ** 2)
            acc[i] = A * (-1 / w * np.sin(w * t) + t) - 4 * accAve
            jerk[i] = A * (-np.cos(w * t) + 1)
            snap[i] = A * w * np.sin(w * t)
        else:
            pos[i] = posStep
            vel[i] = 0.0
            acc[i] = 0.0
            jerk[i] = 0.0
            snap[i] = 0.0

        if not direction:
            # Move to minus direction
            pos[i] *= -1
            vel[i] *= -1
            acc[i] *= -1
            jerk[i] *= -1
            snap[i] *= -1
    pos += posStart
    return TrajInf(time, pos, vel, acc, jerk, snap, Tmove, dt)
