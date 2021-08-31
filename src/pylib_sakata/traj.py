# Copyright (c) 2021 Koichi Sakata

# class TrajInf(time, pos, vel, acc, T, dt)
# TrajInf = traj4th(posStart, posStep, velMax, accAve, dt, Tstay=0)


import numpy as np


class TrajInf():
 
    def __init__(self, time, pos, vel, acc, T, dt):
        self.time = time
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.T = T
        self.dt = dt


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
    Tacc = velMax/accAve
    posCon = velMax*Tacc
    Tcon = (posStep - posCon)/velMax

    if posStep <= posCon:
        Tacc = np.sqrt(posStep/accAve)
        Tcon = 0.0
        velCon = accAve*Tacc

    if Tacc == 0.0:
        A = 0.0
    else:
        A = 6.0 * velCon/(Tacc**3)
    Tmove = 2.0*Tacc + Tcon

    datalength = int((Tstay+Tmove)/dt+1)

    time = np.array(range(datalength))*dt
    pos = np.array([0.0] * datalength)
    vel = np.array([0.0] * datalength)
    acc = np.array([0.0] * datalength)
    for i, t in enumerate(time):
        if t <= 0.0:
            pos[i] = 0.0
            vel[i] = 0.0
            acc[i] = 0.0
        elif 0.0 < t <= Tacc:
            pos[i] = -(1.0/12.0)*A*t**4 + (1.0/6.0)*A*Tacc*t**3
            vel[i] = -(1.0/3.0)*A*t**3 + 0.5*A*Tacc*t**2
            acc[i] = -A*t**2 + A*Tacc*t
        elif Tcon > 0.0 and Tacc < t <= Tacc + Tcon:
            t = t - Tacc
            pos[i] = 0.5*Tacc*velCon + velCon*t
            vel[i] = velCon
            acc[i] = 0.0
        elif Tacc + Tcon < t < Tmove:
            t = t - Tacc - Tcon
            pos[i] = 0.5*Tacc*velCon + velCon*Tcon + velCon*t + (1.0/12.0)*A*t**4 - (1.0/6.0)*A*Tacc*t**3
            vel[i] = velCon + (1.0/3.0)*A*t**3 - 0.5*A*Tacc*t**2
            acc[i] = A*t**2 - A*Tacc*t
        else:
            pos[i] = posStep
            vel[i] = 0.0
            acc[i] = 0.0
        
        if not direction:
            # Move to minus direction
            pos[i] *= -1
            vel[i] *= -1
            acc[i] *= -1
    pos += posStart
    return TrajInf(time, pos, vel, acc, Tmove, dt)
