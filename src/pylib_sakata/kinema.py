# Copyright (c) 2022 Koichi Sakata
# Reference: Humanoid Robot, Author: Shuuji Kajita

# class Link(sister, child, mother, p, R, q, a)
# class Target(p, R)
# forwardkinematics(uLink, j)
# ref = calcref(x, y, z, rol, pitch, yaw, R_init)
# inversekinematics(uLink, i, j, ref)
# inversekinematics_LM(uLink, i, j, ref)
# drawalljoints(ax, uLink, j, cradius=60.0, clen=40.0, styl='-', col='b', width=1.5, alpha=1.0, xrange=[-800, 800],
#                   yrange=[-800, 800], zrange=[0, 800], xlabel='X', ylabel='Y', zlabel='Z',
#                   legend=None, loc='best', title=None, xscale='linear', yscale='linear', zscale='linear',
#                   labelouter=True)
# plot_xyz(ax, x, y, z, styl='-', col='b', width=1.5, alpha=1.0,
#              xrange=None, yrange=None, zrange=None, xlabel=None, ylabel=None, zlabel=None,
#              legend=None, loc='best', title=None, xscale='linear', yscale='linear', zscale='linear', labelouter=True)

import numpy as np
import cmath
from matplotlib import colors

DBL_EPSILON = 2.2204460492503131E-16


class Link:

    def __init__(self, sister=-1, child=-1, mother=-1, p=np.matrix(np.zeros((3, 1))), R=np.matrix(np.eye(3)), q=0.0,
                 a=np.matrix(np.zeros((3, 1)))):
        self.sister = sister
        self.child = child
        self.mother = mother
        self.p = p
        self.R = R
        self.q = q
        self.a = a
        self.b = np.matrix(np.zeros((3, 1)))


class Target:

    def __init__(self, p, R):
        self.p = p
        self.R = R


def forwardkinematics(uLink, j):
    if j == -1:
        return
    if j != 0:
        i = uLink[j].mother
        uLink[j].p = uLink[i].R * uLink[j].b + uLink[i].p
        uLink[j].R = uLink[i].R * _rodrigues(uLink[j].a, uLink[j].q)

    forwardkinematics(uLink, uLink[j].sister)
    forwardkinematics(uLink, uLink[j].child)


def calcref(x, y, z, rol, pitch, yaw, R_init):
    p = np.matrix([[x], [y], [z]])
    R = R_init * _rodrigues(np.matrix([[0.0], [0.0], [1.0]]), rol)
    R = R * _rodrigues(np.matrix([[0.0], [1.0], [0.0]]), pitch)
    R = R * _rodrigues(np.matrix([[1.0], [0.0], [0.0]]), yaw)
    ref = Target(p, R)
    return ref


def inversekinematics(uLink, i, j, ref):
    lamb = 0.9
    forwardkinematics(uLink, i)
    for n in range(10):
        err = _calcvwerr(uLink, ref, uLink[j])
        if np.linalg.norm(err) < 1.0e-6:
            break
        jac = _calcjacobian(uLink, i, j)
        dq = lamb * np.linalg.inv(jac) * err
        _movejoints(uLink, i, j, dq)
        forwardkinematics(uLink, i)


def inversekinematics_LM(uLink, i, j, ref):
    idx = _findroute(uLink, i, j)
    wn_pos = 1.0/0.3
    wn_ang = 1/(2.0 * np.pi)
    We = np.matrix(np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang]))
    Wn = np.matrix(np.eye(len(idx)))
    forwardkinematics(uLink, i)
    for n in range(10):
        err = _calcvwerr(uLink, ref, uLink[j])
        ek = err.T * We * err
        jac = _calcjacobian(uLink, i, j)
        jac_h = jac.T * We * jac + Wn * (ek[0, 0])    # Hk + wn
        gerr = jac.T * We * err     # gk
        dq = np.linalg.inv(jac_h) * gerr
        _movejoints(uLink, i, j, dq)
        forwardkinematics(uLink, i)


def _rodrigues(a, q):
    a_hat = np.matrix([[0.0, -a[2, 0], a[1, 0]], [a[2, 0], 0.0, -a[0, 0]], [-a[1, 0], a[0, 0], 0.0]])
    y = np.matrix(np.eye(3)) + a_hat * np.sin(q) + a_hat ** 2 * (1 - np.cos(q))
    return y


def _calcjacobian(uLink, i, j):
    idx = _findroute(uLink, i, j)
    if not idx:
        return

    jsize = len(idx)
    jac = np.matrix(np.zeros([6, jsize]))
    target = (uLink[j].p)  # Absolute target position

    for n in range(jsize):
        k = idx[n]
        a = uLink[k].R * uLink[k].a  # Joint axis vector in world frame
        jac[:, n] = np.concatenate((np.matrix(np.cross(a.flatten(), target.flatten() - uLink[k].p.flatten())).T, a))
    return jac


def _calcvwerr(uLink, c_ref, c_now):
    p_err = c_ref.p - c_now.p
    R_err = np.linalg.inv(c_now.R) * c_ref.R
    w_err = c_now.R * _rot2omega(R_err)

    err = np.concatenate((p_err, w_err), 0)
    return err


def _movejoints(uLink, i, j, dq):
    route = _findroute(uLink, i, j)
    for n in range(len(route)):
        k = route[n]
        uLink[k].q = uLink[k].q + dq[n, 0]


def _findroute(uLink, i, j):
    k = uLink[j].mother
    if k == -1:
        idx = []
    elif k == i:
        idx = [j]
    else:
        idx2 = _findroute(uLink, i, k)
        if not idx2:
            idx = []
        else:
            idx = idx2 + [j]
    return idx


def _rot2omega(R):
    alpha = (np.trace(R) - 1.0) * 0.5

    if np.abs(alpha - 1.0) < DBL_EPSILON:
        w = np.matrix(np.zeros((3, 1)))
        return w
    th = cmath.acos(alpha)
    w = 0.5 * np.real(th / np.sin(th)) * np.matrix([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])
    return w


def drawalljoints(ax, uLink, j, cradius=60.0, clen=40.0, styl='-', col='k', width=1.5, alpha=1.0, xrange=[-800, 800],
                  yrange=[-800, 800], zrange=[0, 800], xlabel='X', ylabel='Y', zlabel='Z',
                  legend=None, loc='best', title=None, xscale='linear', yscale='linear', zscale='linear',
                  labelouter=True):
    col_list = list(colors.TABLEAU_COLORS.items())
    joint_col = col_list[j - 1][1]

    if int(j) != -1:
        if int(j) == 0:
            drawpolygon(ax, uLink[j].p, uLink[j].R, 10, 100, 100, 'k')
        elif int(j) == 7:
            drawpolygon(ax, uLink[j].p, uLink[j].R, 150, 60, 30, joint_col)
        else:
            drawcylinder(ax, uLink[j].p, uLink[j].R * uLink[j].a, cradius, clen, joint_col)
        i = int(uLink[j].mother)
        if i != -1:
            _connect3d(ax, uLink[i].p, uLink[j].p, styl=styl, col=col, width=width, alpha=alpha,
                      xrange=xrange, yrange=yrange, zrange=zrange, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                      legend=legend, loc=loc, title=title, xscale=xscale, yscale=yscale, zscale=zscale,
                      labelouter=labelouter)

        drawalljoints(ax, uLink, uLink[j].child, cradius, clen)
        drawalljoints(ax, uLink, uLink[j].sister, cradius, clen)


def _connect3d(ax, p1, p2, styl='-', col='k', width=1.5, alpha=1.0,
              xrange=[-800, 800], yrange=[-800, 800], zrange=[0, 800], xlabel='X', ylabel='Y', zlabel='Z',
              legend=None, loc='best', title=None, xscale='linear', yscale='linear', zscale='linear', labelouter=True):
    plot_xyz(ax, [p1[0, 0], p2[0, 0]], [p1[1, 0], p2[1, 0]], [p1[2, 0], p2[2, 0]],
             styl=styl, col=col, width=width, alpha=alpha,
             xrange=xrange, yrange=yrange, zrange=zrange, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
             legend=legend, loc=loc, title=title, xscale=xscale, yscale=yscale, zscale=zscale, labelouter=labelouter)


def plot_xyz(ax, x, y, z, styl='-', col='b', width=1.5, alpha=1.0,
             xrange=None, yrange=None, zrange=None, xlabel=None, ylabel=None, zlabel=None,
             legend=None, loc='best', title=None, xscale='linear', yscale='linear', zscale='linear', labelouter=True):
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_zscale(zscale)
    if xrange is None:
        xmin = min(x)
        xmax = max(x)
        xrange = [xmin, xmax]
    if yrange == None:
        ymin = min(y)
        ymax = max(y)
        yrange = [ymin, ymax]
    if zrange == None:
        zmin = min(z)
        zmax = max(z)
        zrange = [zmin, zmax]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim(zrange)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    if zlabel != None:
        ax.set_zlabel(zlabel)
    ax.grid(which='both', axis='both')
    # plot
    ax.plot(x, y, z, linestyle=styl, color=col, linewidth=width, alpha=alpha)
    # legend and title
    if legend != None:
        ax.legend(legend, loc=loc)
    if title != None:
        ax.set_title(title)
    if labelouter == True:
        ax.label_outer()


def drawcylinder(ax, pos, a_z, radius, length, col):
    # Rotation matrix
    a_z0 = np.matrix([[0.0], [0.0], [1.0]])
    a_x = np.cross(a_z0.flatten(), a_z.flatten()).T
    a_x_n = np.linalg.norm(a_x)

    if a_x_n < DBL_EPSILON:
        rot = np.matrix(np.eye(3))
    else:
        a_x = a_x / a_x_n
        a_y = np.cross(a_z.flatten(), a_x.flatten()).T
        a_y = a_y / np.linalg.norm(a_y)
        rot = np.concatenate((a_x, a_y, a_z), 1)

    a = 40  # Number of side faces
    theta = np.array(range(a + 1)) / a * 2.0 * np.pi
    x = np.matrix([[radius], [radius]]) * np.matrix(np.cos(theta))
    y = np.matrix([[radius], [radius]]) * np.matrix(np.sin(theta))
    z = np.matrix([[0.5 * length], [-0.5 * length]]) * np.matrix(np.ones([1, a + 1]))

    x2 = [[], []]
    y2 = [[], []]
    z2 = [[], []]
    for n in range(len(x)):
        xyz = np.concatenate((x[n], y[n], z[n]), 0)
        xyz2 = rot * xyz
        x2[n] = np.array(xyz2[0])[0] + pos[0, 0]
        y2[n] = np.array(xyz2[1])[0] + pos[1, 0]
        z2[n] = np.array(xyz2[2])[0] + pos[2, 0]

    # Tube
    ax.plot_surface(np.array(x2), np.array(y2), np.array(z2), color=col)

    # Top
    ax.plot_surface(np.array([x2[0], np.zeros(len(x2[0])) + np.average(x2[0])]),
                    np.array([y2[0], np.zeros(len(y2[0])) + np.average(y2[0])]),
                    np.array([z2[0], np.zeros(len(z2[0])) + np.average(z2[0])]), color=col)

    # Bottom
    ax.plot_surface(np.array([x2[1], np.zeros(len(x2[1])) + np.average(x2[1])]),
                    np.array([y2[1], np.zeros(len(y2[1])) + np.average(y2[1])]),
                    np.array([z2[1], np.zeros(len(z2[1])) + np.average(z2[1])]), color=col)


def drawpolygon(ax, pos, R, l, w, h, col):
    a = 4  # Number of side faces
    theta = np.array(range(a + 1)) / a * 2.0 * np.pi
    x = np.matrix([[h], [h]]) * np.matrix(np.cos(theta - 0.25 * np.pi))
    y = np.matrix([[w], [w]]) * np.matrix(np.sin(theta - 0.25 * np.pi))
    z = np.matrix([[0], [l]]) * np.matrix(np.ones([1, a + 1]))

    x2 = [[], []]
    y2 = [[], []]
    z2 = [[], []]
    for n in range(len(x)):
        xyz = np.concatenate((x[n], y[n], z[n]), 0)
        xyz2 = R * xyz
        x2[n] = np.array(xyz2[0])[0] + pos[0, 0]
        y2[n] = np.array(xyz2[1])[0] + pos[1, 0]
        z2[n] = np.array(xyz2[2])[0] + pos[2, 0]

    # Tube
    ax.plot_surface(np.array(x2), np.array(y2), np.array(z2), color=col)

    # Top
    ax.plot_surface(np.array([x2[0], np.zeros(len(x2[0])) + np.average(x2[0])]),
                    np.array([y2[0], np.zeros(len(y2[0])) + np.average(y2[0])]),
                    np.array([z2[0], np.zeros(len(z2[0])) + np.average(z2[0])]), color=col)

    # Bottom
    ax.plot_surface(np.array([x2[1], np.zeros(len(x2[1])) + np.average(x2[1])]),
                    np.array([y2[1], np.zeros(len(y2[1])) + np.average(y2[1])]),
                    np.array([z2[1], np.zeros(len(z2[1])) + np.average(z2[1])]), color=col)
