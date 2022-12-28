# Copyright (c) 2021 Koichi Sakata

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylib_sakata import kinema

print('Start simulation!')

# Common parameters
figurefolderName = 'figure_kinema'
if os.path.exists(figurefolderName):
    shutil.rmtree(figurefolderName)
os.makedirs(figurefolderName)

# Define link structure
uLink = []
for k in range(8):
    if k == 7:
        uLink.append(kinema.Link(-1, -1, k - 1))
    else:
        uLink.append(kinema.Link(-1, k + 1, k - 1))

offsetX = 0.0
offsetY = 0.0
offsetZ = 100.0
uLink[1].p = np.matrix([[0.0], [0.0], [120.0]])
uLink[2].p = np.matrix([[0.0], [-150.0], [120.0]])
uLink[3].p = np.matrix([[0.0], [0.0], [540.0]])
uLink[4].p = np.matrix([[0.0], [0.0], [900.0]])
uLink[5].p = np.matrix([[0.0], [-130.0], [900.0]])
uLink[6].p = np.matrix([[60.0], [-130.0], [1000.0]])
uLink[7].p = np.matrix([[60.0 + offsetZ], [-130.0 + offsetY], [1000.0 + offsetX]])

uLink[2].q = -0.2 * np.pi
uLink[3].q = 0.7 * np.pi

uLink[1].a = np.matrix([[0.0], [0.0], [1.0]])
uLink[2].a = np.matrix([[0.0], [1.0], [0.0]])
uLink[3].a = np.matrix([[0.0], [1.0], [0.0]])
uLink[4].a = np.matrix([[0.0], [1.0], [0.0]])
uLink[5].a = np.matrix([[0.0], [0.0], [1.0]])
uLink[6].a = np.matrix([[1.0], [0.0], [0.0]])
uLink[7].a = np.matrix([[1.0], [0.0], [0.0]])

for k in range(1, 8):
    uLink[k].b = uLink[k].p - uLink[k - 1].p

# Initialization
kinema.forwardkinematics(uLink, 0)

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 0.5])
kinema.drawalljoints(ax, uLink, 0)

# Trajectory
target_init = kinema.Target(uLink[6].p, uLink[6].R)
target = kinema.Target(uLink[6].p, uLink[6].R)

x_ref = target_init.p[0, 0]
y_ref = target_init.p[1, 0]
z_ref = target_init.p[2, 0]

rol_ref = 0.0
pitch_ref = 0.0
yaw_ref = 0.0

dataNum = 300
Ts = 50 * 1.e-3
t = np.array(range(dataNum)) * Ts

# Move joints
def update(t):
    global x_ref, y_ref, z_ref, rol_ref, pitch_ref, yaw_ref
    ax.cla()
    ax.grid()

    print(t)
    if t <= 0:
        x_ref = x_ref
        y_ref = y_ref
        z_ref = z_ref
    elif 0 < t <= 1.0:
        x_ref = x_ref
        y_ref = y_ref
        z_ref = z_ref - 200.0 * Ts
    elif 1.0 < t <= 2.0:
        x_ref = x_ref
        y_ref = y_ref - 400.0 * Ts
        z_ref = z_ref
    elif 2.0 < t <= 3.0:
        x_ref = x_ref + 400.0 * Ts
        y_ref = y_ref
        z_ref = z_ref
    elif 3.0 < t <= 5.0:
        x_ref = x_ref
        y_ref = y_ref + 400.0 * Ts
        z_ref = z_ref
    elif 5.0 < t <= 6.0:
        x_ref = x_ref - 400 * Ts
        y_ref = y_ref
        z_ref = z_ref
    elif 6.0 < t <= 7.0:
        x_ref = x_ref
        y_ref = y_ref - 400.0 * Ts
        z_ref = z_ref
    elif 7.0 < t <= 8.0:
        x_ref = x_ref
        y_ref = y_ref
        z_ref = z_ref + 200.0 * Ts
    elif 8.0 < t <= 9.0:
        yaw_ref = 0.5 * np.pi * np.sin(2.0 * np.pi * (t - 8.0))
    elif 9.0 < t <= 10.0:
        pitch_ref = pitch_ref + 0.25 * np.pi * Ts
    elif 10.0 < t <= 14.0:
        rol_ref = 0.25 * np.pi * np.sin(2.0 * np.pi * (t - 10.0))
        pitch_ref = 0.25 * np.pi * np.cos(2.0 * np.pi * (t - 10.0))
    elif 14.0 < t <= 15.0:
        pitch_ref = pitch_ref - 0.25 * np.pi * Ts

    ref = kinema.calcref(x_ref, y_ref, z_ref, rol_ref, pitch_ref, yaw_ref, target_init.R)
    kinema.inversekinematics(uLink, 0, 6, ref)
    kinema.drawalljoints(ax, uLink, 0)


movie = animation.FuncAnimation(fig, update, frames=t, interval=50)

plt.show()

movie.save(figurefolderName+'/movie.gif', writer="ffmpeg")
plt.close()

print('Finished.')
