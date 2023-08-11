# Copyright (c) 2022 Koichi Sakata


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

print('Start simulation!')

# Common parameters
# srcpathName = 'src'
srcpathName = 'C:/Users/sakat/source/repos/TwinCAT-CppMotionControl-main/TwinCAT-CppMotionControl/StaticLibrary1'
Ts = 1/8000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
print('Common parameters were set.')

# Plant model
M = 0.027
C = 0.7
K = 0.0
Pmechs = ctrl.tf([1.0], [M, C, K])
Pmechz = ctrl.c2d(Pmechs, Ts, method='zoh')
numDelay, denDelay = matlab.pade(Ts*4, n=4)
Ds = ctrl.tf(numDelay, denDelay)
Dz = z**-3
Pns = Pmechs * Ds
Pnz = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
Pnz_frd = ctrl.sys2frd(Pnz, freq)
print('Plant model was set.')

# Design PD controller
freq1 = 10.0
freq2 = 10.0
zeta2 = 1.0
Cz_PD = ctrl.pd(freq1, freq2, zeta2, M, C, K, Ts)
Cz_PD_frd = ctrl.sys2frd(Cz_PD, freq)
print('PD controller was designed.')

# Design PID controller
freq1 = 25.0
zeta1 = 0.7
freq2 = 25.0
zeta2 = 0.7
Cz_PID = ctrl.pid(freq1, zeta1, freq2, zeta2, M, C, K, Ts)
Cz_PID_frd = ctrl.sys2frd(Cz_PID, freq)
print('PID controller was designed.')

print('Creating parameter set Cpp and header files...')
axis_num = 6
Pmechz_axes = [Pmechz for i in range(axis_num)]
Cz_PID_axes = [Cz_PID for i in range(axis_num)]
Cz_PD_axes = [Cz_PD for i in range(axis_num)]

ctrl.makeprmset(srcpathName)
ctrl.defprmset(Pmechz_axes, 'gstModelInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(Cz_PID_axes, 'gstPIDInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(Cz_PD_axes, 'gstPDInf['+str(axis_num)+']', srcpathName)

print('Finished.')
