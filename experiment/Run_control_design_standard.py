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
M = 0.11
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

# Design PI velocity controller
freq1 = 20.0
zeta1 = 0.7
Cz_PI = ctrl.pi(freq1, zeta1, M, C, Ts)
Cz_PI_frd = ctrl.sys2frd(Cz_PI, freq)
print('PI velocity controller was designed.')

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

# Design peak filters
freqPF = [10.0, 20.0, 30.0, 60.0, 70.0, 90.0]
zetaPF = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
depthPF = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PFz = ctrl.pfopt(freqPF, zetaPF, depthPF, ctrl.feedback(Pnz, Cz_PID * NFz_all, sys='T'), Ts)
PFz_frd = 0.0
for i in range(len(freqPF)):
    PFz_frd += ctrl.sys2frd(PFz[i], freq)
print('Peak filters were designed.')

# Design DOB for FB
M_dob = M
C_dob = C
K_dob = K
freq_dob = 100.0
zeta_dob = 0.7
DOBfbu, DOBfby = ctrl.dob(freq_dob, zeta_dob, M_dob, C_dob, K_dob, Ts)
DOBfbu_frd = ctrl.sys2frd(DOBfbu, freq)
DOBfby_frd = ctrl.sys2frd(DOBfby, freq)
print('DOB was designed.')

# Design DOB for detection
freq_dob = 300.0
zeta_dob = 1.0
DOBestu, DOBesty = ctrl.dob(freq_dob, zeta_dob, M, C, K, Ts)
DOBestu_frd = ctrl.sys2frd(DOBestu, freq)
DOBesty_frd = ctrl.sys2frd(DOBesty, freq)
print('DOB was designed.')

# Design ZPETC
Czpetc, Nzpetc = ctrl.zpetc(Pnz, Ts)
Czpetc_frd = ctrl.sys2frd(Czpetc, freq)
lead_frd = ctrl.sys2frd(z**Nzpetc, freq)
print('ZPETC was designed.')

# Impedance control
M_imp = 0.027
C_imp = 0.7
K_imp = 0.0
ImpModel = ctrl.c2d(ctrl.tf([1.0], [M_imp, C_imp, K_imp]), Ts, method='zoh')
ImpModel_frd = ctrl.sys2frd(ImpModel, freq)
print('Impedance controller was designed.')

# Haptics control
freq1 = 800.0
freq2 = 100.0
feta2 = 0.7
Cz_Hap = ctrl.pd(freq1, freq2, zeta2, M_dob, C_dob, K_dob, Ts)
Cz_Hap_frd = ctrl.sys2frd(Cz_Hap, freq)
print('Haptics controller was designed.')

print('Creating parameter set Cpp and header files...')
axis_num = 6
Pmechz_axes = [Pmechz for i in range(axis_num)]
Cz_PID_axes = [Cz_PID for i in range(axis_num)]
Cz_PD_axes = [Cz_PD for i in range(axis_num)]
Cz_PI_axes = [Cz_PI for i in range(axis_num)]
NFz_axes = [[NFz[j] for j in range(len(NFz))] for i in range(axis_num)]
PFz_axes = [[PFz[j] for j in range(len(PFz))] for i in range(axis_num)]
DOBfbu_axes = [DOBfbu for i in range(axis_num)]
DOBfby_axes = [DOBfby for i in range(axis_num)]
DOBestu_axes = [DOBestu for i in range(axis_num)]
DOBesty_axes = [DOBesty for i in range(axis_num)]
Czpetc_axes = [Czpetc for i in range(axis_num)]
ImpModel_axes = [ImpModel for i in range(axis_num)]
Cz_Hap_axes = [Cz_Hap for i in range(axis_num)]

ctrl.makeprmset(srcpathName)
ctrl.defprmset(Pmechz_axes, 'gstModelInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(Cz_PID_axes, 'gstPIDInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(Cz_PD_axes, 'gstPDInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(Cz_PI_axes, 'gstPIInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(NFz_axes, 'gstNFInf['+str(axis_num)+']['+str(len(NFz))+']', srcpathName)
ctrl.defprmset(PFz_axes, 'gstPFInf['+str(axis_num)+']['+str(len(PFz))+']', srcpathName)
ctrl.defprmset(DOBfbu_axes, 'gstDOBfbuInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(DOBfby_axes, 'gstDOBfbyInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(DOBestu_axes, 'gstDOBestuInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(DOBesty_axes, 'gstDOBestyInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(Czpetc_axes, 'gstZPETInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(ImpModel_axes, 'gstImpInf['+str(axis_num)+']', srcpathName)
ctrl.defprmset(Cz_Hap_axes, 'gstHapInf['+str(axis_num)+']', srcpathName)

print('Finished.')
