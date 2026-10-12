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
from pylib_sakata import meas

print('Start simulation!')

# Common parameters
srcpathName = 'src'
# srcpathName = 'C:/Users/sakat/source/repos/TwinCAT-CppMotionControl-main/TwinCAT-CppMotionControl/StaticLibrary1'
ftype = 'cpp'
Ts = 1/8000
dataNum = 10000
freqrange = [1, 1000]
freq = np.logspace(np.log10(freqrange[0]), np.log10(freqrange[1]), dataNum, base=10)
s = ctrl.tf([1, 0], [1])
z = ctrl.tf([1, 0], [1], Ts)
axis_num = 6
print('Common parameters were set.')

# Plant model
M = [0.0226, 0.0226, 0.0226, 0.0226, 0.0226, 0.0226]
C = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
K = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DelaySample = [3, 3, 3, 3, 3, 3]
Pmechz = [[] for i in range(axis_num)]
Pnz = [[] for i in range(axis_num)]
Pnz_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    Pmechs = ctrl.tf([1.0], [M[i], C[i], K[i]])
    Pmechz[i] = ctrl.c2d(Pmechs, Ts, method='zoh')
    Dz = z**-DelaySample[i]
    Pnz[i] = ctrl.c2d(Pmechs, Ts, method='zoh') * Dz
    Pnz_frd[i] = ctrl.sys2frd(Pnz[i], freq)
print('Plant model was set.')

# Design PD controller
freq1 = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
freq2 = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
zeta2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Cz_PD = [[] for i in range(axis_num)]
Cz_PD_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    Cz_PD[i] = ctrl.pd(freq1[i], freq2[i], zeta2[i], M[i], C[i], K[i], Ts)
    Cz_PD_frd[i] = ctrl.sys2frd(Cz_PD[i], freq)
print('PD controller was designed.')

# Design PID controller
freq1 = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
zeta1 = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
freq3 = [25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
freq4 = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
Cz_PID = [[] for i in range(axis_num)]
Cz_PID_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    freq2 = np.sqrt(freq3[i] * freq4[i])
    zeta2 = 0.5 * (freq3[i] + freq4[i]) / freq2
    Cz_PID[i] = ctrl.pid(freq1[i], zeta1[i], freq2, zeta2, M[i], C[i], K[i], Ts)
    Cz_PID_frd[i] = ctrl.sys2frd(Cz_PID[i], freq)
print('PID controller was designed.')

# Design PI velocity controller
freq1 = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
zeta1 = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
Cz_PI = [[] for i in range(axis_num)]
Cz_PI_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    Cz_PI[i] = ctrl.pi(freq1[i], zeta1[i], M[i], C[i], Ts)
    Cz_PI_frd[i] = ctrl.sys2frd(Cz_PI[i], freq)
print('PI velocity controller was designed.')

# Design notch filters
freqNF = [[0.0, 375],
          [0.0, 375],
          [0.0, 375],
          [100, 375],
          [100, 375],
          [100, 375]]
zetaNF = [[0.1, 0.3],
          [0.1, 0.3],
          [0.1, 0.3],
          [0.1, 0],
          [0.1, 0],
          [0.1, 0]]
depthNF = [[0.01, 0.01],
           [0.01, 0.01],
           [0.01, 0.01],
           [0.01, 0],
           [0.01, 0],
           [0.01, 0]]
if np.array_equal(np.array(freqNF).size, np.array(zetaNF).size, np.array(depthNF).size) == False:
    print('NF parameter size is invalid!')
NFz = [[] for i in range(axis_num)]
NFz_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    NFz[i] = ctrl.nf(freqNF[i], zetaNF[i], depthNF[i], Ts)
    NFz_frd[i] = 1.0
    for k in range(len(freqNF[i])):
        NFz_frd[i] *= ctrl.sys2frd(NFz[i][k], freq)
print('Notch filters were designed.')

# Design resonant filters
print('Getting measurement data...')
measfileName = 'data/freq_resp_2mass_20250902.csv'
# Frequency response
Pmeas_frd, coh = meas.measdata2frd(measfileName, 'ServoOutN[0]', 'ActPosUm[0]', 'FlagInject', freq, 1., 1.e-6, 8, 0.8)
L_frd_nf = [[] for i in range(axis_num)]
S_frd_nf = [[] for i in range(axis_num)]
T_frd_nf = [[] for i in range(axis_num)]
for i in range(axis_num):
    L_frd_nf[i] = Pmeas_frd * Cz_PID_frd[i] * NFz_frd[i]
    S_frd_nf[i] = 1/(1 + L_frd_nf[i])
    T_frd_nf[i] = 1 - S_frd_nf[i]

freqRF = [[10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 90.0],
          [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
zetaRF = [[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
          [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
          [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
          [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
          [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
          [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]
depthRF = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
if np.array_equal(np.array(freqRF).size, np.array(zetaRF).size, np.array(depthRF).size) == False:
    print('RF parameter size is invalid!')
RFz = [[] for i in range(axis_num)]
RFz_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    RFz[i] = ctrl.rfopt(freqRF[i], zetaRF[i], depthRF[i], T_frd_nf[i], Ts)
    RFz_frd[i] = 0.0
    for k in range(len(freqRF[i])):
        RFz_frd[i] += ctrl.sys2frd(RFz[i][k], freq)
print('Resonant filters were designed.')

# Design DOB for FB
M_dob = M
C_dob = C
K_dob = K
freq_dob = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
zeta_dob = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
DOBfbu = [[] for i in range(axis_num)]
DOBfby = [[] for i in range(axis_num)]
DOBfbu_frd = [[] for i in range(axis_num)]
DOBfby_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    DOBfbu[i], DOBfby[i] = ctrl.dob(freq_dob[i], zeta_dob[i], M_dob[i], C_dob[i], K_dob[i], Ts)
    DOBfbu_frd[i] = ctrl.sys2frd(DOBfbu[i], freq)
    DOBfby_frd[i] = ctrl.sys2frd(DOBfby[i], freq)
print('DOB was designed.')

# Design DOB for detection
freq_dob = [300.0, 300.0, 300.0, 300.0, 300.0, 300.0]
zeta_dob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DOBestu = [[] for i in range(axis_num)]
DOBesty = [[] for i in range(axis_num)]
DOBestu_frd = [[] for i in range(axis_num)]
DOBesty_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    DOBestu[i], DOBesty[i] = ctrl.dob(freq_dob[i], zeta_dob[i], M[i], C[i], K[i], Ts)
    DOBestu_frd[i] = ctrl.sys2frd(DOBestu[i], freq)
    DOBesty_frd[i] = ctrl.sys2frd(DOBesty[i], freq)
print('DOB was designed.')

# Design ZPETC
Czpetc = [[] for i in range(axis_num)]
Nzpetc = [[] for i in range(axis_num)]
Czpetc_frd = [[] for i in range(axis_num)]
lead_frd = [[] for i in range(axis_num)]
for i in range(axis_num):
    Czpetc[i], Nzpetc[i] = ctrl.zpetc(Pnz[i], Ts)
    Czpetc_frd[i] = ctrl.sys2frd(Czpetc[i], freq)
    lead_frd[i] = ctrl.sys2frd(z**Nzpetc[i], freq)
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
Cz_Hap = ctrl.pd(freq1, freq2, zeta2, M_dob[0], C_dob[0], K_dob[0], Ts)
Cz_Hap_frd = ctrl.sys2frd(Cz_Hap, freq)
print('Haptics controller was designed.')

print('Creating parameter set Cpp and header files...')
Pmechz_axes = [Pmechz[i] for i in range(axis_num)]
Cz_PID_axes = [Cz_PID[i] for i in range(axis_num)]
Cz_PD_axes = [Cz_PD[i] for i in range(axis_num)]
Cz_PI_axes = [Cz_PI[i] for i in range(axis_num)]
NFz_axes = [[NFz[i][k] for k in range(len(NFz[i]))] for i in range(axis_num)]
RFz_axes = [[RFz[i][k] for k in range(len(RFz[i]))] for i in range(axis_num)]
DOBfbu_axes = [DOBfbu[i] for i in range(axis_num)]
DOBfby_axes = [DOBfby[i] for i in range(axis_num)]
DOBestu_axes = [DOBestu[i] for i in range(axis_num)]
DOBesty_axes = [DOBesty[i] for i in range(axis_num)]
Czpetc_axes = [Czpetc[i] for i in range(axis_num)]
ImpModel_axes = [ImpModel for i in range(axis_num)]
Cz_Hap_axes = [Cz_Hap for i in range(axis_num)]

ctrl.makeprmset(srcpathName, ftype, axis_num, len(NFz[0]), len(RFz[0]))
ctrl.defprmset(Pmechz_axes, 'gstModelInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(Cz_PID_axes, 'gstPIDInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(Cz_PD_axes, 'gstPDInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(Cz_PI_axes, 'gstPIInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(NFz_axes, 'gstNFInf[AXIS_NUM][NF_NUM]', srcpathName, ftype)
ctrl.defprmset(RFz_axes, 'gstRFInf[AXIS_NUM][RF_NUM]', srcpathName, ftype)
ctrl.defprmset(DOBfbu_axes, 'gstDOBfbuInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(DOBfby_axes, 'gstDOBfbyInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(DOBestu_axes, 'gstDOBestuInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(DOBesty_axes, 'gstDOBestyInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(Czpetc_axes, 'gstZPETInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(ImpModel_axes, 'gstImpInf[AXIS_NUM]', srcpathName, ftype)
ctrl.defprmset(Cz_Hap_axes, 'gstHapInf[AXIS_NUM]', srcpathName, ftype)

print('Finished.')
