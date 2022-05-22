pylib-sakata User's Manual version-0.1.6
===

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [pylib-sakata User's Manual version-0.1.6](#pylib-sakata-users-manual-version-016)
- [1. Introduction](#1-introduction)
- [2. Environment Setup](#2-environment-setup)
  - [2.1. Installation of Python](#21-installation-of-python)
  - [2.2. Installation  of required Python libraries](#22-installation--of-required-python-libraries)
  - [2.3. Installation of pylib-sakata](#23-installation-of-pylib-sakata)
  - [2.4. Installation of IDE for Python](#24-installation-of-ide-for-python)
    - [2.4.1. Visual Studio Code (VSCode)](#241-visual-studio-code-vscode)
      - [2.4.1.1. Installation of VSCode](#2411-installation-of-vscode)
      - [2.4.1.2. Initial setting of VSCode](#2412-initial-setting-of-vscode)
    - [2.4.2. Spyder](#242-spyder)
      - [2.4.2.1. Installation of Spyder](#2421-installation-of-spyder)
      - [2.4.2.2. Initial setting of Spyder](#2422-initial-setting-of-spyder)
    - [2.4.3. PyCharm](#243-pycharm)
      - [2.4.3.1. Installation of PyCharm](#2431-installation-of-pycharm)
      - [2.4.3.2. Initial setting of PyCharm](#2432-initial-setting-of-pycharm)
    - [2.4.4. Comparison between  VSCode and Spyder and PyCharm](#244-comparison-between--vscode-and-spyder-and-pycharm)
  - [2.5. Getting started](#25-getting-started)
- [3. pylib_sakata.ctrl](#3-pylib_sakatactrl)
  - [3.1. ZpkModel](#31-zpkmodel)
  - [3.2. tf](#32-tf)
  - [3.3. ss](#33-ss)
  - [3.4. zpk](#34-zpk)
  - [3.5. tf2ss](#35-tf2ss)
  - [3.6. tf2zpk](#36-tf2zpk)
  - [3.7. ss2tf](#37-ss2tf)
  - [3.8. ss2zpk](#38-ss2zpk)
  - [3.9. zpk2tf](#39-zpk2tf)
  - [3.10. zpk2ss](#310-zpk2ss)
  - [3.11. sys2frd](#311-sys2frd)
  - [3.12. feedback](#312-feedback)
  - [3.13. frdfeedback](#313-frdfeedback)
  - [3.14. c2d](#314-c2d)
  - [3.15. pi](#315-pi)
  - [3.16. pd](#316-pd)
  - [3.17. pid](#317-pid)
  - [3.18. pl1st](#318-pl1st)
  - [3.19. pl2nd](#319-pl2nd)
  - [3.20. lpf1st](#320-lpf1st)
  - [3.21. lpf2nd](#321-lpf2nd)
  - [3.22. hpf1st](#322-hpf1st)
  - [3.23. hpf2nd](#323-hpf2nd)
  - [3.24. nf](#324-nf)
  - [3.25. pf](#325-pf)
  - [3.26. pfoptparam](#326-pfoptparam)
  - [3.27. pfopt](#327-pfopt)
  - [3.28. dob](#328-dob)
  - [3.29. zpetc](#329-zpetc)
  - [3.30. filt](#330-filt)
  - [3.31. minreal](#331-minreal)
- [4. pylib_sakata.fft](#4-pylib_sakatafft)
  - [4.1. FreqResp](#41-freqresp)
  - [4.2. fft](#42-fft)
  - [4.3. fft_ave](#43-fft_ave)
  - [4.4. tfestimate](#44-tfestimate)
  - [4.5. frdresize](#45-frdresize)
  - [4.6. frdsim](#46-frdsim)
- [5. pylib_sakta.meas](#5-pylib_saktameas)
  - [5.1. MeasData](#51-measdata)
  - [5.2. getcsvdata](#52-getcsvdata)
  - [5.3. gettxtdata](#53-gettxtdata)
  - [5.4. getmatdata](#54-getmatdata)
  - [5.5. getdata](#55-getdata)
  - [5.6. getdataindex](#56-getdataindex)
  - [5.7. measdata2frd](#57-measdata2frd)
- [6. pylib_sakata.traj](#6-pylib_sakatatraj)
  - [6.1. TrajInf](#61-trajinf)
  - [6.2. traj4th](#62-traj4th)
- [7. pylib_sakata.plot](#7-pylib_sakataplot)
  - [7.1. plot_xy](#71-plot_xy)
  - [7.2. plot_tf](#72-plot_tf)
  - [7.3. plot_tffrd](#73-plot_tffrd)
  - [7.4. plot_nyquist](#74-plot_nyquist)
  - [7.5. plot_nyquist_assistline](#75-plot_nyquist_assistline)
  - [7.6. makefig](#76-makefig)
  - [7.7. savefig](#77-savefig)
  - [7.8. showfig](#78-showfig)
- [8. pylib_sakata.init](#8-pylib_sakatainit)
  - [8.1. close_all](#81-close_all)
  - [8.2. clear_all](#82-clear_all)

<!-- /code_chunk_output -->

# 1. Introduction

The pylib-sakata package is a set of python classes and functions that make the python-control package more convenient. This package provide practical level's tools to design controls and  to analysis of performance and stability of SISO LTI systems. These development environments are available on free.

# 2. Environment Setup

## 2.1. Installation of Python
Python installation exe file can be downloaded [HERE](https://www.python.org/downloads/) for Windows. Check "Add Python 3.x to PATH" when you install Python.

## 2.2. Installation  of required Python libraries

The pylib-sakata package requires [numpy](http://www.numpy.org), [scipy](http://www.scipy.org), [matplotlib](https://matplotlib.org), [pandas](https://pandas.pydata.org/), and [python-control](https://github.com/python-control/python-control). In addition, some routines require the [slycot](https://github.com/python-control/Slycot) library in order to implement more advanced features. 
First, pip should be upgraded by the following command on the command prompt for Windows OS.
```shell
python -m pip install --upgrade pip
```
For Linux OS, pip can be upgraded by the following command on the shell.
```shell
$ sudo pip3 install --upgrade pip
```
The required python libraries can be installed using pip as the following commands on the command prompt or on the shell.
```shell
pip install numpy
pip install scipy
pip install matplotlib
pip install pandas
pip install control
```
If those libraries can not be installed, please use pip3 instead of pip.

## 2.3. Installation of pylib-sakata

The [pylib-sakata](https://github.com/Koichi-Sakata/pylib_sakata) package can be installed using pip using pip as following on the command prompt or on the shell.

```shell
pip install pylib-sakata
```

## 2.4. Installation of IDE for Python

There are three recommended IDEs.

### 2.4.1. Visual Studio Code (VSCode)
Although VSCode is actually an editor, it is available as Python IDE due to extensions. It is recommended to install it easily because of the versatility of extensions. The following figure shows VSCode IDE window.

<img src="figure\vscode_disp.png" alt="vscode_disp" style="zoom: 33%;" />

#### 2.4.1.1. Installation of VSCode
The latest version of VSCode installation exe file can be downloaded [HERE](https://code.visualstudio.com/) for Windows OS.
For Linux OS, VSCode can be installed by the following command on the shell.
```shell
$ sudo apt update 
$ sudo apt install code
```

#### 2.4.1.2. Initial setting of VSCode
1. Install Python extension for Visual Studio Code from extensions in the activity bar on the left side.
1. Command (Ctrl+Shift+P) to open the command palette.
1. Select your installed python.exe on  "Python: select interpreter".

### 2.4.2. Spyder
Spyder is an IDE like MATLAB. If you are used to MATLAB, this IDE is recommended. The following figure shows Spyder IDE window.

<img src="figure\spyder_disp.png" alt="spyder_disp" style="zoom: 33%;" />

#### 2.4.2.1. Installation of Spyder
The latest version of Spyder installation exe file can be downloaded [HERE](https://www.spyder-ide.org/) for Windows OS.
For Linux OS, Spyder can be installed by the following command on the shell.

```shell
$ sudo apt update 
$ sudo apt install spyder3
```

#### 2.4.2.2. Initial setting of Spyder
1. Open Tools>Preferences>Python interpreter.
1. Check "Use the following Python interpreter."
1. Enter the path of your installed python.exe.
1. Install spyder-kernels for your python version by the following command `pip install spyder-kernels==2.0.*` on the command prompt or on the shell.
1. Reboot your PC.

### 2.4.3. PyCharm
Professional version is charged. Community version is free. Community version is enough for control system development. The following figure shows PyCharm window.

<img src="figure\pycharm_disp.png" alt="spyder_disp" style="zoom: 33%;" />

#### 2.4.3.1. Installation of PyCharm
The latest version of PyCharm installation exe file can be downloaded [HERE](https://www.jetbrains.com/pycharm/download/). 

#### 2.4.3.2. Initial setting of PyCharm
It is not necessary.

### 2.4.4. Comparison between  VSCode and Spyder and PyCharm

|                    |        VSCode       |      Spyder       |     PyCharm       |
| ------------------ | :-----------------: | :---------------: | :---------------: |
| Cost               |        Free         |       Free        |    Free/Paid      |
| Boot time          |        Fast         |       Slow        |       Normal      |
| Processing speed   |        Fast         |       Normal      |       Normal      |
| REPL mode          | need to switch REPL terminal  | Available | Available |
| Variable Explorer  | Available on Jupyter terminal | Available (Class variables are invisible.) | Available |

Spyder is more suitable than VSCode for early debugging. On the other hand, VSCode is stress-free than Spyder for late debugging you do not need to check inter states of variables. PyCharm is generally well-balanced. My recommendation is PyCharm.

## 2.5. Getting started
The pylib-sakata package can be imported as follows.
```python
>>> import pylib_sakata
```
This package consists of six modules as follows.
1. **ctrl**: to design controllers and filters
1. **fft**: to analysis FFT
1. **meas**: to treat measurement data
1. **traj**: to generate target trajectory
1. **plot**: to plot figures
1. **init**: to initialize current variable

These modules can be imported individually as follows.
```python
>>> from pylib_sakata import ctrl
>>> from pylib_sakata import fft
>>> from pylib_sakata import meas
>>> from pylib_sakata import traj
>>> from pylib_sakata import plot
>>> from pylib_sakata import init
```

Example codes are [HERE](https://github.com/Koichi-Sakata/pylib_sakata/tree/main/example).

# 3. pylib_sakata.ctrl

## 3.1. ZpkModel

class pylib_sakata.ctrl.**ZpkModel**(*z, p, k, dt=0*)

- Parameters:
  - z: zeros array of the LTI model
  - p: poles array of the LTI model
  - k: gain of the LTI model. Note: the gain is not system dc gain but coefficient of monic polynomials. This is different from the definition of dc gain in TransferFunction class.
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.

**Examples**
```python
>>> ctrl.ZpkModel([-1., -2.], [-3., -4., -5.], 2.)

      (s+1)(s+2)
2 * ---------------
    (s+3)(s+4)(s+5)
```
If you set same zeros and poles, the pole-zero cancellation is automatically done as following. Unnecessary increase of the system order can be avoided.
```python
>>> ctrl.ZpkModel([0.1, 0.3, 0.4], [0.3, 0.4, 0.5], 2., 0.001)
The common pole-zeros of the zpk model have been deleted.

    (z-0.1)
2 * -------
    (z-0.5)

dt = 0.001
```

**Methods**
- \__**neg**__()
	Negate a zpk model.

- \__**add**__(*other*)
	Add two zpk models (parallel connection).

- \__**radd**__(*other*)
	Right add two zpk models (parallel connection).

- \__**sub**__(*other*)
	Subtract two zpk models.

- \__**rsub**__(*other*)
	Right subtract two zpk models.

- \__**mul**__(*other*)
	Multiply two zpk models (serial connection).

- \__**rmul**__(*other*)
	Right multiply two zpk models (serial connection).

- \__**truediv**__(*other*)
	Divide two zpk models.

- \__**rtruediv**__(*other*)
	Right divide two zpk models.

- \__**pow**__(*other*)
	A zpk model to the power of x.

- **feedback**(*other=1, sys='S'*)
	Calculate the feedback system that consist of two zpk model (P: self and C: other).
	sys: FB type (Optional), Default: 'S', Set in 'S': sensitivity function, 'T': complementary sensitivity function, 'SP': response from input disturbance to output

## 3.2. tf

pylib_sakata.ctrl.**tf**(*num, den, dt=0*)

This function calls **tf** function in control library.

- Parameters:
  - num: polynomial coefficients of the numerator of the LTI model
  - den: polynomial coefficients of the denominator of the LTI model
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
- Returns:
  - out: instance of TransferFuntion class

**Examples**
```python
>>> print(ctrl.tf([1., 2.], [3., 4., 5.]))

     s + 2
---------------
3 s^2 + 4 s + 5
```

## 3.3. ss

pylib_sakata.ctrl.**ss**(*A, B, C, D, dt=0*)

This function calls **ss** function in control library.

- Parameters:
  - A, B, C, D: state space matrices
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
- Returns:
  - out: instance of StateSpace class

**Examples**
```python
>>> print(ctrl.ss("1. -2.; 3. -4.", "5.; 7.", "6. 8.", "9."))
A = [[ 1. -2.]
     [ 3. -4.]]

B = [[5.]
     [7.]]

C = [[6. 8.]]

D = [[9.]]
```

## 3.4. zpk

pylib_sakata.ctrl.**zpk**(*z, p, k, dt=0*)

- Parameters:
  - A, B, C, D: state space matrices
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
- Returns:
  - out: instance of ZpkModel class

**Examples**
```python
>>> ctrl.zpk([-1., -2.], [-3., -4., -5.], 2.)

      (s+1)(s+2)
2 * ---------------
    (s+3)(s+4)(s+5)
```

## 3.5. tf2ss

pylib_sakata.ctrl.**tf2ss**(*tf, form='reachable'*)

- Parameters:
  - tf: instance of TransferFunction class
  - form: canonical form (Optional), Default: reachable. You can select the canonical form in reachable, observable, and modal.
- Returns:
  - out: instance of StateSpace class

**Examples**
```python
>>> Sys_tf = ctrl.tf([1., 2.], [3., 4., 5.])
>>> print(ctrl.tf2ss(Sys_tf))
A = [[-1.33333333 -1.66666667]
     [ 1.          0.        ]]

B = [[1.]
     [0.]]

C = [[0.33333333 0.66666667]]

D = [[0.]]
```

## 3.6. tf2zpk

pylib_sakata.ctrl.**tf2zpk**(*tf*)

- Parameters:
  - tf: instance of TransferFunction class
- Returns:
  - out: instance of ZpkModel class

**Examples**
```python
>>> Sys_tf = ctrl.tf([1., 2.], [3., 4., 5.])
>>> ctrl.tf2zpk(Sys_tf)

                       (s+2)
0.3333 * ----------------------------------
         (s+0.6667+1.106j)(s+0.6667-1.106j)
```


## 3.7. ss2tf

pylib_sakata.ctrl.**ss2tf**(*ss*)

- Parameters:
  - ss: instance of StateSpace class
- Returns:
  - out: instance of TransferFunction class

**Examples**
```python
>>> Sys_ss = ctrl.ss("1. -2.; 3. -4.", "5.; 7.", "6. 8.", "9.")
>>> print(ctrl.ss2tf(Sys_ss))

9 s^2 + 113 s + 118
-------------------
   s^2 + 3 s + 2
```

## 3.8. ss2zpk

pylib_sakata.ctrl.**ss2zpk**(*ss*)

- Parameters:
  - ss: instance of StateSpace class
- Returns:
  - out: instance of ZpkModel class

**Examples**
```python
>>> Sys_ss = ctrl.ss("1. -2.; 3. -4.", "5.; 7.", "6. 8.", "9.")
>>> ctrl.ss2zpk(Sys_ss)

    (s+11.41)(s+1.149)
9 * ------------------
        (s+2)(s+1)
```

## 3.9. zpk2tf

pylib_sakata.ctrl.**zpk2tf**(*zpk*)

- Parameters:
  - zpk: instance of ZpkModel class
- Returns:
  - out: instance of TransferFunction class

**Examples**
```python
>>> Sys_zpk = ctrl.zpk([-1., -2.], [-3., -4., -5.], 2.)
>>> print(ctrl.zpk2tf(Sys_zpk))

    2 s^2 + 6 s + 4
------------------------
s^3 + 12 s^2 + 47 s + 60
```

## 3.10. zpk2ss

pylib_sakata.ctrl.**zpk2ss**(*zpk, form='reachable'*)

- Parameters:
  - zpk: instance of ZpkModel class
  - form: canonical form (Optional), Default: reachable. You can select the canonical form in reachable, observable, and modal.
- Returns:
  - out: instance of StateSpace class

**Examples**
```python
>>> Sys_zpk = ctrl.zpk([-1., -2.], [-3., -4., -5.], 2.)
>>> print(ctrl.zpk2ss(Sys_zpk))
A = [[-12. -47. -60.]
     [  1.   0.   0.]
     [  0.   1.   0.]]

B = [[1.]
     [0.]
     [0.]]

C = [[2. 6. 4.]]

D = [[0.]]
```

## 3.11. sys2frd

pylib_sakata.ctrl.**sys2frd**(*sys, freq*)

- Parameters:
  - sys: LTI model (StateSpace or TransferFunction or ZpkModel)
  - freq: 1-D array frequency data [Hz]
- Returns:
  - freqresp: instance of FreqResp class (Refer [4.1. FreqResp](#41-freqresp).)

**Examples**
```python
>>> import numpy as np
>>> freq = np.logspace(np.log10(1.), np.log10(1000.), 10000, base=10)
>>> Sys_tf = ctrl.tf([1., 2.], [3., 4., 5.])
>>> ctrl.sys2frd(Sys_tf, freq)

freq = array([   1.            1.00069108    1.00138264 ...  998.61926487  999.30939397
 1000.        ])
resp = array([-5.10821217e-03-5.65218353e-02j -5.10191815e-03-5.64779702e-02j
 -5.09563061e-03-5.64341455e-02j ... -5.64453060e-09-5.31250027e-05j
 -5.63673700e-09-5.30883142e-05j -5.62895416e-09-5.30516511e-05j])
```
```python
>>> Sys_zpk = ctrl.zpk([-1., -2.], [-3., -4., -5.], 2.)
>>> ctrl.sys2frd(Sys_zpk, freq)

freq = array([   1.            1.00069108    1.00138264 ...  998.61926487  999.30939397
 1000.        ])
resp = array([1.89109027e-01-0.06951671j 1.89033162e-01-0.06962001j
 1.88957165e-01-0.0697232j  ... 4.57206511e-07-0.00031875j
 4.56575231e-07-0.00031853j 4.55944822e-07-0.00031831j])
```

## 3.12. feedback

pylib_sakata.ctrl.**feedback**(*sysP, sysC, sys='S'*)

This function is for calculating the feedback system that consist of two LTI model (P: plant and C: controller).

- Parameters:
  - sysP: LTI model (StateSpace or TransferFunction or ZpkModel) of the plant
  - sysC: LTI model (StateSpace or TransferFunction or ZpkModel) of the controller
  - sys: FB type (Optional), Default: 'S', Set in 'S': sensitivity function, 'T': complementary sensitivity function, 'SP': response from input 
- Returns:
  - out: LTI model (StateSpace or TransferFunction or ZpkModel) of the feedback system

**Examples**
```python
>>> P_tf = ctrl.tf([1.], [1., 2., 0.])
>>> C_tf = ctrl.tf([2., 4., 1.], [1., 2., 0.])
>>> print(ctrl.feedback(P_tf, C_tf))

     s^4 + 4 s^3 + 4 s^2
-----------------------------
s^4 + 4 s^3 + 6 s^2 + 4 s + 1
```
```python
>>> P_zpk = ctrl.zpk([], [0., -2.])
>>> C_zpk = ctrl.zpk([-1.7071067811865475, -0.29289321881345254], [0., -2.], 2.)
>>> ctrl.feedback(P_zpk, C_zpk)

                     s(s+2)(s+2)s
1 * -----------------------------------------------
    (s+1)(s+1-0.0001508j)(s+1+0.0001508j)(s+0.9998)
```

## 3.13. frdfeedback

pylib_sakata.ctrl.**frdfeedback**(*frdP, frdC, sys='S'*)

This function is for calculating the feedback system that consist of two freqresp (P: plant and C: controller).

- Parameters:
  - frdP: 1-D array complex data of the frequency response of the plant
  - frdC: 1-D array complex data of the frequency response of the controller
  - sys: FB type (Optional), Default: 'S', Set in 'S': sensitivity function, 'T': complementary sensitivity function, 'SP': response from input 
- Returns:
  - out: 1-D array complex data of the frequency response of the feedback system


**Examples**
```python
>>> import numpy as np
>>> freq = np.logspace(np.log10(1.), np.log10(1000.), 10000, base=10)
>>> P_tf = ctrl.tf([1.], [1., 2., 0.])
>>> C_tf = ctrl.tf([2., 4., 1.], [1., 2., 0.])
>>> P_frd = ctrl.tf2frd(P_tf, freq)
>>> C_frd = ctrl.tf2frd(C_tf, freq)
>>> ctrl.frdfeedback(P_frd, C_frd)
array([1.04746047+1.56990662e-02j, 1.04739933+1.56671729e-02j,
       1.04733826+1.56353434e-02j, ..., 1.00000005+1.61927479e-11j,
       1.00000005+1.61592227e-11j, 1.00000005+1.61257668e-11j])
```

## 3.14. c2d

pylib_sakata.ctrl.**c2d**(*sysC, dt, method='tustin'*)

The matched method of **c2d** in control library does not supported for pure integrals and pure derivatives because of dc gain inf error. This function solved this problem.

- Parameters:
  - sysC: continuous time LTI model (StateSpace or TransferFunction or ZpkModel) of the plant
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value > 0.
  - method: discretized method (Optional), Default: 'tustin', set a method in 'tustin', 'matched', 'zoh', and etc.
- Returns:
  - out: discrete time LTI model (StateSpace or TransferFunction or ZpkModel) of the feedback system

**Examples**
```python
>>> C_tf = ctrl.tf([1., 3., 2.], [1., 3., 0.])
>>> print(ctrl.c2d_matched(C_tf, 0.001))

z^2 - 1.997 z + 0.997
---------------------
z^2 - 1.997 z + 0.997

dt = 0.001
```
```python
>>> C_zpk = ctrl.zpk([-1., -2.,], [0., -3.,], 1.)
>>> ctrl.c2d_matched(C_zpk, 0.001)

    (z-0.999)(z-0.998)
1 * ------------------
      (z-1)(z-0.997)

dt = 0.001
```

## 3.15. pi

pylib_sakata.ctrl.**pi**(*freq, zeta, L, R, dt=None, method='tustin'*)

This function is for design of a PI controller.
$$
C_{PI}(s) = K_P + \frac{K_I}{s} = \frac{b_1s+b_0}{s}
$$
$$
P(s) = \frac{1}{Ls+R}
$$

- Parameters:
  - freq: frequency[Hz] of the pole pair of the feedback system with the PI controller
  - zeta: damping of the pole pair of the feedback system with the PI controller
  - L: inductance[H] of the plant
  - R: resistance[$\Omega$] of the plant
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the PI controller

**Examples**
```python
>>> print(ctrl.pi(1000., 0.7, 0.1, 10.))

869.6 s + 3.948e+06
-------------------
         s
```
```python
>>> print(ctrl.pi(1000., 0.7, 0.1, 10., 0.001))

2844 z + 1104
-------------
    z - 1

dt = 0.001
```

## 3.16. pd

pylib_sakata.ctrl.**pd**(*freq1, freq2, zeta2, M, C, K, dt=None, method='tustin'*)

This function is for design of a PD controller.
$$
C_{PID}(s) = K_P + \frac{K_D s}{\tau_D s+1} = \frac{b_1s+b_0}{s+a_1}
$$
$$
P(s) = \frac{1}{Ms^2+Cs+K}
$$

- Parameters:
  - freq1: frequency[Hz] of the first pole of the feedback system with the PD controller
  - freq2: frequency[Hz] of the second pole pair of the feedback system with the PD controller
  - zeta2: damping of the second pole pair of the feedback system with the PD controller
  - M: mass[kg] of the plant
  - C: viscosity[N/(m/s)] of the plant
  - K: stiffness[N/m] of the plant
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the PD controller

**Examples**
```python
>>> print(ctrl.pd(10., 10., 0.7, 2., 10., 0.))

1.749e+04 s + 4.961e+05
-----------------------
       s + 145.8
```
```python
>>> print(ctrl.pd(10., 10., 0.7, 2., 10., 0., 0.001))

1.653e+04 z - 1.607e+04
-----------------------
      z - 0.8641

dt = 0.001
```

## 3.17. pid

pylib_sakata.ctrl.**pid**(*freq1, zeta1, freq2, zeta2, M, C, K, dt=None, method='tustin'*)

This function is for design of a PID controller.
$$
C_{PID}(s) = K_P + \frac{K_I}{s} + \frac{K_D s}{\tau_D s+1} = \frac{b_2s^2+b_1s+b_0}{s^2+a_1s}
$$
$$
P(s) = \frac{1}{Ms^2+Cs+K}
$$

- Parameters:
  - freq1: frequency[Hz] of the first pole pair of the feedback system with the PID controller
  - zeta1: damping of the first pole pair of the feedback system with the PID controller
  - freq2: frequency[Hz] of the second pole pair of the feedback system with the PID controller
  - zeta2: damping of the second pole pair of the feedback system with the PID controller
  - M: mass[kg] of the plant
  - C: viscosity[N/(m/s)] of the plant
  - K: stiffness[N/m] of the plant
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the PID controller

**Examples**
```python
>>> print(ctrl.pid(10., 1., 10., 0.7, 2., 10., 0.))

3.581e+04 s^2 + 1.687e+06 s + 3.117e+07
---------------------------------------
             s^2 + 208.6 s
```
```python
>>> print(ctrl.pid(10., 1., 10., 0.7, 2., 10., 0., 0.001))

3.32e+04 z^2 - 6.485e+04 z + 3.167e+04
--------------------------------------
        z^2 - 1.811 z + 0.8111

dt = 0.001
```

## 3.18. pl1st

pylib_sakata.ctrl.**pl1st**(*freq1, freq2, dt=None, method='tustin'*)

This function is for design of a first order phase lead filter.
$$
F_{PL}(s) = \frac{f_2}{f_1} * \frac{s+2\pi f_1}{s+2\pi f_2}
$$

- Parameters:
  - freq1: frequency[Hz] of numerator of the phase lead filter
  - freq2: frequency[Hz] of denominator of the phase lead filter
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the phase lead filter

**Examples**
```python
>>> ctrl.tf2zpk(ctrl.pl1st(50., 100.))

    (s+314.2)
2 * ---------
    (s+628.3)
```
```python
>>> ctrl.tf2zpk(ctrl.pl1st(50., 100., 0.001))

        (z-0.7285)
1.761 * ----------
        (z-0.5219)

dt = 0.001
```

## 3.19. pl2nd

pylib_sakata.ctrl.**pl2nd**(*freq1, zeta1, freq2, zeta2, dt=None, method='tustin'*)

This function is for design of a second order phase lead filter.
$$
F_{PL^2}(s) = \left( \frac{f_2}{f_1} \right)^2 * \frac{s^2+2\zeta_1 (2\pi f_1)s+(2\pi f_1)^2}{s^2+2\zeta_2 (2\pi f_2)s+(2\pi f_2)^2}
$$

- Parameters:
  - freq1: frequency[Hz] of numerator of the phase lead filter
  - zeta1: damping of numerator of the phase lead filter
  - freq2: frequency[Hz] of denominator of the phase lead filter
  - zeta2: damping of denominator of the phase lead filter
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the phase lead filter

**Examples**
```python
>>> print(ctrl.pl2nd(50., 0.7, 100., 0.7))

2 s^2 + 879.6 s + 1.974e+05
---------------------------
 s^2 + 879.6 s + 3.948e+05
```
```python
>>> print(ctrl.pl2nd(50., 0.7, 100., 0.7, 0.001))

1.618 z^2 - 2.536 z + 1.046
---------------------------
  z^2 - 1.172 z + 0.4283

dt = 0.001
```

## 3.20. lpf1st

pylib_sakata.ctrl.**lpf1st**(*freq, dt=None, method='tustin'*)

This function is for design of a first order low pass filter.
$$
F_{LP}(s) = \frac{2\pi f}{s+2\pi f}
$$

- Parameters:
  - freq: frequency[Hz] of the low pass filter
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the low pass filter

**Examples**
```python
>>> print(ctrl.lpf1st(100.))

  628.3
---------
s + 628.3
```
```python
>>> print(ctrl.lpf1st(100., 0.001))

0.2391 z + 0.2391
-----------------
   z - 0.5219

dt = 0.001
```

## 3.21. lpf2nd

pylib_sakata.ctrl.**lpf2nd**(*freq, zeta, dt=None, method='tustin'*)

This function is for design of a second order low pass filter.
$$
F_{LP^2}(s) = \frac{(2\pi f)^2}{s^2+2\zeta (2\pi f)s+(2\pi f)^2}
$$

- Parameters:
  - freq: frequency[Hz] of the low pass filter
  - zeta1: damping of the low pass filter
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the low pass filter

**Examples**
```python
>>> print(ctrl.lpf2nd(100., 0.7))

        3.948e+05
-------------------------
s^2 + 879.6 s + 3.948e+05
```
```python
>>> print(ctrl.lpf2nd(100., 0.7, 0.001))

0.06415 z^2 + 0.1283 z + 0.06415
--------------------------------
     z^2 - 1.172 z + 0.4283

dt = 0.001
```

## 3.22. hpf1st

pylib_sakata.ctrl.**hpf1st**(*freq, dt=None, method='tustin'*)

This function is for design of a first order high pass filter.
$$
F_{HP}(s) = 1 - F_{LP}(s) = \frac{s}{s+2\pi f}
$$

- Parameters:
  - freq: frequency[Hz] of the high pass filter
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the high pass filter

**Examples**
```python
>>> print(ctrl.hpf1st(100.))

    s
---------
s + 628.3
```
```python
>>> print(ctrl.hpf1st(100., 0.001))

0.7609 z - 0.7609
-----------------
   z - 0.5219

dt = 0.001
```

## 3.23. hpf2nd

pylib_sakata.ctrl.**hpf2nd**(*freq, zeta, dt=None, method='tustin'*)

This function is for design of a second order high pass filter.
$$
F_{HP^2}(s) = 1 - F_{LP^2}(s) = \frac{s^2+2\zeta (2\pi f)s}{s^2+2\zeta (2\pi f)s+(2\pi f)^2}
$$

- Parameters:
  - freq: frequency[Hz] of the high pass filter
  - zeta1: damping of the high pass filter
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: instance of TransferFunction class of the high pass filter

**Examples**
```python
>>> print(ctrl.hpf2nd(100., 0.7))

      s^2 + 879.6 s
-------------------------
s^2 + 879.6 s + 3.948e+05
```
```python
>>> print(ctrl.hpf2nd(100., 0.7, 0.001))

0.9358 z^2 - 1.3 z + 0.3641
---------------------------
  z^2 - 1.172 z + 0.4283

dt = 0.001
```

## 3.24. nf

pylib_sakata.ctrl.**nf**(*freq, zeta, depth, dt=None, method='matched'*)

This function is for design of notch filters.
$$
F_{notch}(s) = \frac{s^2+2d\zeta (2\pi f)s+(2\pi f)^2}{s^2+2\zeta (2\pi f)s+(2\pi f)^2}
$$

- Parameters:
  - freq: array of frequency[Hz] of the notch filters
  - zeta: array of damping of the notch filters
  - depth: array of depth of the notch filters (0 < depth < 1)
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'matched', set a method if dt > 0
- Returns:
  - out: array of instance of TransferFunction class of the notch filters

**Examples**
```python
>>> ctrl.nf([100., 200., 300.], [0.02, 0.02, 0.02], [0.01, 0.01, 0.01])
array([TransferFunction(array([1.00000000e+00, 2.51327412e-01, 3.94784176e+05]), array([1.00000000e+00, 2.51327412e+01, 3.94784176e+05])),
       TransferFunction(array([1.00000000e+00, 5.02654825e-01, 1.57913670e+06]), array([1.00000000e+00, 5.02654825e+01, 1.57913670e+06])),
       TransferFunction(array([1.00000000e+00, 7.53982237e-01, 3.55305758e+06]), array([1.00000000e+00, 7.53982237e+01, 3.55305758e+06]))],
      dtype=object)
```
```python
>>> ctrl.nf([100., 200., 300.], [0.02, 0.02, 0.02], [0.01, 0.01, 0.01], 0.001)
array([TransferFunction(array([ 0.9876627 , -1.59787102,  0.9874145 ]), array([ 1.        , -1.59797428,  0.97518046]), 0.001),
       TransferFunction(array([ 0.97553399, -0.6027617 ,  0.97504375]), array([ 1.        , -0.60316088,  0.95097692]), 0.001),
       TransferFunction(array([0.96362486, 0.59532837, 0.96289858]), array([1.        , 0.59447771, 0.92737411]), 0.001)],
      dtype=object)
```

## 3.25. pf

pylib_sakata.ctrl.**pf**(*freq, zeta, k, phi, dt=None, method='tustin'*)

This function is for design of peak filters ([resonant filters](https://ieeexplore.ieee.org/document/4291569)).
$$
F_{peak}(s) = \frac{k(s^2-\phi s)}{s^2+2\zeta (2\pi f)s+(2\pi f)^2}
$$

- Parameters:
  - freq: array of frequency[Hz] of the peak filters
  - zeta: array of damping of the peak filters
  - k: array of peak width of the peak filters
  - phi: array of phase lead of the peak filter
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: array of instance of TransferFunction class of the peak filters

**Examples**
```python
>>> ctrl.pf([2., 3., 5.], [0.001, 0.001, 0.001], [-0.00025695, -0.00049616, 0.0003898], [860.21053991, 633.22924516, -1090.49879949])
>>> ctrl.pf([2., 3., 5.], [0.001, 0.001, 0.001], [-0.00025695, -0.00049616, 0.0003898], [860.21053991, 633.22924516, -1090.49879949])
array([TransferFunction(array([-0.00025695,  0.2210311 ,  0.        ]), array([1.00000000e+00, 2.51327412e-02, 1.57913670e+02])),
       TransferFunction(array([-0.00049616,  0.31418302,  0.        ]), array([1.00000000e+00, 3.76991118e-02, 3.55305758e+02])),
       TransferFunction(array([3.89800000e-04, 4.25076432e-01, 0.00000000e+00]), array([1.00000000e+00, 6.28318531e-02, 9.86960440e+02]))],
      dtype=object)
```
```python
>>> ctrl.pf([2., 3., 5.], [0.001, 0.001, 0.001], [-0.00025695, -0.00049616, 0.0003898], [860.21053991, 633.22924516, -1090.49879949], 0.001)
array([TransferFunction(array([-0.00014643,  0.00051387, -0.00036745]), array([ 1.        , -1.99981696,  0.99997487]), 0.001),
       TransferFunction(array([-0.00033903,  0.00099221, -0.00065318]), array([ 1.        , -1.99960704,  0.9999623 ]), 0.001),
       TransferFunction(array([ 0.00060217, -0.00077938,  0.00017721]), array([ 1.        , -1.9989505 ,  0.99993719]), 0.001)],
      dtype=object)
```

## 3.26. pfoptparam
pylib_sakata.ctrl.**pfoptparam**(freq, zeta, depth, sysT)

This function is for getting parameters of optimized peak filters ([resonant filters](https://ieeexplore.ieee.org/document/4291569)).

- Parameters:
  - freq: array of frequency[Hz] of the peak filters
  - zeta: array of damping of the peak filters
  - depth: array of depth of the peak filters (0 < depth < 1)
  - sysT: LTI model (StateSpace or TransferFunction or ZpkModel or FreqResp) of complementary sensitivity function of the previous feedback system
- Returns:
  - freq: array of frequency[Hz] of the peak filters
  - zeta: array of damping of the peak filters
  - k: array of peak width of the peak filters
  - phi: array of phase lead of the peak filter

**Examples**
```python
>>> Ps = ctrl.tf([1.], [2., 10., 0.])
>>> Cs = ctrl.pid(10., 1., 10., 0.7, 2., 10., 0.)
>>> ctrl.pfoptparam([2., 3., 5.], [0.001, 0.001, 0.001], [0.1, 0.1, 0.1], ctrl.feedback(Ps, Cs, sys='T'))
The common pole-zeros of the zpk model have been deleted.
([2.0, 3.0, 5.0], [0.001, 0.001, 0.001], array([-0.00025695, -0.00049616,  0.0003898 ]), array([  860.21053991,   633.22924516, -1090.49879949]))
```

## 3.27. pfopt

pylib_sakata.ctrl.**pfopt**(*freq, zeta, depth, sysT, dt=None, method='tustin'*)

This function is for design of optimized peak filters ([resonant filters](https://ieeexplore.ieee.org/document/4291569)).

- Parameters:
  - freq: array of frequency[Hz] of the peak filters
  - zeta: array of damping of the peak filters
  - depth: array of depth of the peak filters (0 < depth < 1)
  - sysT: LTI model (StateSpace or TransferFunction or ZpkModel or FreqResp) of complementary sensitivity function of the previous feedback system
  - dt: sampling time of the LTI model (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.
  - method: discretized method (Optional), Default: 'tustin', set a method if dt > 0
- Returns:
  - out: array of instance of TransferFunction class of the peak filters

**Examples**
```python
>>> Ps = ctrl.tf([1.], [2., 10., 0.])
>>> Cs = ctrl.pid(10., 1., 10., 0.7, 2., 10., 0.)
>>> ctrl.pfopt([2., 3., 5.], [0.001, 0.001, 0.001], [0.1, 0.1, 0.1], ctrl.feedback(Ps, Cs, sys='T'))
The common pole-zeros of the zpk model have been deleted.
array([TransferFunction(array([0.00486434, 0.24323591, 0.        ]), array([1.00000000e+00, 2.51327412e-02, 1.57913670e+02])),
       TransferFunction(array([0.00723091, 0.35823306, 0.        ]), array([1.00000000e+00, 3.76991118e-02, 3.55305758e+02])),
       TransferFunction(array([0.01199547, 0.56160871, 0.        ]), array([1.00000000e+00, 6.28318531e-02, 9.86960440e+02]))],
      dtype=object)
```
```python
>>> ctrl.pfopt([2., 3., 5.], [0.001, 0.001, 0.001], [0.1, 0.1, 0.1], ctrl.feedback(Ps, Cs, sys='T'), 0.001)
The common pole-zeros of the zpk model have been deleted.
array([TransferFunction(array([ 0.0049857 , -0.00972818,  0.00474248]), array([ 1.        , -1.99981696,  0.99997487]), 0.001),
       TransferFunction(array([ 0.00740923, -0.01446026,  0.00705103]), array([ 1.        , -1.99960704,  0.9999623 ]), 0.001),
       TransferFunction(array([ 0.01227286, -0.02398427,  0.01171141]), array([ 1.        , -1.9989505 ,  0.99993719]), 0.001)],
      dtype=object)
```

## 3.28. dob
pylib_sakata.ctrl.**dob**(*freq, zeta, M, C, K, dt, nd = 0*)

This function is for design of a discrete-time disturbance observer (DOB).
$$
\hat{d} = -z^{n_d} Q[z] u + Q[z] P^{-1}[z] y
$$
Here, it is defined that disturbance $d$ is injected in the system as plus sign.

- Parameters:
  - freq: frequency[Hz] of the pole pair of the DOB
  - zeta: damping of the pole pair of the DOB
  - M: mass[kg] of the plant
  - C: viscosity[N/(m/s)] of the plant
  - K: stiffness[N/m] of the plant
  - dt: sampling time of the system
  - nd: sampling number of the dead-time of the system
- Returns:
  - DOBu: $z^{n_d} Q[z]$
  - DOBy: $Q[z] P^{-1}[z]$

**Examples**
```python
>>> DOBu, DOBy = ctrl.dob(5., 0.7, 1., 10., 0., 0.001, 1)
>>> print(DOBu)

 0.0004836 z + 0.0004819
-------------------------
z^3 - 1.956 z^2 + 0.957 z

dt = 0.001

>>> print(DOBy)

970.3 z^2 - 1931 z + 960.7
--------------------------
  z^2 - 1.956 z + 0.957

dt = 0.001
```

## 3.29. zpetc
pylib_sakata.ctrl.**zpetc**(*Pz, dt, zerothr=0.99*)

This function is for design of a zero phase error tracking controller ([ZPETC](https://engineering.purdue.edu/ME576/ZPETC_Tomizuka.pdf)).

- Parameters:
  - Pz: instance of TransferFunction class of discrete-time LTI system
  - dt: sampling time of the system
  - zerothr: threshold to recognize unstable zeros (Optional), Default: 0.99
- Returns:
  - Czpetc: instance of TransferFunction class of ZPETC
  - Nzpetc: number of forward samples

```python
>>> Ps = ctrl.tf([1.],[1., 10., 0.])
>>> Pz = ctrl.c2d(Ps, 0.001)
>>> print(Pz)

2.488e-07 z^2 + 4.975e-07 z + 2.488e-07
---------------------------------------
          z^2 - 1.99 z + 0.99

dt = 0.001

>>> Czpetc, Nzpetc = ctrl.zpetc(Pz)
The common pole-zeros of the zpk model have been deleted.
>>> print(Czpetc)

2.512e+05 z^4 + 2500 z^3 - 5e+05 z^2 - 2500 z + 2.487e+05
---------------------------------------------------------
                           z^4

dt = 0.001

>>> Nzpetc
2
```

## 3.30. filt
pylib_sakata.ctrl.**filt**(*num, den, dt*)

This function is to create transfer functions as rational expressions in $z^{-1}$ and to order the numerator and denominator terms in ascending powers of $z^{-1}$.

- Parameters:
  - num: polynomial coefficients of the numerator of the discrete-time LTI model
  - den: polynomial coefficients of the denominator of the discrete-time LTI model
  - dt: sampling time of the discrete-time LTI model
- Returns:
  - out: instance of TransferFunction class of ZPETC

## 3.31. minreal
pylib_sakata.ctrl.**minreal**(*sys*)

This function is to delete the common pole-zeros of the system.

- Parameters:
  - sys: LTI model (StateSpace or TransferFunction or ZpkModel)
- Returns:
  - out: LTI model whose the common pole-zeros were deleted

# 4. pylib_sakata.fft

## 4.1. FreqResp

class pylib_sakata.fft.**FreqResp**(*freq, resp, dt=0*)

- Parameters:
  - freq: 1-D array frequency data [Hz]
  - resp: 1-D array frequency response data [complex data]
  - dt: sampling time (Optional), Default: 0, set the value >= 0. If dt = 0, the system is continuous time system.

**Examples**
Refer examples of [3.11. sys2frd](#311-sys2frd).

**Methods**
- \__**neg**__()
	Negate a FRD.

- \__**add**__(*other*)
	Add two FRDs (parallel connection).

- \__**radd**__(*other*)
	Right add two FRDs (parallel connection).

- \__**sub**__(*other*)
	Subtract two FRDs.

- \__**rsub**__(*other*)
	Right subtract two FRDs.

- \__**mul**__(*other*)
	Multiply two FRDs (serial connection).

- \__**rmul**__(*other*)
	Right multiply two FRDs (serial connection).

- \__**truediv**__(*other*)
	Divide two FRDs.

- \__**rtruediv**__(*other*)
	Right divide two FRDs.

- \__**pow**__(*other*)
	A FRD to the power of x.

## 4.2. fft

pylib_sakata.fft.**fft**(*data, dt*)

This function is for calculating FFT from 1-D array data.

- Parameters:
  - data: 1-D array time response data
  - dt: sampling time of the time response data
- Returns:
  - freq_data: 1-D array frequency data [Hz]
  - fft_data: 1-D array FFT data

**Examples**
```python
import numpy as np
>>> dt = 0.001
>>> t = np.linspace(0.0, 10., int(10./dt))
>>> x = np.sin(2*np.pi*10.0*t) + np.sin(2*np.pi*50.0*t)
>>> fft.fft(x, dt)
(array([0.00000000e+00, 1.00010001e-01, 2.00020002e-01, ...,
       9.99799980e+02, 9.99899990e+02, 1.00000000e+03]), array([3.93463040e-17, 2.39600781e-06, 4.79322523e-06, ...,
       7.19286425e-06, 4.79322523e-06, 2.39600781e-06]))
```

## 4.3. fft_ave

pylib_sakata.fft.**fft_ave**(*data, dt, windivnum=4, overlap=0.5*)

This function is for calculating averaged FFT from 1-D array data.

- Parameters:
  - data: 1-D array time response data
  - dt: sampling time of the time response data
  - windivnum: number of windows to divide the time response data
  - overlap: overlap retio divided time response data (0 <= overlap < 1)
- Returns:
  - freq_data: 1-D array frequency data [Hz]
  - fft_data: 1-D array FFT data

**Examples**
```python
import numpy as np
>>> dt = 0.001
>>> t = np.linspace(0.0, 10., int(10./dt))
>>> x = np.sin(2*np.pi*10.0*t) + np.sin(2*np.pi*50.0*t)
>>> fft.fft_ave(x, dt, 4, 0.5)
(array([0.00000000e+00, 4.88519785e-01, 9.77039570e-01, ...,
       9.99022960e+02, 9.99511480e+02, 1.00000000e+03]), array([5.33388114e-05, 5.46444944e-05, 5.85999322e-05, ...,
       6.53498409e-05, 5.85999322e-05, 5.46444944e-05]))
```

## 4.4. tfestimate

pylib_sakata.fft.**tfestimate**(*x, y, freq, dt, windivnum=4, overlap=0.5*)

This function is for system identification from input and output time response data.

- Parameters:
  - x: 1-D array time response data of input
  - y: 1-D array time response data of output
  - freq: 1-D array frequency data [Hz]
  - dt: sampling time of the time response data
  - windivnum: number of windows to divide the time response data
  - overlap: overlap retio divided time response data (0 <= overlap < 1)
- Returns:
  - freqresp: instance of FreqResp class
  - coh: 1-D array coherence data

**Examples**
```python
>>> import numpy as np
>>> from scipy import signal
>>> from control import matlab
>>> P_tf = ctrl.tf([1.], [1., 2., 0.])
>>> C_tf = ctrl.tf([2., 4., 1.], [1., 2., 0.])
>>> dt = 0.001
>>> t = np.linspace(0.0, 10., int(10./dt))
>>> d = signal.chirp(t, f0=0.1, f1=500., t1=50., method='logarithmic', phi=-90.)
>>> u, tout, xout = matlab.lsim(ctrl.feedback(P_tf, C_tf), d, t)
>>> y, tout, xout = matlab.lsim(ctrl.feedback(P_tf, C_tf, 'SP'), d, t)
The common pole-zeros of the zpk model have been deleted.
>>> freq = np.logspace(np.log10(1.), np.log10(500.), 10000, base=10)
>>> fft.tfestimate(u, y, freq, dt)
(array([-0.11586071-1.49076948e-01j, -0.11590257-1.49078279e-01j,
       -0.11594446-1.49079611e-01j, ..., -0.13155661-1.45847229e-01j,
       -0.14201211-1.03211830e-01j, -0.15745492-1.88460358e-14j]), array([0.77271576, 0.77244166, 0.7721674 , ..., 0.23080322, 0.32843388,
       0.43848208]))
```

## 4.5. frdresize

pylib_sakata.fft.**frdresize**(*freqresp, freq*)

This function is for resizing a frequency response data.

- Parameters:
  - freqresp: instance of FreqResp class
  - freq: new 1-D array frequency data [Hz] for resize
- Returns:
  - freqresp: resized instance of FreqResp class based on new frequency data array

**Examples**
```python
>>> import numpy as np
>>> freq = np.logspace(np.log10(1.), np.log10(1000.), 100, base=10)
>>> Sys_tf = ctrl.tf([1., 2.], [3., 4., 5.])
>>> freqresp = ctrl.sys2frd(Sys_tf, freq)
>>> freq_resize = np.logspace(np.log10(1.), np.log10(1000.), 10000, base=10)
>>> fft.frdresize(freqresp, freq_resize)

freq = array([   1.            1.00069108    1.00138264 ...  998.61926487  999.30939397
 1000.        ])
resp = array([-5.10821217e-03-5.65218353e-02j -5.10244172e-03-5.64813436e-02j
 -5.09666728e-03-5.64408239e-02j ... -5.64622394e-09-5.31301950e-05j
 -5.63759203e-09-5.30909366e-05j -5.62895416e-09-5.30516511e-05j])
```

## 4.6. frdsim

pylib_sakata.fft.**frdsim**(*freqresp, x, dt*)

This function is for simulation steady time response data when a time-domain data is input to a system written by frequency response data.
$$
y(t) = \text{ifft}(\text{FreqResp}(\omega)\times\text{fft}(u(t)))
$$

- Parameters:
  - freqresp: instance of FreqResp class
  - x: 1-D array time response data of input
  - dt: sampling time of the time response data
- Returns:
  - y: 1-D array time response data of output

**Examples**
```python
>>> import numpy as np
>>> freq = np.logspace(np.log10(1.), np.log10(1000.), 100, base=10)
>>> Sys_tf = ctrl.tf([1., 2.], [3., 4., 5.])
>>> freqresp = ctrl.sys2frd(Sys_tf, freq)
>>> dt = 0.001
>>> t = np.linspace(0.0, 10., int(10./dt))
>>> x = np.sin(2*np.pi*10.0*t) + np.sin(2*np.pi*50.0*t)
>>> fft.frdsim(freqresp, x, dt)
array([-0.00635301, -0.00630568, -0.00612106, ..., -0.00610225,
       -0.0062943 , -0.00634922])
```

# 5. pylib_sakta.meas

## 5.1. MeasData

class pylib_sakata.meas.**MeasData**(*time, list, value, time, dt*)

- Parameters:
  - dataList: array of data list (*str*)
  - dataValue: array of data value
  - time: 1-D array time data [s]
  - dt: sampling time of the time response data

## 5.2. getcsvdata

pylib_sakata.meas.**getcsvdata**(*filePath*)

This function is for getting measurement data from a csv file.

- Parameters:
  - filePath: csv file path of measurement file
- Returns:
  - instance of MeasData class

**Examples**
```python
measdata = meas.getcsvdata('data\001-inject.csv')
```

## 5.3. gettxtdata

pylib_sakata.meas.**gettxtdata**(*filePath*)

This function is for getting measurement data from a txt file.

- Parameters:
  - filePath: txt file path of measurement file
- Returns:
  - instance of MeasData class

**Examples**
```python
measdata = meas.gettxtdata('data\001-inject.txt')
```

## 5.4. getmatdata

pylib_sakata.meas.**getmatdata**(*filePath*)

This function is for getting measurement data from a mat file.

- Parameters:
  - filePath: mat file path of measurement file
- Returns:
  - instance of MeasData class

**Examples**
```python
measdata = meas.getmatdata('data\001-inject.mat')
```

## 5.5. getdata

pylib_sakata.meas.**getdata**(*filePath*)

This function is for getting measurement data from a file.

- Parameters:
  - filePath: file path of measurement file (.csv, .txt, and .mat are supported.)
- Returns:
  - instance of MeasData class

**Examples**
```python
measdata = meas.getmatdata('data\001-inject.csv')
```
```python
measdata = meas.getmatdata('data\001-inject.txt')
```
```python
measdata = meas.getmatdata('data\001-inject.mat')
```

## 5.6. getdataindex

pylib_sakata.meas.**getdataindex**(*measdata, dataName*)

- Parameters:
  - measdata: instance of MeasData class
  - dataName: data name
- Returns:
  - index: index of dataName

**Examples**
```python
index = meas.getdataindex(measdata, 'ServoOut')
```

## 5.7. measdata2frd

pylib_sakata.meas.**measdata2frd**(*filePath, inputName, outputName, flagName, freq, inputGain=1.0, outputGain=1.0, windivnum=4, overlap=0.5*)

This function is for system identification from input and output time response data of measurement file.

- Parameters:
  - filePath: file path of measurement file (.csv, .txt, and .mat are supported.)
  - inputName: input data name in the measurement file
  - outputName: output data name in the measurement file
  - flagName: flag data name in the measurement file
  - freq: 1-D array frequency data [Hz]
  - inputGain: inputdata gain (Optional), Default: 1.0, unit of input can be fixed by this parameter.
  - outputGain: outputdata gain (Optional), Default: 1.0, unit of output can be fixed by this parameter.
  - dt: sampling time of the time response data
  - windivnum: number of windows to divide the time response data
  - overlap: overlap retio divided time response data (0 <= overlap < 1)
- Returns:
  - freqresp: instance of FreqResp class
  - coh: 1-D array coherence data

**Examples**
```python
import numpy as np
freq = np.logspace(np.log10(1.), np.log10(1000.), 10000, base=10)
freqresp, coh = measdata2frd('data\001-inject.csv', 'ServoOut', 'PosErrUm', 'FlagNoise', freq, 1., 1.e-6)
```

# 6. pylib_sakata.traj

## 6.1. TrajInf

class pylib_sakata.traj.**TrajInf**(*time, pos, vel, acc, T, dt*)

- Parameters:
  - time: 1-D array time data [s]
  - pos: 1-D array position trajectory data [m]
  - vel: 1-D array velocity trajectory data [m/s]
  - acc: 1-D array acceleration trajectory data [m/s^2]
  - T: moving time [s]
  - dt: sampling time of the trajectory data

## 6.2. traj4th

pylib_sakata.traj.**traj4th**(*posStart, posStep, velMax, accAve, dt, Tstay=0*)

This function is for generation of a 4th order polynomial trajectory.

- Parameters:
  - posStart: start position of the trajectory
  - posStep: step position of the trajectory
  - velMax: maximum of velocity of the trajectory
  - accAve: average of accelation (= decelation) of the trajectory
  - dt: sampling time of the trajectory data.
- Returns:
  - out: instance of TrajInf class of the 4th order polynomial trajectory

**Examples**
```python
traj = traj.traj4th(0, 100, 100, 200, 0.001, 0.5)

fig = plot.makefig()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plot.plot_xy(ax1, traj.time, traj.pos, ylabel='[m]', legend=['Pos'], title='TrajInf')
plot.plot_xy(ax2, traj.time, traj.vel, ylabel='[m/s]', legend=['Vel'])
plot.plot_xy(ax3, traj.time, traj.acc, xlabel='Time [s]', ylabel='[m/s^2]', legend=['Acc'])
plot.savefig('time_traj.png')
```

<img src="figure\time_traj.png" alt="vscode_disp" style="zoom: 80%;" />

# 7. pylib_sakata.plot

## 7.1. plot_xy

pylib_sakata.plot.**plot_xy**(*ax, x, y, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, xlabel=None, ylabel=None, legend=None, title=None, xscale='linear', yscale='linear', labelouter=True*)

This function is for drawing a 2-D figure from x and y data. You can select xy scale in linear type, log type and so on.

- Parameters:
  - ax: handle of axis
  - x: 1-D array data of x-axis
  - y: 1-D array data of y-axis
  - styl: line style (Optional), Default: '-', Select in '-' (solid), '--' (dashed), '.' (dotted), '-.' (dashdot)
  - col: line color (Optional), Default: 'b' (blue)
  - width: line width (Optional), Default: 1.5
  - alpha: transmittance of line, Default: 1.0, set from 0 to 1.0
  - xrange: plot range of x-axis (Optional), Default: None, set as [xmin, xmax]
  - yrange: plot range of y-axis (Optional), Default: None, set as [ymin, ymax]
  - xlabel: label of x-axis (Optional), Default: None, set strings data
  - ylabe: label of y-axis (Optional), Default: None, set strings data
  - legend: legend of data, Default: None, set a list of strings data
  - title: title of figure (Optional), Default: None, set strings data
  - xscale: scale type of x-axis (Optional), Default: 'linear', set in 'linear', 'log', ...
  - labelouter: only display outer label of xy-axes, Default: True
- Returns:
  - None

**Examples**
```python
fig1 = plot.makefig()
ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)
plot.plot_xy(ax1, t, chirp, '-', 'm', 0.5, 1.0, [0, 50], [-3.0, 3.0], ylabel='Input [N]', legend=['Chirp'], title='Time response')
plot.plot_xy(ax2, tout, u, '-', 'b', 0.5, 1.0, [0, 50], [-3.0, 3.0], ylabel='Input [N]', legend=['Servo Out'])
plot.plot_xy(ax3, tout, y*1.0e3, '-', 'b', 0.5, 1.0, [0, 50], [-0.3, 0.3], xlabel='Time [s]', ylabel='Output [mm]', legend=['Position'])

fig2 = plot.makefig()
ax1 = fig2.add_subplot(311)
ax2 = fig2.add_subplot(312)
ax3 = fig2.add_subplot(313)
plot.plot_xy(ax1, fft_axis, chirp_fft, '-', 'm', 1.5, 1.0, freqrange, [0, 1.0], ylabel='Input [N]', legend=['Chirp'], title='Power spectrum density', xscale='log')
plot.plot_xy(ax2, fft_axis, u_fft, '-', 'b', 1.5, 1.0, freqrange, [0, 0.1], ylabel='Input [N]', legend=['Servo Out'], xscale='log')
plot.plot_xy(ax3, fft_axis, y_fft*1.0e6, '-', 'b', 1.5, 1.0, freqrange, [0, 20], xlabel='Frequency [Hz]', ylabel='Output [um]', legend=['Position'], xscale='log')
```

## 7.2. plot_tf

pylib_sakata.plot.**plot_tf**(*ax_mag, ax_phase, sys, freq, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, title=None, labelouter=True*)

This function is for drawing a Bode diagram from a LTI model.

- Parameters:
  - ax_mag: handle of magnitude axis
  - ax_phase: handle of phase axis, if you set as None, phase axis is skipped.
  - sys: LTI model (StateSpace or TransferFunction)
  - freq: 1-D array frequency data [Hz]
  - styl: line style (Optional), Default: '-', Select in '-' (solid), '--' (dashed), '.' (dotted), '-.' (dashdot)
  - col: line color (Optional), Default: 'b' (blue)
  - width: line width (Optional), Default: 1.5
  - alpha: transmittance of line, Default: 1.0, set from 0 to 1.0
  - freqrange: plot range of frequency-axis (Optional), Default: None, set as [freqmin, freqmax]
  - magrange: plot range of magnitude-axis (Optional), Default: None, set as [magmin, magmax]
  - legend: legend of data, Default: None, set a list of strings data
  - title: title of figure (Optional), Default: None, set strings data
  - labelouter: only display outer label of xy-axes, Default: True
- Returns:
  - None

**Examples**
```python
# Sensitivity function
fig1 = plot.makefig()
ax_mag = fig1.add_subplot(111)
ax_phase = None
plot.plot_tf(ax_mag, ax_phase, S, freq, '-', 'm', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tf(ax_mag, ax_phase, Sn, freq, '--', 'b', 1.5, 1.0, [1, 1000], [-60, 10], legend=['Measurement','Model'])

# Complementary sensitivity function
fig2 = plot.makefig()
ax_mag = fig2.add_subplot(211)
ax_phase = fig2.add_subplot(212)
plot.plot_tf(ax_mag, ax_phase, T, freq, '-', 'm', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tf(ax_mag, ax_phase, Tn, freq, '--', 'b', 1.5, 1.0, [1, 1000], [-60, 10], legend=['Measurement','Model'])
```

## 7.3. plot_tffrd

pylib_sakata.plot.**plot_tffrd**(*ax_mag, ax_phase, freqresp, styl='-', col='b', width=1.5, alpha=1.0, freqrange=None, magrange=None, legend=None, title=None, labelouter=True, ax_coh=None, coh=None*)

This function is for drawing a Bode diagram from a frequency response data.

- Parameters:
  - ax_mag: handle of magnitude axis
  - ax_phase: handle of phase axis, if you set as None, phase axis is skipped.
  - freqresp: instance of FreqResp class
  - freq: 1-D array frequency data [Hz]
  - styl: line style (Optional), Default: '-', Select in '-' (solid), '--' (dashed), '.' (dotted), '-.' (dashdot)
  - col: line color (Optional), Default: 'b' (blue)
  - width: line width (Optional), Default: 1.5
  - alpha: transmittance of line, Default: 1.0, set from 0 to 1.0
  - freqrange: plot range of frequency-axis (Optional), Default: None, set as [freqmin, freqmax]
  - magrange: plot range of magnitude-axis (Optional), Default: None, set as [magmin, magmax]
  - legend: legend of data, Default: None, set a list of strings data
  - title: title of figure (Optional), Default: None, set strings data
  - labelouter: only display outer label of xy-axes, Default: True
  - ax_coh: handle of coherence axis, Default: None
  - coh: 1-D array coherence data, Default: None
- Returns:
  - None

**Examples**
```python
# Plant
fig1 = plot.makefig()
ax_mag = fig.add_subplot(311)
ax_phase = fig.add_subplot(312)
ax_coh = fig.add_subplot(313)
plot.plot_tffrd(ax_mag, ax_phase, Pmeas_frd, '-', 'm', 1.5, 1.0, ax_coh=ax_coh, coh=coh, title='Frequency response of plant')
plot.plot_tffrd(ax_mag, ax_phase, Pnz_frd, '--', 'b', 1.5, 1.0, freqrange, legend=['Measurement','Model'])

# Sensitivity function
fig2 = plot.makefig()
ax_mag = fig1.add_subplot(111)
ax_phase = None
plot.plot_tffrd(ax_mag, ax_phase, S_frd, '-', 'm', 1.5, 1.0, title='Frequency response of sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Sn_frd, '--', 'b', 1.5, 1.0, [1, 1000], [-60, 10], legend=['Measurement','Model'])

# Complementary sensitivity function
fig3 = plot.makefig()
ax_mag = fig2.add_subplot(211)
ax_phase = fig2.add_subplot(212)
plot.plot_tffrd(ax_mag, ax_phase, T_frd, '-', 'm', 1.5, 1.0, title='Frequency response of complementary sensitivity function')
plot.plot_tffrd(ax_mag, ax_phase, Tn_frd, '--', 'b', 1.5, 1.0, [1, 1000], [-60, 10], legend=['Measurement','Model'])
```

## 7.4. plot_nyquist

pylib_sakata.plot.**plot_nyquist**(*ax, freqresp, styl='-', col='b', width=1.5, alpha=1.0, xrange=None, yrange=None, legend=None, title=None, labelouter=True*)

This function is for drawing a Nyquist diagram from a frequency response data of a open loop system

- Parameters:
  - ax: handle of axis
  - freqresp: instance of FreqResp class
  - styl: line style (Optional), Default: '-', Select in '-' (solid), '--' (dashed), '.' (dotted), '-.' (dashdot)
  - col: line color (Optional), Default: 'b' (blue)
  - width: line width (Optional), Default: 1.5
  - alpha: transmittance of line, Default: 1.0, set from 0 to 1.0
  - xrange: plot range of real-axis (Optional), Default: None, set as [realmin, realmax]
  - yrange: plot range of imaginary-axis (Optional), Default: None, set as [imagmin, imagmax]
  - legend: legend of data, Default: None, set a list of strings data
  - title: title of figure (Optional), Default: None, set strings data
  - labelouter: only display outer label of xy-axes, Default: True
- Returns:
  - None

**Examples**
```python
fig = plot.makefig()
ax = fig.add_subplot(111)
plot.plot_nyquist(ax, G_frd, '-', 'm', 1.5, 1.0, title='Nyquist Diagram')
plot.plot_nyquist(ax, Gn_frd, '--', 'b', 1.5, 1.0, legend=['Measurement','Model'])
```

## 7.5. plot_nyquist_assistline

pylib_sakata.plot.**plot_nyquist_assistline**(*ax*)

This function is for drawing assist line of a Nyquist diagram.

- Parameters:
  - ax: handle of axis
- Returns:
  - None

**Examples**
```python
plot.plot_nyquist_assistline(ax)
```

## 7.6. makefig

pylib_sakata.plot.**makefig**(dpi=100, popwin=False)

This function will make a new figure handle.

- Parameters:
  - dpi: dot per inch of figure (Optional), Default: 100
  - popwin: switch to fix the popup window of the figure (Optional), Default: False
- Returns:
  - fig: figure handle

**Examples**
```python
fig = plot.makefig()
```

## 7.7. savefig

pylib_sakata.plot.**savefig**(*figName*)

This function will save a current figure.

- Parameters:
  - figName: figure name for saving a current figure, set strings data
- Returns:
  - None

**Examples**
```python
plot.savefig('freq_P.png')
```

## 7.8. showfig

pylib_sakata.plot.**showfig**()

This function will call matplotlib.pyplot.show()

**Examples**
```python
showfig()
```

# 8. pylib_sakata.init

## 8.1. close_all

pylib_sakata.init.**close_all**()

This function is for closing current opened figures.

**Examples**
```python
close_all()
```

## 8.2. clear_all

pylib_sakata.init.**clear_all**()

This function is for deleting all defined variables.

**Examples**
```python
clear_all()
```
