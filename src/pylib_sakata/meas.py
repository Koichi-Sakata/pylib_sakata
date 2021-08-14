# Copyright (c) 2021 Koichi Sakata

# dataList, dataValue, t, Ts = getcsvdata(filePath)
# dataList, dataValue, t, Ts = gettxtdata(filePath)
# dataList, dataValue, t, Ts = getmatdata(filePath)
# out = selectdataNum(dataList, dataName)
# freqresp, coh = measdata2frd(filePath, inputName, outputName, flagName, freq, datatype='csv', inputGain=1.0, outputGain=1.0, windivnum=4, overlap=0.5)


from scipy import io
import numpy as np
import pandas as pd
from .fft import tfestimate


def getcsvdata(filePath):
    """For example, data measured by TwinCAT"""
    df = pd.read_csv(filePath ,header=4)
    dataList = df.columns.values
    dataValue = df.loc[:, dataList]
    Ts = (dataValue[0][1] - dataValue[0][0]) * 1.0e-3
    t = np.linspace(0.0, len(dataValue[0]) * Ts, int(len(dataValue[0])))
    return dataList, dataValue, t, Ts


def gettxtdata(filePath):
    """For example, data measured by PMAC"""
    f = open(filePath)
    line = f.readlines()
    f.close()
    dataList = line[0].split()
    dataValue = [[] for i in range(len(dataList))]
    for n in range(len(dataList)):
        dataline = []
        for k in range(1, len(line)):
            dataline.append(float(line[k].split()[n]))
        dataValue[n] = np.array(dataline)
    Ts = dataValue[0][1] - dataValue[0][0]
    t = np.linspace(0.0, len(dataValue[0]) * Ts, int(len(dataValue[0])))
    return dataList, dataValue, t, Ts


def getmatdata(filePath):
    """For example, data measured by Matlab-base software"""
    matdata = io.loadmat(filePath)
    dataNum = len(matdata['dataY'][0])
    Fs = matdata['samplefreq'][0][0]
    Ts = 1.0/Fs
    dataList = [[] for i in range(dataNum)]
    dataValue = [[] for i in range(dataNum)]
    for k in range(dataNum):
        dataList[k] = matdata['dataY'][0][k][0][0]
        dataValue[k] = matdata['dataY'][0][k][1][0]
    t = np.linspace(0.0, len(dataValue[0]) * Ts, int(len(dataValue[0])))
    return dataList, dataValue, t, Ts


def selectdataNum(dataList, dataName):
    out = -1
    for k in range(len(dataList)):
        if dataName in dataList[k]:
            out = k
    if out == -1:
        print('the dataName can not be found in the dataList.')
    return out


def measdata2frd(filePath, inputName, outputName, flagName, freq, inputGain=1.0, outputGain=1.0, windivnum=4, overlap=0.5):
    if filePath[-3:] == 'csv':
        dataList, dataValue, t, Ts = getcsvdata(filePath)
    elif filePath[-3:] == 'txt':
        dataList, dataValue, t, Ts = gettxtdata(filePath)
    elif filePath[-3:] == 'mat':
        dataList, dataValue, t, Ts = getmatdata(filePath)
    else:
        raise Exception('Error: This file type is not supported.')
    flagdata = dataValue[selectdataNum(dataList, flagName)]
    inputdata_tmp = dataValue[selectdataNum(dataList, inputName)]
    outputdata_tmp = dataValue[selectdataNum(dataList, outputName)]
    flaglist = np.where(flagdata > 0)
    inputdata = inputdata_tmp[flaglist] * inputGain
    outputdata = outputdata_tmp[flaglist] * outputGain
    freqresp, coh = tfestimate(inputdata, outputdata, freq, Ts, windivnum=windivnum, overlap=overlap)
    return freqresp, coh


