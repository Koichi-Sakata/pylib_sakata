# Copyright (c) 2021 Koichi Sakata

# MeasData = getcsvdata(filePath)
# MeasData = gettxtdata(filePath)
# MeasData = getmatdata(filePath)
# MeasData = getdata(filePath)
# index = getdataindex(measdata, dataName)
# freqresp, coh = measdata2frd(filePath, inputName, outputName, flagName, freq, inputGain=1.0, outputGain=1.0, windivnum=4, overlap=0.5)


from scipy import io
import numpy as np
import pandas as pd
from .fft import tfestimate

class MeasData:
 
    def __init__(self, list, value, time, dt):
        self.list = list
        self.value = value
        self.time = time
        self.dt = dt


def getcsvdata(filePath):
    """For example, data measured by TwinCAT"""
    df = pd.read_csv(filePath, skiprows=range(5), dtype=object)
    dataList = df.columns.values
    dataValue = df[:-1].values.T.astype(float)
    dt = (dataValue[0][1] - dataValue[0][0]) * 1.0e-3
    t = np.linspace(0.0, len(dataValue[0]) * dt, int(len(dataValue[0])))
    return MeasData(dataList, dataValue, t, dt)


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
    dt = dataValue[0][1] - dataValue[0][0]
    t = np.linspace(0.0, len(dataValue[0]) * dt, int(len(dataValue[0])))
    return MeasData(dataList, dataValue, t, dt)


def getmatdata(filePath):
    """For example, data measured by Matlab-base software"""
    matdata = io.loadmat(filePath)
    dataNum = len(matdata['dataY'][0])
    Fs = matdata['samplefreq'][0][0]
    dt = 1.0/Fs
    dataList = [[] for i in range(dataNum)]
    dataValue = [[] for i in range(dataNum)]
    for k in range(dataNum):
        dataList[k] = matdata['dataY'][0][k][0][0]
        dataValue[k] = matdata['dataY'][0][k][1][0]
    t = np.linspace(0.0, len(dataValue[0]) * dt, int(len(dataValue[0])))
    return MeasData(dataList, dataValue, t, dt)


def getdata(filePath):
    if filePath[-3:] == 'csv':
        measdata = getcsvdata(filePath)
    elif filePath[-3:] == 'txt':
        measdata = gettxtdata(filePath)
    elif filePath[-3:] == 'mat':
        measdata = getmatdata(filePath)
    else:
        raise Exception('Error: This file type is not supported.')
    return measdata


def getdataindex(measdata, dataName):
    index = -1
    for k in range(len(measdata.list)):
        if dataName in measdata.list[k]:
            index = k
    if index == -1:
        print('the dataName can not be found in the dataList.')
    return index


def measdata2frd(filePath, inputName, outputName, flagName, freq, inputGain=1.0, outputGain=1.0, windivnum=4, overlap=0.5):
    measdata = getdata(filePath)
    flagdata = measdata.value[getdataindex(measdata, flagName)]
    inputdata_tmp = measdata.value[getdataindex(measdata, inputName)]
    outputdata_tmp = measdata.value[getdataindex(measdata, outputName)]
    flaglist = np.where(flagdata > 0)
    inputdata = inputdata_tmp[flaglist] * inputGain
    outputdata = outputdata_tmp[flaglist] * outputGain
    freqresp, coh = tfestimate(inputdata, outputdata, freq, measdata.dt, windivnum=windivnum, overlap=overlap)
    return freqresp, coh


