# Copyright (c) 2022 Koichi Sakata


from tkinter import *
from tkinter import ttk
import tkinter
import pyads

sigName = [
    'FBType',
    'NF',
    'PF',
    'DOB',
    'DOBType',
    'ZPET',
    'IMP',
    'HAP'
]
adsName = [
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.FBType[0]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.FBType[1]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.FBType[2]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.FBType[3]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.FBType[4]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.FBType[5]'
    ],
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.NF[0]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.NF[1]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.NF[2]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.NF[3]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.NF[4]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.NF[5]'
    ],
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.PF[0]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.PF[1]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.PF[2]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.PF[3]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.PF[4]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.PF[5]'
    ],
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOB[0]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOB[1]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOB[2]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOB[3]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOB[4]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOB[5]'
    ],
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOBType[0]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOBType[1]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOBType[2]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOBType[3]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOBType[4]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.DOBType[5]'
    ],
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.ZPET[0]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.ZPET[1]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.ZPET[2]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.ZPET[3]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.ZPET[4]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.ZPET[5]'
    ],
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.IMP[0]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.IMP[1]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.IMP[2]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.IMP[3]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.IMP[4]',
        'Motion1_Obj1 (Module1).Inputs.ControlMode.IMP[5]'
    ],
    [
        'Motion1_Obj1 (Module1).Inputs.ControlMode.HAP'
    ]
]


def watch_value(text_sigvalue, net_id_text, port_text):
    print("read value")
    for i in range(len(adsName)):
        for j in range(len(adsName[i])):
            read_value(adsName[i][j], text_sigvalue[i][j], net_id_text, port_text)


def read_write_value(text_sigvalue, net_id_text, port_text):
    print("write value")
    for i in range(len(adsName)):
        for j in range(len(adsName[i])):
            value = int(text_sigvalue[i][j].get(), 0)
            write_value(adsName[i][j], value, net_id_text, port_text)


def read_value(adsName, text, net_id_text, port_text, datatype=0):
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    try:
        value = ads_client.read_by_name(adsName)
        value = round(value, 8)
        if datatype == 'hex':
            value = hex(value)
        if datatype == 'bin':
            value = bin(value)
        ads_client.close()
        text.delete(0, tkinter.END)
        text.insert(0, value)
    except:
        pass


def write_value(adsName, value, net_id_text, port_text):
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name(adsName, value)
    ads_client.close()


def func_dispUI(root, net_id_text, port_text):
    frm5 = ttk.Frame(root,padding=10)
    frm5.grid(column=0, row=1, sticky=NSEW)
    frm6 = ttk.Frame(root,padding=10)
    frm6.grid(column=0, row=2, sticky=NSEW)

    rowNum = 0
    ttk.Label(frm5, text="Axis").grid(column=0, row=rowNum)
    ttk.Label(frm5, text="0").grid(column=1, row=rowNum)
    ttk.Label(frm5, text="1").grid(column=2, row=rowNum)
    ttk.Label(frm5, text="2").grid(column=3, row=rowNum)
    ttk.Label(frm5, text="3").grid(column=4, row=rowNum)
    ttk.Label(frm5, text="4").grid(column=5, row=rowNum)
    ttk.Label(frm5, text="5").grid(column=6, row=rowNum)
    rowNum += 1

    sigvalue_read = [[] for i in range(len(adsName))]
    for i in range(len(adsName)):
        for j in range(len(adsName[i])):
            sigvalue_read[i].append(StringVar(frm5))

    text_sigvalue = [[] for i in range(len(adsName))]
    for i in range(len(adsName)):
        ttk.Label(frm5, text=sigName[i], width=16).grid(column=0, row=rowNum + i, sticky=W)
        for j in range(len(adsName[i])):
            text_sigvalue[i].append(ttk.Entry(frm5, textvariable=sigvalue_read[i][j], width=16))
            text_sigvalue[i][j].grid(column=1+j, row=rowNum + i)

    ttk.Button(frm6, text='Read', command=lambda:watch_value(text_sigvalue, net_id_text, port_text), width=20).grid(column=0, row=rowNum)
    ttk.Button(frm6, text='Write', command=lambda:read_write_value(text_sigvalue, net_id_text, port_text), width=20).grid(column=1, row=rowNum)

