# Copyright (c) 2022 Koichi Sakata


from tkinter import *
from tkinter import ttk
import tkinter
import pyads


sigName = [
            'FlagServoExe',
            'FlagMotionExe',
            'FlagNoiseExe',
          ]
adsName = [
            'Motion1_Obj1 (Module1).Data.FlagServoExe',
            'Motion1_Obj1 (Module1).Data.FlagMotionExe',
            'Motion1_Obj1 (Module1).Data.FlagNoiseExe'
          ]


def servo_on(text_sigvalue, net_id_text, port_text):
    print("Servo ON")
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.ServoExe', 1)
    ads_client.close()

    watch_value(text_sigvalue, net_id_text, port_text)


def servo_off(text_sigvalue, net_id_text, port_text):
    print("Servo OFF")
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.ServoExe', 0)
    ads_client.close()
    style = ttk.Style()

    watch_value(text_sigvalue, net_id_text, port_text)


def motion_id(net_id_text, port_text, var):
    id_num = var.get()
    print('motion_id:', id_num)
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.MotionId', id_num)
    ads_client.close()


def motion_exe(text_sigvalue, net_id_text, port_text, var):
    motion_id(net_id_text, port_text, var)
    print("Execute Motion")
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name( 'Motion1_Obj1 (Module1).Inputs.MotionExe', 1)
    ads_client.close()
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.MotionExe', 0)
    ads_client.close()

    watch_value(text_sigvalue, net_id_text, port_text)


def motion_reset(text_sigvalue, net_id_text, port_text):
    print("Reset Motion")
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.MotionReset', 1)
    ads_client.close()
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.MotionReset', 0)
    ads_client.close()

    watch_value(text_sigvalue, net_id_text, port_text)

def watch_value(text_sigvalue, net_id_text, port_text):
    print("read value")
    for i in range(len(adsName)):
            read_value(adsName[i], text_sigvalue[i], net_id_text, port_text)


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


def func_dispUI(root, net_id_text, port_text):
    frm1 = ttk.Frame(root, padding=10)
    frm1.grid(column=0, row=0, sticky=NSEW)
    frm2 = ttk.Frame(root, padding=10)
    frm2.grid(column=1, row=0, sticky=NSEW)
    frm3 = ttk.Frame(root, padding=10)
    frm3.grid(column=2, row=0, sticky=NSEW)

    sigvalue_read = [[] for i in range(len(adsName))]
    for i in range(len(adsName)):
        sigvalue_read[i].append(StringVar(frm1))

    rowNum = 0
    text_sigvalue = []
    for i in range(len(adsName)):
        ttk.Label(frm1, text=sigName[i]).grid(column=0, row=rowNum + i, sticky=W)
        text_sigvalue.append(ttk.Entry(frm1, textvariable=sigvalue_read[i], width=6))
        text_sigvalue[i].grid(column=1, row=rowNum + i)

    var = tkinter.IntVar()
    rowNum = 0
    ttk.Radiobutton(frm2, value=0, variable=var, text='Step Motion').grid(column=0, row=rowNum, sticky=W)
    rowNum += 1
    ttk.Radiobutton(frm2, value=1, variable=var, text='Reciprocating Motion').grid(column=0, row=rowNum, sticky=W)
    rowNum += 1
    ttk.Radiobutton(frm2, value=2, variable=var, text='Infinite Reciprocating Motion').grid(column=0, row=rowNum, sticky=W)
    rowNum += 1
    ttk.Radiobutton(frm2, value=3, variable=var, text='Chirp Noise Injection').grid(column=0, row=rowNum, sticky=W)
    rowNum += 1
    ttk.Radiobutton(frm2, value=4, variable=var, text='Constant Velocity Motion').grid(column=0, row=rowNum, sticky=W)
    rowNum += 1

    rowNum = 0
    ttk.Button(frm3, text='Servo ON', command=lambda:servo_on(text_sigvalue, net_id_text, port_text), width=20).grid(column=0, row=rowNum)
    ttk.Button(frm3, text='Servo OFF', command=lambda:servo_off(text_sigvalue, net_id_text, port_text), width=20).grid(column=1, row=rowNum)
    rowNum += 1
    ttk.Button(frm3, text='Execute Motion', command=lambda:motion_exe(text_sigvalue, net_id_text, port_text, var), width=20).grid(column=0, row=rowNum)
    ttk.Button(frm3, text='Reset Motion', command=lambda:motion_reset(text_sigvalue, net_id_text, port_text), width=20).grid(column=1, row=rowNum)


