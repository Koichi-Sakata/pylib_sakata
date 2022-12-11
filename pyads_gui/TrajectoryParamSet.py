# Copyright (c) 2022 Koichi Sakata


from tkinter import *
from tkinter import ttk
import tkinter
import pyads


def watch_value():
    print("read value")
    for i in range(len(adsName)):
        for j in range(len(adsName[i])):
            if i == 3:
                read_value(adsName[i][j], text_sigvalue[i][j], datatype='hex')
            else:
                read_value(adsName[i][j], text_sigvalue[i][j])


def read_write_value():
    print("write value")
    for i in range(len(adsName)):
        for j in range(len(adsName[i])):
            if i >= 3:
                value = int(text_sigvalue[i][j].get(), 0)
            else:
                value = float(text_sigvalue[i][j].get())
            write_value(adsName[i][j], value)


def read_value(adsName, text, datatype=0):
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


def write_value(adsName, value):
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name(adsName, value)
    ads_client.close()

    
root = Tk()
root.title("Trajectory Param Set")
frm = ttk.Frame(root,padding=10)
frm.grid(column=0, row=0, sticky=NSEW)
frm5 = ttk.Frame(root,padding=10)
frm5.grid(column=0, row=1, sticky=NSEW)
frm6 = ttk.Frame(root,padding=10)
frm6.grid(column=0, row=2, sticky=NSEW)

# default value
ads_net_id  = StringVar(frm, value="192.168.10.3.1.1")
port_number = StringVar(frm, value="350")

# place widgets
ttk.Label(frm, text="net id").grid(column=0, row=0)
net_id_text = ttk.Entry(frm, textvariable=ads_net_id, width=16)
net_id_text.grid(column=1, row=0)
ttk.Label(frm, text="port").grid(column=0, row=1)
port_text = ttk.Entry(frm, textvariable=port_number, width=16)
port_text.grid(column=1, row=1)

rowNum = 0
ttk.Label(frm5, text="Axis").grid(column=0, row=rowNum)
ttk.Label(frm5, text="0").grid(column=1, row=rowNum)
ttk.Label(frm5, text="1").grid(column=2, row=rowNum)
ttk.Label(frm5, text="2").grid(column=3, row=rowNum)
ttk.Label(frm5, text="3").grid(column=4, row=rowNum)
ttk.Label(frm5, text="4").grid(column=5, row=rowNum)
ttk.Label(frm5, text="5").grid(column=6, row=rowNum)
rowNum += 1

sigName = [
            'MovingPosMm',
            'VelMaxMm',
            'AccAveMm',
            'MovingAxis',
            'MovingTimes',
            'MovingWaitTimeMs'
          ]
adsName = [
            [
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.MovingPosMm[0]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.MovingPosMm[1]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.MovingPosMm[2]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.MovingPosMm[3]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.MovingPosMm[4]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.MovingPosMm[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.VelMaxMm[0]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.VelMaxMm[1]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.VelMaxMm[2]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.VelMaxMm[3]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.VelMaxMm[4]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.VelMaxMm[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.AccAveMm[0]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.AccAveMm[1]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.AccAveMm[2]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.AccAveMm[3]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.AccAveMm[4]',
                'Motion1_Obj1 (Module1).Inputs.TrajParaSet.AccAveMm[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Inputs.SequenceParaSet.MovingAxis'
            ],
            [
                'Motion1_Obj1 (Module1).Inputs.SequenceParaSet.MovingTimes'
            ],
            [
                'Motion1_Obj1 (Module1).Inputs.SequenceParaSet.MovingWaitTimeMs'
            ]
          ]

sigvalue_read = [[] for i in range(len(adsName))]
for i in range(len(adsName)):
    for j in range(len(adsName[i])):
        sigvalue_read[i].append(StringVar(frm5))

text_sigvalue = [[] for i in range(len(adsName))]
for i in range(len(adsName)):
    ttk.Label(frm5, text=sigName[i]).grid(column=0, row=rowNum + i, sticky=W)
    for j in range(len(adsName[i])):
        text_sigvalue[i].append(ttk.Entry(frm5, textvariable=sigvalue_read[i][j], width=16))
        text_sigvalue[i][j].grid(column=1+j, row=rowNum + i)

ttk.Button(frm6, text='Read', command=watch_value, width=20).grid(column=0, row=rowNum)
ttk.Button(frm6, text='Write', command=read_write_value, width=20).grid(column=1, row=rowNum)

# Style
s = ttk.Style()
s.theme_use('classic')

# root.after(100, repeat)
root.mainloop()
