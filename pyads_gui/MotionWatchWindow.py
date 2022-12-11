# Copyright (c) 2022 Koichi Sakata


from tkinter import *
from tkinter import ttk
import tkinter
import pyads


def repeat():
    for i in range(len(adsName)):
        for j in range(len(adsName[i])):
            if i <= 1:
                read_value(adsName[i][j], text_sigvalue[i][j], datatype='bin')
            else:
                read_value(adsName[i][j], text_sigvalue[i][j])

    root.after(100, repeat)


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

    
root = Tk()
root.title("Motion Watch Window")
frm = ttk.Frame(root,padding=10)
frm.grid(column=0, row=0, sticky=NSEW)
frm4 = ttk.Frame(root,padding=10)
frm4.grid(column=0, row=1, sticky=NSEW)

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
ttk.Label(frm4, text="Axis").grid(column=0, row=rowNum)
ttk.Label(frm4, text="0").grid(column=1, row=rowNum)
ttk.Label(frm4, text="1").grid(column=2, row=rowNum)
ttk.Label(frm4, text="2").grid(column=3, row=rowNum)
ttk.Label(frm4, text="3").grid(column=4, row=rowNum)
ttk.Label(frm4, text="4").grid(column=5, row=rowNum)
ttk.Label(frm4, text="5").grid(column=6, row=rowNum)
rowNum += 1

sigName = [
            'StatusWord',
            'ControlWord',
            'RefPosUm',
            'ActPosUm',
            'ErrPosUm',
            'ServoOutN',
            'NoiseOut'
          ]
adsName = [
            [
                'Motion1_Obj1 (Module1).Inputs.StatusWord[0]',
                'Motion1_Obj1 (Module1).Inputs.StatusWord[1]',
                'Motion1_Obj1 (Module1).Inputs.StatusWord[2]',
                'Motion1_Obj1 (Module1).Inputs.StatusWord[3]',
                'Motion1_Obj1 (Module1).Inputs.StatusWord[4]',
                'Motion1_Obj1 (Module1).Inputs.StatusWord[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Outputs.ControlWord[0]',
                'Motion1_Obj1 (Module1).Outputs.ControlWord[1]',
                'Motion1_Obj1 (Module1).Outputs.ControlWord[2]',
                'Motion1_Obj1 (Module1).Outputs.ControlWord[3]',
                'Motion1_Obj1 (Module1).Outputs.ControlWord[4]',
                'Motion1_Obj1 (Module1).Outputs.ControlWord[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Data.RefPosUm[0]',
                'Motion1_Obj1 (Module1).Data.RefPosUm[1]',
                'Motion1_Obj1 (Module1).Data.RefPosUm[2]',
                'Motion1_Obj1 (Module1).Data.RefPosUm[3]',
                'Motion1_Obj1 (Module1).Data.RefPosUm[4]',
                'Motion1_Obj1 (Module1).Data.RefPosUm[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Data.ActPosUm[0]',
                'Motion1_Obj1 (Module1).Data.ActPosUm[1]',
                'Motion1_Obj1 (Module1).Data.ActPosUm[2]',
                'Motion1_Obj1 (Module1).Data.ActPosUm[3]',
                'Motion1_Obj1 (Module1).Data.ActPosUm[4]',
                'Motion1_Obj1 (Module1).Data.ActPosUm[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Data.ErrPosUm[0]',
                'Motion1_Obj1 (Module1).Data.ErrPosUm[1]',
                'Motion1_Obj1 (Module1).Data.ErrPosUm[2]',
                'Motion1_Obj1 (Module1).Data.ErrPosUm[3]',
                'Motion1_Obj1 (Module1).Data.ErrPosUm[4]',
                'Motion1_Obj1 (Module1).Data.ErrPosUm[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Data.ServoOutN[0]',
                'Motion1_Obj1 (Module1).Data.ServoOutN[1]',
                'Motion1_Obj1 (Module1).Data.ServoOutN[2]',
                'Motion1_Obj1 (Module1).Data.ServoOutN[3]',
                'Motion1_Obj1 (Module1).Data.ServoOutN[4]',
                'Motion1_Obj1 (Module1).Data.ServoOutN[5]'
            ],
            [
                'Motion1_Obj1 (Module1).Data.NoiseOut[0]',
                'Motion1_Obj1 (Module1).Data.NoiseOut[1]',
                'Motion1_Obj1 (Module1).Data.NoiseOut[2]',
                'Motion1_Obj1 (Module1).Data.NoiseOut[3]',
                'Motion1_Obj1 (Module1).Data.NoiseOut[4]',
                'Motion1_Obj1 (Module1).Data.NoiseOut[5]'
            ]
          ]

sigvalue_read = [[] for i in range(len(adsName))]
for i in range(len(adsName)):
    for j in range(len(adsName[i])):
        sigvalue_read[i].append(StringVar(frm4))

text_sigvalue = [[] for i in range(len(adsName))]
for i in range(len(adsName)):
    ttk.Label(frm4, text=sigName[i]).grid(column=0, row=rowNum + i, sticky=W)
    for j in range(len(adsName[i])):
        text_sigvalue[i].append(ttk.Entry(frm4, textvariable=sigvalue_read[i][j], width=16))
        text_sigvalue[i][j].grid(column=1+j, row=rowNum + i)

# Style
s = ttk.Style()
s.theme_use('classic')

root.after(100, repeat)
root.mainloop()
