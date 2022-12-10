# Copyright (c) 2022 Koichi Sakata


from tkinter import *
from tkinter import ttk
import tkinter
import pyads


def repeat():
    for i in range(len(sigName)):
            read_value(adsName[i], text_sigvalue[i])

    root.after(100, repeat)


def servo_on():
    print("Servo ON")
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.ServoExe', 1)
    ads_client.close()


def servo_off():
    print("Servo OFF")
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.ServoExe', 0)
    ads_client.close()
    style = ttk.Style()


def motion_id():
    id_num = var.get()
    print('motion_id:', id_num)
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name('Motion1_Obj1 (Module1).Inputs.MotionId', id_num)
    ads_client.close()


def motion_exe():
    motion_id()
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


def motion_reset():
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


def write_value(adsName, value):
    print("write value")
    net_id = net_id_text.get()
    port = int(port_text.get())
    value = int(value)
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    ads_client.write_by_name(adsName, value)
    ads_client.close()


def read_value(adsName, text):
    net_id = net_id_text.get()
    port = int(port_text.get())
    ads_client = pyads.Connection(net_id, port)
    ads_client.open()
    value = ads_client.read_by_name(adsName)
    ads_client.close()
    text.delete(0, tkinter.END)
    text.insert(0, value)


root = Tk()
root.title("Motion Command Window")
root.configure()
frm0 = ttk.Frame(root, padding=10)
frm0.grid(column=0, row=0, sticky=NSEW)
frm1 = ttk.Frame(root, padding=10)
frm1.grid(column=0, row=1, sticky=NSEW)
frm2 = ttk.Frame(root, padding=10)
frm2.grid(column=1, row=1, sticky=NSEW)
frm3 = ttk.Frame(root, padding=10)
frm3.grid(column=1, row=0, sticky=NSEW)

# default value
ads_net_id = StringVar(frm0, value="192.168.10.3.1.1")
port_number = StringVar(frm0, value="350")

# place widgets
ttk.Label(frm0, text="net id").grid(column=0, row=0)
net_id_text = ttk.Entry(frm0, textvariable=ads_net_id, width=16)
net_id_text.grid(column=1, row=0)
ttk.Label(frm0, text="port").grid(column=0, row=1)
port_text = ttk.Entry(frm0, textvariable=port_number, width=16)
port_text.grid(column=1, row=1)

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
text_sigvalue = []

sigvalue_read = [[] for i in range(len(sigName))]
for i in range(len(sigName)):
    for j in range(6):
        sigvalue_read[i].append(StringVar(frm1))

rowNum = 0
for i in range(len(sigName)):
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
ttk.Button(frm3, text='Servo ON', command=servo_on, width=20).grid(column=0, row=rowNum)
ttk.Button(frm3, text='Servo OFF', command=servo_off, width=20).grid(column=1, row=rowNum)
rowNum += 1
ttk.Button(frm3, text='Execute Motion', command=motion_exe, width=20).grid(column=0, row=rowNum)
ttk.Button(frm3, text='Reset Motion', command=motion_reset, width=20).grid(column=1, row=rowNum)

# Style
s = ttk.Style()
s.theme_use('classic')

root.after(100, repeat)
root.mainloop()


