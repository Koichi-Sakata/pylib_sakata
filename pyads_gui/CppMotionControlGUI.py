from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
import pyads
import func_MotionCommandWindow
import func_ControlModeSet
import func_NoiseParamSet
import func_TrajectoryParamSet


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
        text.delete(0, tk.END)
        text.insert(0, value)
    except:
        pass


def tab1_main(frm):
    func_MotionCommandWindow.func_dispUI(frm, net_id_text, port_text)


def tab2_main(frm):
    func_ControlModeSet.func_dispUI(frm, net_id_text, port_text)


def tab3_main(frm):
    func_NoiseParamSet.func_dispUI(frm, net_id_text, port_text)


def tab4_main(frm):
    func_TrajectoryParamSet.func_dispUI(frm, net_id_text, port_text)

root = tk.Tk()
root.title("CppMotionControl GUI")
frm = ttk.Frame(root,padding=10)
frm.grid(column=0, row=0, sticky=NSEW)
tab=ttk.Notebook(root)
tab.grid(column=0, row=1, sticky=NSEW)
frm4 = ttk.Frame(root,padding=10)
frm4.grid(column=0, row=2, sticky=NSEW)

tab1=tk.Frame(tab)
tab2=tk.Frame(tab)
tab3=tk.Frame(tab)
tab4=tk.Frame(tab)

tab.add(tab1, text="Motion Command", padding=3)
tab.add(tab2, text="Control Mode Set",padding=3)
tab.add(tab3, text="Noise Param Set", padding=3)
tab.add(tab4, text="Trajectory Param Set", padding=3)

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

tab.grid()

tab1_main(tab1)
tab2_main(tab2)
tab3_main(tab3)
tab4_main(tab4)


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
    ttk.Label(frm4, text=sigName[i], width=16).grid(column=0, row=rowNum + i, sticky=W)
    for j in range(len(adsName[i])):
        text_sigvalue[i].append(ttk.Entry(frm4, textvariable=sigvalue_read[i][j], width=16))
        text_sigvalue[i][j].grid(column=1+j, row=rowNum + i)

# Style
s = ttk.Style()
s.theme_use('classic')

repeat()
root.mainloop()



