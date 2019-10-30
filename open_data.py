from NImodules import  IV_Retrieve_Titles_log_py, sin_py
import os
import matplotlib.pyplot as plt
import analysis
import numpy as np


#start = "C:\\D\experiment\\program\\Y.H Chang-20190322T114529Z-001\\VNA-20190708T092528Z-001\\VNA\\S21(B for Q2)@3.5to8GHz,I=0to0.5mA,101 points"
#start = "C:\\D\\experiment\\program\\wcc\\2015 TRANSMON　sample 1\\s21@4-8GHz"
start = "C:\\D\\experiment\\program\\all_data\\transmon data\\S parameter (15mK)\\S21(Pp)@fp=4-8GHz IFB=10Hz 8001pts"

def open_data(start):
    title_log = IV_Retrieve_Titles_log_py(start)
    title = title_log[0]
    log = title_log[1]
    log_name = "0 to %s step 1" %log
    freq = []
    Amp = []
    phase = []   
    for i in range(log):
        data = analysis.freq_analysis(start,log_name,i)
        freq += data[0]
        Amp += data[1]
        phase += data[2]
    return freq,Amp,phase
    

p = open_data(start)
plt.plot(p[0],p[1])
plt.show()
