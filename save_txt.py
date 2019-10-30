import analysis
import numpy as np
import pylab as plt

start = "C:\\D\\experiment\\program\\all_data\\transmon data\\S parameter (15mK)\\S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts"
file_name = "S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts"
analysis.save_data(start,file_name,0)#start,file_name,k,TF(connect log)

Amp = np.loadtxt("./txt_data/" + file_name + "/Amp.txt")
freq = np.loadtxt("./txt_data/" + file_name + "/freq.txt")

plt.plot(freq,Amp)
plt.show()