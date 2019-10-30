import analysis
import numpy as np
import pylab as plt

#data path = C:\\D\\experiment\\program\\txt_data\\

freq1 = np.loadtxt("./txt_data/s21@4-8GHz/freq.txt")
Amp1 = np.loadtxt("./txt_data/s21@4-8GHz/Amp.txt")
phase1 = np.loadtxt("./txt_data/s21@4-8GHz/phase.txt")

freq = np.loadtxt("./txt_data/X-mon/X-mon freq.txt")
Amp = np.loadtxt("./txt_data/X-mon/X-mon Amp.txt")
phase = np.loadtxt("./txt_data/X-mon/X-mon phase.txt")

freq2 = np.loadtxt("./txt_data/S21(Pp)@fp=4-8GHz IFB=10Hz 8001pts/freq.txt")
Amp2 = np.loadtxt("./txt_data/S21(Pp)@fp=4-8GHz IFB=10Hz 8001pts/Amp.txt")
phase2 = np.loadtxt("./txt_data/S21(Pp)@fp=4-8GHz IFB=10Hz 8001pts/phase.txt")

freq3 = np.loadtxt("./txt_data/S21@fp=4-8GHz Pp=-5 IFB=100Hz 4001pts/freq.txt")
Amp3 = np.loadtxt("./txt_data/S21@fp=4-8GHz Pp=-5 IFB=100Hz 4001pts/Amp.txt")
phase3 = np.loadtxt("./txt_data/S21@fp=4-8GHz Pp=-5 IFB=100Hz 4001pts/phase.txt")

freq4 = np.loadtxt("./txt_data/S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts/freq.txt")
Amp4 = np.loadtxt("./txt_data/S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts/Amp.txt")
phase4 = np.loadtxt("./txt_data/S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts/phase.txt")

data = [freq2,Amp2,phase2]

#print(analysis.fre_point(freq3,750))

RI = analysis.R_I(data,2500)
plt.plot(RI[0],RI[1])
plt.show()

mat = analysis.make_twoD(data,0,4,28,50)#data,start,stop,n
for i in mat:
    plt.imshow(i,cmap = "gray_r")
    plt.show()
'''
data = [freq,Amp,phase]

Range = 50#50*0.001/((max(freq)-min(freq))/len(freq))
start = int(Range)
stop = len(freq) - int(Range)
mat = analysis.make_twoD(data,3406,3407,128,Range)#data,start,stop,n

plt.imshow(mat[0],cmap = "gray_r")
plt.show()
'''
