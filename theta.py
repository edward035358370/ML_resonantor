import analysis
import numpy as np
import pylab as plt

freq = np.loadtxt("./txt_data/X-mon/X-mon freq.txt")
Amp = np.loadtxt("./txt_data/X-mon/X-mon Amp.txt")
phase = np.loadtxt("./txt_data/X-mon/X-mon phase.txt")

freq1 = np.loadtxt("./txt_data/s21@4-8GHz/freq.txt")
Amp1 = np.loadtxt("./txt_data/s21@4-8GHz/Amp.txt")
phase1 = np.loadtxt("./txt_data/s21@4-8GHz/phase.txt")

freq2 = np.loadtxt("./txt_data/S21(Pp)@fp=4-8GHz IFB=10Hz 8001pts/freq.txt")
Amp2 = np.loadtxt("./txt_data/S21(Pp)@fp=4-8GHz IFB=10Hz 8001pts/Amp.txt")
phase2 = np.loadtxt("./txt_data/S21(Pp)@fp=4-8GHz IFB=10Hz 8001pts/phase.txt")

freq3 = np.loadtxt("./txt_data/S21@fp=4-8GHz Pp=-5 IFB=100Hz 4001pts/freq.txt")
Amp3 = np.loadtxt("./txt_data/S21@fp=4-8GHz Pp=-5 IFB=100Hz 4001pts/Amp.txt")
phase3 = np.loadtxt("./txt_data/S21@fp=4-8GHz Pp=-5 IFB=100Hz 4001pts/phase.txt")

freq4 = np.loadtxt("./txt_data/S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts/freq.txt")
Amp4 = np.loadtxt("./txt_data/S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts/Amp.txt")
phase4 = np.loadtxt("./txt_data/S21(B)@fp=4.86-4.92GHz Pp=-5dBm B=1mA IFB=100Hz 601pts/phase.txt")

data = [freq1,Amp1,phase1]
start = 40
stop = len(freq)- 40

def theta_labeling(data):
    database = analysis.make_twoD(data,start,stop,128,80)
    Th = []
    for i in range(len(database)):
        RI = analysis.R_I(data,i)
        R = RI[0]
        I = RI[1]
        if analysis.dots(database[i]) >= 85:
            error = analysis.for_error(R,I)
            if error >= 0.85 and error <= 1:
                print(i,error)
                theta = analysis.ura(R,I)
            else:
                theta = analysis.circle_fit(R,I)[1]
            Th.append(theta)
        else:
            Th.append(0)
    angle = analysis.scatter(Th,10)
    plt.plot(angle[0],angle[1])
    plt.show()
    Th_label = []
    for i in Th:
        if i >= 220:
            Th_label.append(1)
        else:
            Th_label.append(0)
    return Th_label

Th_label = theta_labeling(data)

plt.plot(freq1[start:stop],Th_label)
plt.show()
