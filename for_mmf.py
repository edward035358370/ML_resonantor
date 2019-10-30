import analysis
import numpy as np
import matplotlib.pyplot as plt


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
#------------
data = [freq4,Amp4,phase4]

def try_plot(data):
    Phase = analysis.connect_phase(data[0],data[2])
    plt.plot(data[0],Phase)
    plt.show()
    plt.plot(data[0],data[1])
    plt.show()

def two_axis(data,start,stop):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(data[0][start:stop],data[2][start:stop],label = "phase")
    ax1.set_ylabel('phase after mmf')
    ax1.set_title("after mmf from %s to %s" %(round(data[0][start],2),round(data[0][stop],2)))

    #plt.xlim(4.5,5)

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(data[0][start:stop],data[1][start:stop],label = "amp",color = 'r')
    ax2.set_ylabel('amp after mmf')
    ax2.set_xlabel('freq')
    plt.legend()
    plt.show()



Phase = analysis.connect_phase(data[0],data[2])

#try_plot([data[0],data[1],Phase])

start = 0#500*i
stop = len(data[0])-1

rn = 3
bg = 25
amp_mmf = analysis.MMF(data[1],rn,bg)[0]
pha_mmf = analysis.MMF(Phase,rn,bg)[0]
x = np.linspace(0,len(data[0]),len(data[0]))
#two_axis([data[0],data[1],Phase],start,stop)
two_axis([data[0],amp_mmf,pha_mmf],start,stop)

pha_mmf = np.array(pha_mmf)
pha_max = max(abs(pha_mmf))
pha_min = pha_max*0.2
print(pha_min)
for i in range(len(data[0])):
    if i <= 10 or i >= len(data[0])-10:
        amp_mmf[i] = 0
    elif abs(pha_mmf[i]) >= pha_min:# and abs(pha_mmf[i+1]) <= pha_min:
        amp_mmf[i] = 0
    
    
amp_mmf = np.array(amp_mmf)
Max = max(abs(amp_mmf))
print(Max)

for i in range(len(data[0])):
    if abs(amp_mmf[i]) <= Max*0.5:
        amp_mmf[i] = 0

start = 0
stop = len(data[0])-1
print(bg,"--------------------")
#analysis.for_origin("mmf","all_amp",[freq[start:stop],amp_mmf[start:stop]])
#analysis.for_origin("mmf","all_freq",[freq[start:stop],pha_mmf[start:stop]])

Data = [data[0],amp_mmf,pha_mmf]
Data1 = [x,amp_mmf,pha_mmf]


#two_axis([data[0],data[1],amp_mmf],start,stop)
two_axis(Data,start,stop)
two_axis(Data1,start,stop)
