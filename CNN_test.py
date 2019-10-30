import analysis
import numpy as np
import pylab as plt
from keras.models import load_model

#load data
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

#data package
data = [freq, Amp, phase]
data1 = [freq1,Amp1,phase1]
data2 = [freq2,Amp2,phase2]
data3 = [freq3,Amp3,phase3]
data4 = [freq4,Amp4,phase4]
subede = [data,data1,data2,data3,data4]
#make data in 2D
all_database = []
n = 28


start = 24
stop = len(freq)-24
database = analysis.make_twoD(data,start,stop,n)
all_database.append([database,freq,start])

start = int(0.025/((max(freq1)-min(freq1))/len(freq1)))
stop = len(freq1)-start
database1 =  analysis.make_twoD(data1,start,stop,n,start*2)
all_database.append([database1,freq1,start])

start = 24
stop = len(freq2)-24
database2 =  analysis.make_twoD(data2,start,stop,n)
all_database.append([database2,freq2,start])

start = 24
stop = len(freq3)-24
database3 =  analysis.make_twoD(data3,start,stop,n)
all_database.append([database3,freq3,start])

start = 250
stop = len(freq4)-250
database4 =  analysis.make_twoD(data4,start,stop,n,500)
all_database.append([database4,freq4,start])

#load training model
model = load_model("./training_model/MMF n=28 epoch = 100.h5")
#model1 = load_model("./training_model/theta n=28 epoch = 100.h5")

def test_model(model,Database,Freq,n):
    pre_train = np.array(Database)
    prediction = model.predict(pre_train.reshape(-1,1,n,n))

    zero = []
    one = []
    Long = len(Database)
    for i in range(Long):
        zero.append(prediction[i][0])
        one.append(prediction[i][1])
    x = np.linspace(0,Long,Long)

    for i in range(len(one)):
        if one[i] >= 0.9:
            print(Freq[i])
    return one

def two_axis(data,start,stop):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(data[0][start:stop],data[2],label = "phase")
    ax1.set_ylabel('phase after mmf')
    ax1.set_title("after mmf from %s to %s" %(round(data[0][start],2),round(data[0][stop],2)))

    #plt.xlim(4.5,5)

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(data[0][start:stop],data[1][start:stop],label = "amp",color = 'r')
    ax2.set_ylabel('amp after mmf')
    ax2.set_xlabel('freq')
    plt.legend()
    plt.show()

k = 0
for i in all_database:
    pre = test_model(model,i[0],i[1],n)
    #pre1 = test_model(model1,i[0],i[1],n)
    start = i[2]
    data = [i[1][start:len(pre)+start],pre]
    #analysis.for_origin(str(k),str(k),data)
    two_axis([i[1],subede[k][1],pre],start,len(pre)+start)
    plt.plot(i[1][start:len(pre)+start],pre)
    #plt.plot(i[1][start:len(pre1)+start],pre1)
    plt.show()
    k += 1


