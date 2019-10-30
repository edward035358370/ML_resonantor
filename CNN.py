import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import pylab as plt
import analysis

#path = "C:\\D\\experiment\\program\\txt_data\\"
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

#Phase1 = analysis.connect_phase(freq,phase1)
print("-------------")
raw = [[freq,Amp,phase,3,12,25,300],
       [freq1,Amp1,phase1,15,37,40,210]
       ]#,[freq2,Amp2,phase2,9,12,25],[freq3,Amp3,phase3,9,12,25],[freq4,Amp4,phase4,9,12,25]]
position = []
print("raw",len(raw))
#package data

data = [freq, Amp, phase]
data1 = [freq1,Amp1,phase1]
data2 = [freq2,Amp2,phase2]
data3 = [freq3,Amp3,phase3]
data4 = [freq4,Amp4,phase4]

All = []
n = 28

start = 24
stop = len(freq)-24
database = analysis.make_twoD(data,start,stop,n)
All.append(database)

start = int(0.025/((max(freq1)-min(freq1))/len(freq1)))
stop = len(freq1)-start
database1 =  analysis.make_twoD(data1,start,stop,n,start*2)
All.append(database1)
'''
start = 24
stop = len(freq2)-24
database2 =  analysis.make_twoD(data2,start,stop,n)
All.append(database2)

start = 24
stop = len(freq3)-24
database3 =  analysis.make_twoD(data3,start,stop,n)
All.append(database3)

start = 250
stop = len(freq4)-250
database4 =  analysis.make_twoD(data4,start,stop,n,100)
All.append(database4)
'''

print("All",len(All))
#labeling-MMF
def MMF_labeling(raw,All):
    for raw_data in raw:
        Phase = analysis.connect_phase(raw_data[0],raw_data[2])
        start = 0
        stop = len(raw_data[0])

        rn = raw_data[3]
        bg = raw_data[4]

        amp_mmf = analysis.MMF(raw_data[1],rn,bg)[0]
        pha_mmf = analysis.MMF(Phase,rn,bg)[0]

        amp_mmf = np.array(amp_mmf)
        
        pha_mmf = np.array(pha_mmf)
        pha_max = max(abs(pha_mmf))
        pha_min = pha_max*0.05
        for i in range(len(raw_data[0])):
            if i <= 10 or i >= len(raw_data[0])-10:
                    amp_mmf[i] = 0

            elif abs(pha_mmf[i]) >= pha_min:
                amp_mmf[i] =0
        amp_mmf = np.array(amp_mmf)
        Max = max(abs(amp_mmf))


        for i in range(len(raw_data[0])):
            if abs(amp_mmf[i]) <= Max*0.5:
                amp_mmf[i] = 0

        start = 0
        stop = len(raw_data[0])-1
        #analysis.two_axis([raw_data[0],amp_mmf,pha_mmf],start,stop)

        resonance = []
        for reson in range(len(amp_mmf)):
            if amp_mmf[reson] != 0:
                resonance.append(reson)
        position.append(resonance)
    
    label_mmf = []
    for part in range(len(All)):
        Range = raw[part][5]
        num = 25
        for i in All[part]:
            k = 0
            for res in position[part]:
                if num >= res-Range and num <= res + Range:
                    label_mmf.append(1)
                    k = 1
                    break
            if k == 0:
                label_mmf.append(0)
            num += 1
    return label_mmf
#labeling-theta

def theta_labeling(raw):
    Th_label = []
    for part in range(len(All)):
        start = raw[part][5]-1
        stop = len(raw[part][0])-start
        database = analysis.make_twoD(raw[part],start,stop,128)
        Th = []
        for i in range(len(database)):
            RI = analysis.R_I(raw[part],i)
            R = RI[0]
            I = RI[1]
            if analysis.dots(database[i]) >= 85:
                error = analysis.for_error(R,I)
                if error >= 0.85 and error <= 1:
                    theta = analysis.ura(R,I)
                else:
                    theta = analysis.circle_fit(R,I)[1]
                Th.append(theta)
            else:
                Th.append(0)
        
        for i in Th:
            if i >= raw[part][6]:
                Th_label.append(1)
            else:
                Th_label.append(0)
    return Th_label

label = theta_labeling(raw)#MMF_labeling(raw,All)#

#plt.plot(np.linspace(0,len(label),len(label)),label)
#plt.show()
print("label",len(label))
#turning into training format
pre_train = []
for plus in All:
    pre_train += plus
print("pre_train",len(pre_train))
train = []
for i in range(len(pre_train)):
    train.append([])
    train[i].append(pre_train[i])
    train[i].append(label[i])

#np.random.shuffle(train)
x_train = []
y_train = []
for i in train:
    x_train.append(i[0])
    y_train.append(i[1])

#plt.imshow(x_train[1032])
#plt.show()
x_train = np.array(x_train)
print("x_train",x_train.shape)

x_train = x_train.reshape(-1,1,n,n)
y_train = np_utils.to_categorical(y_train)

print("x_train",x_train.shape)
print("y_train",y_train.shape)

X_train = x_train[:4453]
Y_train = y_train[:4453]
print("X_train",X_train.shape)
print("Y_train",Y_train.shape)
X_test = x_train[4453:]
Y_test = y_train[4453:]
print("X_test",X_test.shape)
print("Y_test",Y_test.shape)

# Another way to build your CNN

N = n
out = 2
# Conv layer 1 output shape (32, 28, 28)
def cnn(N,out):
    model = Sequential()
    model.add(Convolution2D(
        batch_input_shape=(None, 1, N, N),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',     # Padding method
        data_format='channels_first',
    ))

    model.add(Activation('relu'))

    # Pooling layer 1 (max pooling) output shape (32, 14, 14)
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
        data_format='channels_first',
    ))
    N /= 2
    # Conv layer 2 output shape (64, 14, 14)
    model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    N /= 2
    # Pooling layer 2 (max pooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

    # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(out))
    model.add(Activation('softmax'))

    # Another way to define your optimizer
    adam = Adam(lr=1e-4)

    # We add metrics to get more results you want to see
    model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = cnn(N,out)

#training
hist = model.fit(X_train, Y_train, epochs=100, batch_size=64)

par_his = hist.history
loss = par_his["loss"]
acc = par_his["acc"]
x = np.linspace(0,len(loss),len(loss))

plt.plot(x,loss)
plt.show()

analysis.for_origin("cnn X-mon ep=100","loss",[x,loss])

plt.plot(x,acc)
plt.show()

analysis.for_origin("cnn X-mon ep=100","accuracy",[x,acc])

model.save("./training_model/theta n=28 epoch = 100 v2.h5")

print("model is saved")
