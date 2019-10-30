
from NImodules import  IV_Retrieve_py, IV_Retrieve_Current_py,IV_Retrieve_Titles_log_py
import pylab as plt
import sys
import numpy as np
import os
import scipy.ndimage as nd
from scipy import optimize,arccos, sin, cos

#cw-          
def get_data_path(start_path):
    file_path = []
    for root, dirs, files in os.walk(start_path, topdown=True):
        for name in files:
            file_name = os.path.join(root, name)
            #file_name.replace(splash,"\\")
            file_path.append(file_name)
        for name in dirs:
            file_name = os.path.join(root, name)
            #file_name.replace(splash,"\\")
            #file_path.append(file_name)
    return file_path

def CW_analysis(title,log_name,start):#title =  str, log_name = "0 to %s step 1" %log, start = str
    raw_data = IV_Retrieve_py([start,log_name])[0]
    C_array_position = []
    for Current in title:
        if Current =="Current(A)" or Current =="I(nA)" or Current =="I setting (A)" or Current == "Current, mA":
            C_array_position.append("c")
        elif Current == "Time(s)":
            C_array_position.append("x")
        elif Current == "S21 Amp(mU)":
            C_array_position.append("y")
        else:
            C_array_position.append(" ")
    C_Array = IV_Retrieve_Current_py([start,C_array_position,log_name])[0]#output()
    CW_list = []
    for log in range(len(raw_data)):
        k = 0
        for i in raw_data[log][1][2]:
            k = i+k
        CW_list.append(k/float(len(raw_data[log][1][2])))
    J = 0
    current_array = []
    CW_array = []
    I_position = 0

    while J == 0:
        add = 0
        separate_array = []
        CW_separate_array = []
        while add == 0:
            separate_array.append(C_Array[I_position])
            CW_separate_array.append(CW_list[I_position])
            I_position += 1
            if I_position == len(C_Array)-1:
                J = 1
                add = 1
            elif C_Array[I_position-1] > C_Array[I_position]:
                add = 1
        current_array.append(separate_array)
        CW_array.append(CW_separate_array)
    return current_array,CW_array  #output[[current],[average]],current = [],[]....,average = [],[],.....
#freq
def connect_phase(freq,phase):
    Phase = []
    move = 0
    j = 0
    connect = 0
    while j == 0:
        k = 0
        if len(Phase) == len(phase):
            j = 1
        else:
            while abs(phase[move + k - 1] - phase[move + k]) <= 5. :
                Phase.append(phase[k + move]-2*np.pi*connect)
                #if abs(phase[0][move + k - 1] - phase[0][move + k]) ==
                k += 1
            
                if len(Phase) == len(phase):
                    break

            if len(Phase) == len(phase):
                j = 1
            else:
                connect += 1
                Phase.append(phase[k+move]-2*np.pi*connect)
                move += k + 1
    
    z = np.polyfit(freq,Phase,1)
    p = np.poly1d(z)
    Phase = Phase-p(freq)

    return Phase

def freq_analysis(start,log_name,log):
    raw_data = IV_Retrieve_py([start,log_name])[0]
    Freq = raw_data[log][0][2]
    Amp = raw_data[log][1][2]
    phase = raw_data[log][3][2]
    
    return Freq, Amp, phase
#get labview data----------------------------------
def open_data(start):
    title_log = IV_Retrieve_Titles_log_py(start)
    title = title_log[0]
    log = title_log[1]
    log_name = "0 to %s step 1" %log
    freq = []
    Amp = []
    phase = []   
    for i in range(log):
        data = freq_analysis(start,log_name,i)
        freq += data[0]
        Amp += data[1]
        phase += data[2]
    return freq,Amp,phase

def peak_and_deep(v, delta, x = None):
    maxnum = []
    minnum = []
    maxpos = {}
    minpos = {}
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxnum.append(mx)
                maxpos[mx] = mxpos
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                minnum.append(mn)
                minpos[mn] = mnpos
                mx = this
                mxpos = x[i]
                lookformax = True
    peakpos = []
    deeppos = []
    maxnum.sort()
    minnum.sort(reverse=True)
    for POS in maxnum:
        peakpos.append(maxpos[POS])
    for pos in minnum:
        deeppos.append(minpos[pos])
        
    return peakpos, maxnum, deeppos, minnum

def get_data(start_path):
    file_path = [start_path]
    for root, dirs, files in os.walk(start_path, topdown=True):
        for name in files:
            file_name = os.path.join(root, name)
            #file_name.replace(splash,"\\")
            file_path.append(file_name)
        for name in dirs:
            file_name = os.path.join(root, name)
            #file_name.replace(splash,"\\")
            #file_path.append(file_name)
    return file_path

def MMF(Y,rn,bg):
    stru_len_op = int(bg)
    stru_len_clo = int(bg*1.5)
    
    stru_ele_op = np.linspace(0,0,stru_len_op)
    stru_ele_clo = np.linspace(0,0,stru_len_clo)
    
    #triangular wave
    tri_wave = []
    amp = 1.0
    width = 1
    samp = rn
    asym = 0.5
    points = 1
    
    while points <= samp:
    
        Xi = 0.1*points
        if 0 <= Xi and Xi <= width*asym:
            tri_wave.append(amp*Xi/(width*asym))
        elif Xi > width*asym and Xi <width:
            tri_wave.append(amp*(width-Xi)/(width*(1-asym)))
        else:
            tri_wave.append(0)
        points += 1

    #low-pass
    op_flat = nd.grey_opening(Y,size = (stru_len_op),structure = stru_ele_op)
    clo_flat = nd.grey_closing(op_flat,size = (stru_len_clo),structure = stru_ele_clo)
    
    reducing = []
    for reduce in range(len(Y)):
        reducing.append(Y[reduce] - clo_flat[reduce])
    
    op_tri = nd.grey_opening(reducing,size = (rn),structure = tri_wave)
    clo_tri = nd.grey_closing(reducing,size = (rn),structure = tri_wave)

    after_stru_ele =np.linspace(0,0,rn) 
    
    op_than_clo = nd.grey_closing(op_tri,size = (rn),structure = after_stru_ele)
    clo_than_op = nd.grey_opening(clo_tri,size = (rn),structure = after_stru_ele)

    plusing = []
    for plus in range(len(op_than_clo)):
        plusing.append((op_than_clo[plus]+clo_than_op[plus])/2.0)
    
    return plusing, clo_flat

def save_txt(data_name,data):
    f = open(data_name,"w")
    for i in data:
        f.write(" " + str(i))
    return None

def fre_point(freq, pos):
    for i in freq:
        if freq[pos] == i:
            return i
        
def lin_pha(freq,phase):
    Phase = connect_phase(freq,phase)
    z = np.polyfit(freq,Phase,1)
    p = np.poly1d(z)
    Phase = Phase-p(freq)
    return Phase

def R_I(data,RAN,Range=50):
    freq = data[0]
    Amp = data[1]
    phase = data[2]
    Phase = lin_pha(freq,phase)  
    Range = int(Range/2)  
    R = []
    I = []
    RAN = int(RAN)
    for i in range(RAN-Range,RAN+Range):
        R.append(Amp[i]*np.cos(Phase[i]))
        I.append(Amp[i]*np.sin(Phase[i]))
    return R,I

def smooth(R,I,smo):
    smooth_R = np.copy(R)
    for i in range(len(R)):
        sameR = 0
        for k in range(smo):
            if i == k:
                smooth_R[i] = smooth_R[i]
                sameR = 1
        if sameR == 0:
            for k in range(1,smo):
                smooth_R[i] += smooth_R[i-k]
            smooth_R[i] /= smo

    smooth_I = np.copy(I)
    for i in range(len(R)):
        sameI = 0
        for k in range(smo):
            if i == k:
                smooth_I[i] = smooth_I[i]
                sameI = 1
        if sameI == 0:
            for k in range(1,smo):
                smooth_I[i] += smooth_I[i-k]
            smooth_I[i] /= smo
    return smooth_R,smooth_I 

def theta(R,I):
    L = 0
    RI = smooth(R,I,3)
    R = RI[0]
    I = RI[1]
    for length in range(len(R)):
        if length == 0:
            continue
        else:
            X_len = R[length]-R[length-1]
            X = X_len**2
            Y_len = I[length]-I[length-1]
            Y = Y_len**2
            long = (X+Y)**0.5
            L += long
    
    half_L = L/2.
    temp_L = 0
    
    for half in range(len(R)):
        if half == 0:
            continue
        else:
            X_len = R[half]-R[half-1]
            temp_x = X_len**2
            
            Y_len = I[half]-I[half-1]
            temp_y = Y_len**2
            
            temp = (temp_x + temp_y)**0.5
            temp_L += temp
    
        if temp_L >= half_L:
            xm = R[half]
            ym = I[half]
            break
    
    x1 = R[40]
    y1 = I[40]
    xN = R[10]
    yN = I[10]
    
    X = (y1-yN)/2. + (x1+xm)*(x1-xm)/(2.*(y1-ym)) - (xN+xm)*(xN-xm)/(2.*(yN-ym))
    X = X/((x1-xm)/(y1-ym) - (xN-xm)/(yN-ym))
    
    Y = (x1-xN)/2. + (y1+ym)*(y1-ym)/(2.*(x1-xm)) - (yN+ym)*(yN-ym)/(2.*(xN-xm))
    Y = Y/((y1-ym)/(x1-xm) - (yN-ym)/(xN-xm))
    
    r = ((xm-X)**2 + (ym-Y)**2)**0.5
    
    Theta = L/r
    return Theta*180/np.pi

def circle_fit(R,I):
    RI = smooth(R,I,3)
    smo_R = RI[0]
    smo_I = RI[1]
    L = 0
    for length in range(len(smo_R)):
        if length == 0:
            continue
        else:
            X_len = smo_R[length]-smo_R[length-1]
            X = X_len**2
            Y_len = smo_I[length]-smo_I[length-1]
            Y = Y_len**2
            long = (X+Y)**0.5
            L += long
    Aa = []
    for i in range(len(smo_R)):
        Aa.append([])
        Aa[i].append(2*smo_R[i])
        Aa[i].append(2*smo_I[i])
        Aa[i].append(1)
    Aa = np.array(Aa)
    
    Bb = []
    for i in range(len(smo_R)):
        z = smo_R[i]**2 + smo_I[i]**2
        Bb.append(z)
    Bb = np.array(Bb)
    
    answer = np.linalg.inv(np.dot(Aa.T,Aa))
    answer = np.dot(answer,Aa.T)
    answer = np.dot(answer,Bb)
    r = answer[2]+answer[1]**2 + answer[0]**2
    r = r**0.5
    Theta = L/r*180/np.pi
    center = [answer[0],answer[1]]
    return r, Theta, center

def cen_th(R,I):
    circle = circle_fit(R,I)
    center = circle[2]
    L = 0
    RI = smooth(R,I,3)
    R = RI[0]
    I = RI[1]
    for length in range(len(R)):
        if length == 0:
            continue
        else:
            X_len = R[length]-R[length-1]
            X = X_len**2
            Y_len = I[length]-I[length-1]
            Y = Y_len**2
            long = (X+Y)**0.5
            L += long

    half_L = L/2.
    temp_L = 0

    for half in range(len(R)):
        if half == 0:
            continue
        else:
            X_len = R[half]-R[half-1]
            temp_x = X_len**2
            
            Y_len = I[half]-I[half-1]
            temp_y = Y_len**2
            
            temp = (temp_x + temp_y)**0.5
            temp_L += temp

        if temp_L >= half_L:
            xm = R[half]
            ym = I[half]
            break
    vec = [xm-center[0],ym-center[1]]
    if vec[1] <=0:
        way = -1
    else:
        way = 1
    to_one = (vec[0]**2+vec[1]**2)**0.5
    for_th = vec[0]/to_one
    center_th = arccos(for_th)
    return center_th,way

def ura(R,I):
    vector = np.array([R[1]-R[-1],I[1]-I[-1]])
    vec_val = (vector[0]**2+vector[1]**2)**0.5
    vector /= vec_val
    th = 0
    for i in range(0,len(R)-1):
        vector1 = np.array([R[i]-R[-1],I[i]-I[-1]])
        vec_val = (vector1[0]**2+vector1[1]**2)**0.5
        vector1 /= vec_val
        COS = np.dot(vector,vector1)
        th += np.arccos(COS)
        vector = vector1
    Theta = th*180/np.pi
    return Theta

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def for_error(x,y,bo=0):
    fitting = circle_fit(x,y)
    radius = fitting[0]
    Theta = fitting[1]
    center = fitting[2]
    xc = center[0]
    yc = center[1]
    center_theta = cen_th(x,y)[0]
    if cen_th(x,y)[1] == -1:
        center_theta = 2*np.pi-center_theta
    theta_fit = np.linspace(center_theta - Theta*3.14/360,center_theta + Theta*3.14/360,len(x))
    x_fit = xc + radius*np.cos(theta_fit)
    y_fit = yc + radius*np.sin(theta_fit)
    '''
    plt.plot(x,y)
    plt.plot(x_fit,y_fit)
    plt.show()
    '''
    if bo == 1:
        return x_fit,y_fit
    
    error = []
    for i in range(len(x)):
        temp_list = []
        for ii in range(len(x)):
            temp = (x_fit[i]-x[ii])**2 + (y_fit[i]-y[ii])**2
            temp_list.append(temp)
        
        temp_min = min(temp_list)
        for find in range(len(x)):
            if temp_list[find] == temp_min:
                error.append(temp_min**0.5)
    dif = 0
    for i in error:
        dif += i/50
    return dif

def scatter(theta_list,step):
    x = []
    Max = max(theta_list)
    for i in range(int((Max+1)/step)):
        x.append(i*step)
        
    high = np.zeros(int((Max+1)/step))
    for n in theta_list:
        for k in range(int((Max+1)/step)):
            if k == int((Max+1)/step)-1:
                continue
            else:
                if x[k] <= n and x[k+1] >= n:
                    high[k] += 1
    X = np.linspace(0,Max,int((Max+1)/step))
    return X,high

def plot_lim(R,I):
    plt.figure(figsize = (4,4))
    plt.plot(R,I)
    x_space = (max(R) - min(R))
    y_space = (max(I) - min(I))
    if x_space >= y_space:   
        plt.ylim((min(I),x_space+min(I)))
        plt.xlim((min(R),max(R)))
    elif y_space >= x_space:   
        plt.ylim((min(I),max(I)))
        plt.xlim((min(R),y_space+min(R)))
    return None

def ROT_UP(R,I):#before vector
    L = 0
    for length in range(len(R)):
        if length == 0:
            continue
        else:
            X_len = R[length]-R[length-1]
            X = X_len**2
            Y_len = I[length]-I[length-1]
            Y = Y_len**2
            long = (X+Y)**0.5
            L += long
    
    half_L = L/2.
    temp_L = 0
    
    for half in range(len(R)):
        if half == 0:
            continue
        else:
            X_len = R[half]-R[half-1]
            temp_x = X_len**2
            
            Y_len = I[half]-I[half-1]
            temp_y = Y_len**2
            
            temp = (temp_x + temp_y)**0.5
            temp_L += temp
    
        if temp_L >= half_L:
            xm = R[half]
            ym = I[half]
            break
    
    x1 = R[40]
    y1 = I[40]
    
    xN = R[10]
    yN = I[10]
    
    X = (y1-yN)/2. + (x1+xm)*(x1-xm)/(2.*(y1-ym)) - (xN+xm)*(xN-xm)/(2.*(yN-ym))
    X = X/((x1-xm)/(y1-ym) - (xN-xm)/(yN-ym))
    
    Y = (x1-xN)/2. + (y1+ym)*(y1-ym)/(2.*(x1-xm)) - (yN+ym)*(yN-ym)/(2.*(xN-xm))
    Y = Y/((y1-ym)/(x1-xm) - (yN-ym)/(xN-xm))
    vector = [X-xm,Y-ym]
    
    up = np.array([0,1])
    for_one = (vector[0]**2+vector[1]**2)**0.5
    vector = np.array(vector)/for_one
    COS = np.dot(up,vector)
    TH = np.arccos(COS)
    ROT_R = []
    ROT_I = []
    
    if X - xm > 0:
        A = np.sin(TH)
        B = np.cos(TH)
    else:
        A = np.sin(2*np.pi-TH)
        B = np.cos(2*np.pi-TH)    
    turning = [[B,-A],[A,B]]
    turning = np.array(turning)
            
    for turn in range(len(R)):
        vec = np.array([R[turn],I[turn]])
        turn = np.dot(turning,vec)
        ROT_R.append(turn[0])
        ROT_I.append(turn[1])
        
    return ROT_R, ROT_I

def bef_twoD2(data):
    freq = data[0]
    Amp = data[1]
    phase = data[2]
    Phase = lin_pha(freq,phase)    
    x_space = []
    y_space = []
    for ran in range(25,len(freq)-25):
        R = []
        I = []
        for i in range(ran-25,ran+25):
            R.append(Amp[i]*np.cos(Phase[i]))
            I.append(Amp[i]*np.sin(Phase[i]))
            x_space.append(max(R)-min(R))
            y_space.append(max(I)-min(I))
    return max(x_space), max(y_space)

def two_D2(R,I,NU,n):
    N = int(n)
    x_nu = NU[0]
    y_nu = NU[1]
    sorted_R = (max(R) + min(R))/2
    sorted_I = (max(I) + min(I))/2
    if x_nu > y_nu:
        x_axis = np.linspace(sorted_R-x_nu*0.5,sorted_R+x_nu*0.5,N)
        y_axis = np.linspace(sorted_I-x_nu*0.5,sorted_I+x_nu*0.5,N)
        step = x_nu/float(n)
    else:
        x_axis = np.linspace(sorted_R-y_nu*0.5,sorted_R+y_nu*0.5,N)
        y_axis = np.linspace(sorted_I-y_nu*0.5,sorted_I+y_nu*0.5,N)
        step = y_nu/float(n)
    
    x_pos = []
    for R_mov in range(len(R)):
        for x_mov in range(N):
            if R[R_mov] >= x_axis[x_mov] -0.6*step and R[R_mov] <= x_axis[x_mov]+0.6*step:
                break
        x_pos.append(x_mov)
    
    y_pos = []
    for I_mov in range(len(I)):
        for y_mov in range(N):
            if I[I_mov] >= y_axis[y_mov] -0.6*step and I[I_mov] <= y_axis[y_mov]+0.6*step:
                break
        y_pos.append(y_mov)
        
    twoD = np.zeros((N,N))
    for i in range(len(R)):
        twoD[x_pos[i]][y_pos[i]] = 1
        
        if i == len(R)-1:
            continue
        else:
            couple = [x_pos[i+1]-x_pos[i],y_pos[i+1]-y_pos[i]]
            if abs(couple[0]) >= abs(couple[1]) and couple[1] != 0:
                st = abs(couple[0]/couple[1])
                k = 1
            elif abs(couple[1]) >= abs(couple[0]) and couple[0] != 0:
                st = abs(couple[1]/couple[0])
                k = 0
            else:
                st = 1
                k = 2
            #------------------------
            if st -int(st) >= 0.5:
                st = int(st)+1
            else:
                st = int(st)
            #-----------------------
            if k == 1:
                vec = abs(couple[0]/st)
                if vec -int(vec) >= 0.5:
                    vec = int(vec) + 1
                else:
                    vec = int(vec)
            elif k == 0:
                vec = abs(couple[1]/st)
                if vec -int(vec) >= 0.5:
                    vec = int(vec) + 1
                else:
                    vec = int(vec)
            elif k == 2:
                if couple[1] == 0:
                    vec = abs(couple[0])
                if couple[0] == 0:
                    vec = abs(couple[1])
            step = [0,0]
            #------------------
            for line in range(vec):
                for ev_ste in range(st):
                    
                    if k == 1 :
                        step[0] += 1
                    elif k == 0:
                        step[1] += 1
                    elif k == 2 and couple[1] == 0:
                        
                        step[0] += 1
                    elif k == 2 and couple[0] == 0:
                        
                        step[1] += 1
                        
                    if couple[0] >= 0 and couple[1] >= 0:
                        dot = [step[0] + x_pos[i],step[1] + y_pos[i]]
                    elif couple[0] >= 0 and couple[1] <= 0:
                        dot = [step[0] + x_pos[i],-step[1] + y_pos[i]]
                    elif couple[0] <= 0 and couple[1] >= 0:
                        dot = [-step[0] + x_pos[i],step[1] + y_pos[i]]
                    elif couple[0] <= 0 and couple[1] <= 0:
                        dot = [-step[0] + x_pos[i],-step[1] + y_pos[i]]
                        
                    if dot[0] >= N:
                        dot[0] = N-1
                    if dot[1] >= N:
                        dot[1] = N-1
                        
                    twoD[dot[0]][dot[1]] = 1
                if k == 1:
                    step[1] += 1
                elif k == 0:
                    step[0] += 1                          
    return twoD
#get RI matrix------------------------------------
def make_twoD(data,start,stop,n,Range = 50):
    twoMat = []
    freq = data[0]
    Amp = data[1]
    phase = data[2]
    Phase = lin_pha(freq,phase)    
    
    
    Range /= 2
    Range = int(Range)
    x_space = []
    y_space = []
    for ran in range(Range,len(freq)-Range):
        R = []
        I = []
        for i in range(ran-Range,ran+Range):
            R.append(Amp[i]*np.cos(Phase[i]))
            I.append(Amp[i]*np.sin(Phase[i]))
            x_space.append(max(R)-min(R))
            y_space.append(max(I)-min(I))
    NU = [max(x_space), max(y_space)]
    for i in range(start,stop):
        R = []
        I = []
        for ii in range(i-Range,i+Range):
            R.append(Amp[ii]*np.cos(Phase[ii]))
            I.append(Amp[ii]*np.sin(Phase[ii]))
        twoMat.append(two_D2(R,I,NU,n))
    return twoMat
    
def dots(twoD):
    num = 0
    for i in twoD:
        for ii in i:
            if ii == 1:
                num += 1
    return num

def save_data(start,file_name,k,TF = 0):
    title_log = IV_Retrieve_Titles_log_py(start)# take log
    log = title_log[1]
    log_name = "0 to %s step 1" %log
    freq = []
    Amp = []
    phase = []
    
    data = freq_analysis(start,log_name,0) #get data
    freq = data[0]
    Amp = data[1]
    phase = data[2]
    if TF == "connect log":
        for i in range(log-k):
            data = freq_analysis(start,log_name,i+1)
            freq += data[0]
            Amp += data[1]
            phase += data[2]

    file = "./txt_data/"+file_name
    folder = os.path.exists(file)
    if not folder:               
        os.makedirs(file) 
    data = [freq,Amp,phase]
    data_na = ["freq","Amp","phase"]    
    k = 0    
    for name in data_na:
        f = open( "./txt_data/"+file_name+"/"+name+".txt","w")
        for i in data[k]:
            f.write(" " + str(i))
        k+=1
    return None

def for_origin(file_name,name,data):
    file = "./txt_data/origin/"+file_name
    folder = os.path.exists(file)
    if not folder:               
        os.makedirs(file) 

    k = 0    

    f = open( "./txt_data/origin/"+file_name+"/"+name+".txt","w")
    for i in range(len(data[0])):
        f.write( str(data[0][i]) + " " + str(data[1][i]) + "\n")
    return None

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