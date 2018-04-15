# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:02:55 2018

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.interpolate as interpolate
from lmfit import minimize, Parameters
from math import log as log

path=r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests" + "\\"
"""575 g start and length is [167, 167+77], [625, 625+77], [1020, 1020+77],
    77 (200 for full test)
    savgol parameters are 11, 3

1100 g start and length is [201, 633, 1086], 71

1667 g start and length is [179, 655, 1069], 71

free spinning.txt starts and ends: [64: 2613],[2685: 4809], [4849: 6010]
[6045: 7561], [7589: 9311], [9349: 10837], [10858: 12505], [12529: 14345]
[14371: 15474], [15493: 16532], [16549: 18043], [18073: 19331], [19340: 19977]
[19995: 21478], [21500: 22763], [22781: 23864], [23888: 24970], [24990: 26291]
[26349: 29501]
    
usage procedure:
-raw_data() to import the data
-plot_angle() to see where to trim the data (only keep the sharp accelerations)
-break_three() on each chunk of data to break the data into each individual sensor
-brute_force() using the output of break_three() for each data chunk to optimise
    the spacing between each sensor (this accounts for the sensors not having
    identical spacing)
-combine_three(alpha=False) to get each optimised chunk. use the output of
    break_three() and the D values found in brute_force(). Set filt=True to
    return the interpolation-filtered values

to convert conditioned data chunks to files:
    alldata = raw_data(path)[0]; c=0
    ind = np.array([[lower, upper]....])
    
    while c < len(ind):
        res = plot_opt(alldata[ind[c][0]: ind[c][1]])
        np.save(path+"1667\\%i" % (c), res)
        c+=1
"""

theta = 360/30 # degrees per pulse

def raw_data(path):
    try:
        path
    except SyntaxError:
        path = "%r" % path # stops the character escapes from happening
    file = np.loadtxt(path)
    mass = (path.split("\\")[-1].split()[0])
    return file, mass

def plot_angle(data, show_t=True, theta=360/90):
    """plots the angle against time. doesn't account for reverse motion"""
    times = data/10**6
    times-=times[0]
    angles = np.linspace(0, theta*len(times), len(times), endpoint=False)
    vel = theta/(times[1:]-times[:-1])
    filt_vel = signal.savgol_filter(vel, 11, 3)
    a_times = times[1:-1]
    accel = (filt_vel[1:]-filt_vel[:-1])/(times[2:]-times[:-2])
    if show_t is True:
        plt.plot(times[:-1], vel)
        plt.scatter(times[:-1], vel)
        #plt.plot(times[:-1], filt_vel)
    else:
        plt.plot(vel)
    #plt.plot(a_times, accel)
    #plt.plot(times)
    

def break_three(times, p=False, trim=False):
    """breaks the combined data into each individual sensor\n
    tries to optimise the angle between edges (i.e. account for magnetic threshold)"""
    theta = 360/30 # degrees per pulse
    data = times/10**6
    #print(data[0])
    data -= data[0]
    #print(len(data))
    a_t = data*np.append(np.tile([1, 0, 0], int(len(data)/3)), [1, 0, 0][:len(data)%3])
    a_t = np.append(np.array([0]), a_t[np.nonzero(a_t)])
    #print(len(a_t))
    a_A = np.linspace(0, theta*len(a_t), len(a_t), endpoint=False)
    a_w = (a_A[1:]-a_A[:-1])/(a_t[1:]-a_t[:-1])
    #plt.plot(a_t[:-1], a_w)
    da = find_d(a_t)
    a_A += da*theta*np.append(np.tile([1, -1], int(len(a_t)/2)), [1][:len(a_t)%2])
    a_A -= da*theta
    a_w = (a_A[1:]-a_A[:-1])/(a_t[1:]-a_t[:-1])
    
    b_t = data*np.append(np.tile([0, 1, 0], int(len(data)/3)), [0, 1, 0][:len(data)%3])
    b_t = b_t[np.nonzero(b_t)]
    b_A = np.linspace(0, theta*len(b_t), len(b_t), endpoint=False)
    db = find_d(b_t)
    b_A += db*theta*np.append(np.tile([1, -1], int(len(b_t)/2)), [1][:len(b_t)%2])
    b_A -= da*theta
    b_w = (b_A[1:]-b_A[:-1])/(b_t[1:]-b_t[:-1])
    
    c_t = data*np.append(np.tile([0, 0, 1], int(len(data)/3)), [0, 0, 1][:len(data)%3])
    c_t = c_t[np.nonzero(c_t)]
    c_A = np.linspace(0, theta*len(c_t), len(c_t), endpoint=False)
    dc = find_d(c_t)
    c_A += dc*theta*np.append(np.tile([1, -1], int(len(c_t)/2)), [1][:len(c_t)%2])
    c_A -= da*theta
    c_w = (c_A[1:]-c_A[:-1])/(c_t[1:]-c_t[:-1])
    
    alpha = (a_w[1:]-a_w[:-1])/(a_t[2:]-a_t[:-2])
    
    if p is True:
        plt.plot(a_t[:-1], a_w)
        plt.plot(b_t[:-1], b_w)
        plt.plot(c_t[:-1], c_w)
        #plt.plot(a_t[:-2], alpha)
        #plt.scatter(b_t, b_A)
        #plt.scatter(c_t, c_A)
    if trim is True:
        lens = [len(a_A), len(c_A), len(b_A)]
        length = min(lens)
        
        a_data = np.vstack([a_t[:length], a_A[:length]])
        b_data = np.vstack([b_t[:length], b_A[:length]])
        c_data = np.vstack([c_t[:length], c_A[:length]])
    else:
        a_data = np.vstack([a_t, a_A])
        b_data = np.vstack([b_t, b_A])
        c_data = np.vstack([c_t, c_A])
    return a_data, b_data, c_data
    
def combine_three(data, D1, D2, sav=[13, 2], p=False, alpha=True, filt=False):
    """recombines the three data streams from break_three() into 1 stream"""
    a_data, b_data, c_data = data
    a_t, a_A = a_data
    b_t, b_A = b_data
    c_t, c_A = c_data
    #lens = [len(a_A), len(c_A), len(b_A)]
    #length = min(lens)
    end = None
    if len(a_t) > len(c_t):
        c_t = np.append(c_t, c_t[-1]+1)
        c_A = np.append(c_A, c_A[-1]+1)
        end = -1
    if len(a_t) > len(b_t):
        b_t = np.append(b_t, b_t[-1]+1)
        b_A = np.append(b_A, b_A[-1]+1)
        end = -2
    
    A = np.stack([a_A, b_A+D1+theta/3., c_A+D2+2*theta/3.]).flatten('F')[:end]
    T = np.stack([a_t, b_t, c_t]).flatten('F')[:end]
    W = (A[1:]-A[:-1])/(T[1:]-T[:-1])
    #plt.plot(T, A)
    if p is True:
        plt.plot(T[:-1], W)
    #filt_W = signal.savgol_filter(W, sav[0], sav[1], mode="nearest")
    #filt_times, filt_W = interp(data,W=True)
    data[1][1]+=D1
    data[2][1]+=D2
    filt_times, filt_a = interp(data, alpha=alpha)
    l = np.argwhere(T==filt_times[0])[0][0]
    h = np.argwhere(T==filt_times[-1])[0][0]+1
    W = W[l: h]
    a = (W[1:]-W[:-1])/(filt_times[1:]-filt_times[:-1])
    if p is True:
        plt.plot(filt_times, filt_a)
        #plt.plot(filt_times[:-1], a)
        #plt.plot(filt_times[:-1], filt_a)
    #print(D1,"\t", D2, "\t", np.sum((W-filt_W)**2))
    #return np.sum((W[3:-3]-filt_W)**2)
    if filt is True:
        return T, interp(data, alpha=False, W=False)
    if alpha is True:
        return np.sum((a-filt_a)**2)
    return T, A
    
def interp(dataset, alpha=True, W=True, p=False, tail=False):
    """combines the sensor data through interpolation\n
    dataset should be a list of numpy arrays, where each array has the form 
    [[times], [angles]]"""
    if type(dataset[0]) is tuple:
        dataset = dataset[0]
    n = len(dataset)
    newtimes = np.array([])
    for i in dataset:
        newtimes = np.append(newtimes, i[0])
    newtimes.sort()
    if tail is not False:
        tailtimes_s = newtimes[:n]
        tailtimes_e = newtimes[-n:]
    #print(newtimes[:5])
    #print(tailtimes_s, "\t", tailtimes_e)
    newtimes = newtimes[n-1:1-n]
    average = np.zeros(len(newtimes))
    for i in dataset:
        average += (interpolate.interp1d(i[0], i[1])(newtimes))
        #print(i[0][1], "\t", i[1][1])
    average = average/n
        
    
    if tail == "start" or tail =="both":
        int_dict = {}
        c = 0
        for i in dataset:
            int_dict["%i" % c] = interpolate.interp1d(i[0][:2], i[1][:2])
            c+=1
        c = 0
        s_r = np.zeros(n)
        for s in tailtimes_s:
            n = 0
            ic = 0
            #print(s)
            for i in dataset:
                if i[0][0] <= s:
                    s_r[c] += int_dict["%i" % ic](s)
                    #print(s_r[c])
                    n+=1
                ic += 1
            s_r[c] = s_r[c]/n
            c+=1
        #print(s_r)
        average[0] = (s_r[-1]+average[0])/2
        average = np.append(s_r[:-1], average)
        newtimes = np.append(tailtimes_s[:-1], newtimes)
    
    if tail == "end" or tail =="both":
        int_dict = {}
        c = 0
        for i in dataset:
            int_dict["%i" % c] = interpolate.interp1d(i[0][-2:], i[1][-2:])
            c+=1
        c = 0
        e_r = np.zeros(n)
        for e in tailtimes_e:
            n = 0
            ic = 0
            for i in dataset:
                if i[0][-1] >= e:
                    e_r[c] += int_dict["%i" % ic](e)
                    n+=1
                ic += 1
            e_r[c] = e_r[c]/n
            c+=1
        average[-1] = (e_r[0]+average[-1])/2
        average = np.append(average, e_r[-2:])
        newtimes = np.append(newtimes, tailtimes_e[1:])

    w = (average[1:]-average[:-1])/(newtimes[1:]-newtimes[:-1])
    w_times = newtimes[:-1]
    a = (w[1:]-w[:-1])/(w_times[1:]-w_times[:-1])
    
    if alpha is True:
        if p is True:
            plt.plot(w_times[:-1], a)
        return w_times, a
    if W is True:
        if p is True:
            plt.plot(newtimes[:-1], w)
        return newtimes[:-1], w
    if p is True:
        plt.plot(newtimes, average)
    return newtimes, average


def find_d(times):
    """finds the d (the pulse offset value) by minimizing n_peaks."""
    params = Parameters()
    params.add('delta', value=0, min=-1, max=1)
    out = minimize(square_filt_d, params, args=(times, None))
    return out.params.valuesdict()['delta']


def square_filt_d(param, t, dummy, sav=[11, 2]):
    """Returns the square of the difference between unfiltered and filtered data.\n
    param is the lmfit parameter for delta \n
    dummy is an empty variable. Has to be there for the code to work"""
    if type(param) is float:
        da = param
    elif type(param) is int:
        da=param
    else:
        da = param.valuesdict()['delta']
    #da=param
    theta = 360/30
    A = np.linspace(0, theta*len(t), len(t), endpoint=False)
    A += da*theta*np.append(np.tile([1, -1], int(len(t)/2)), [1][:len(t)%2])
    W = (A[1:]-A[:-1])/(t[1:]-t[:-1])
    filt_W = signal.savgol_filter(W, sav[0], sav[1])
    return np.sum((W-filt_W)**2)


def brute_force(data, tol=360/30, it=None, res=10**6):
    """a brute force algorithm that optimises plot three./n
    this has to exist because multi-dimensional scipy doesn't work for shit"""
    N=5
    if it is None:
        it = 1+int(log(res, N)) # auto-range the iterations.
    print("Number of iterations:\t", it)
    D1_l = -tol
    D1_h = tol
    D2_l = -tol
    D2_h = tol
    for z in range(it):
        i = 0   #D1 index
        j = 0   #D2 index
        D1_list = np.linspace(D1_l, D1_h, N)
        D2_list = np.linspace(D2_l, D2_h, N)
        res = np.zeros((N, N))
        print("Iteration:\t", z)
        while i < N:
            j=0
            while j<N:
                r = combine_three(break_three(data), D1_list[i], D2_list[j])
                res[i, j] +=  r
                j+=1
            i+=1
        min_value = res.min()
        min_pos = np.argwhere((res==min_value))[0]
        D1_l = D1_list[min_pos[0]-1]
        D1_h = D1_list[min_pos[0]+1]
        D2_l = D2_list[min_pos[1]-1]
        D2_h = D2_list[min_pos[1]+1]
    return min_value, min_pos, D1_list[min_pos[0]], D2_list[min_pos[1]]

def plot_opt(data, alpha=False):
    """a function to help optimise chunk size. it accepts a chunk of data,
    breaks, optimises and recombines, and plots the result, all in one step"""
    D1, D2 = brute_force(data)[-2:]
    res = combine_three(break_three(data), D1, D2, p=True, alpha=alpha, filt=False)
    return res

def combine(data, start, length):
    """combines the data from the trials into one\n
    this is poorly implemented and might not actually be used"""
    s = np.zeros(length)
    for i in start:
        s+=data[i:i+length]
    return s/len(start)

#data1 = combine(raw_data(path+"575 g.txt")[0], [167, 625, 1020], 71)
#data2 = combine(raw_data(path+"1100 g.txt")[0], [201, 633, 1086], 71)
#data3 = combine(raw_data(path+"1667 g.txt")[0], [179, 655, 1069], 71)
