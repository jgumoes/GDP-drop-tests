# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:02:55 2018

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.interpolate as interpolate
from lmfit import minimize, Parameters, report_fit
from math import log as log

path=r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests" + "\\"
"""575 g start and length is [167, 625, 1020], 77 (200 for full test)
        savgol parameters are 11, 3

1100 g start and length is [201, 633, 1086], 71

1667 g start and length is [179, 655, 1069], 71

free spinning.txt starts and ends: [64: 2614], [2685: 4810], [4849: 6011]
[6036: 7562], [7589: 9312], [9349: 10838], [10865: 13506], [12529: 14346]
[14371: 15475], [15493: 16533], [16549: 18044], [18073: 19331], [19340: 19767]
[19995: 21476], [21497: 22764], [22781: 23865], [23888: 24971], [24990: 26291]
[26341: 29501]
    
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
    
"""

theta = 360/30 # degrees per pulse

d_o = 0.138/(((0.29996/2))**2 - 20*0.002**2 - 0.04**2)
d_i = 0.283/((0.288/2000)**2 - 20*0.002**2 - 0.04**2)
J_h = 20*(0.002**4) + 28*(0.002**2)*(0.137**2) + 12*(0.002**2)*(0.059**2) + (0.080)**4
J_o = d_o/2 * ((0.29996/2)**4 - J_h)
J_i = d_o/2 * ((0.288/2)**4 - J_h)
J_b = 0.061/40 * (14*(0.137**2) + 6*(0.059**2))
J_A = J_o*2 + J_i + J_b     #moment of inertia for the testing wheel

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
    

def break_three(times, pc=False, p=False):
    """breaks the combined data into each individual sensor\n
    tries to optimise the angle between edges (i.e. account for magnetic threshold)"""
    theta = 360/30 # degrees per pulse
    data = times/10**6
    #print(data[0])
    data -= data[0]
    #print(data[0])
    a_t = data*np.append(np.tile([1, 0, 0], int(len(data)/3)), [1, 0, 0][:len(data)%3])
    a_t = np.append(np.array([0]), a_t[np.nonzero(a_t)])
    #print(a_t)
    a_A = np.linspace(0, theta*len(a_t), len(a_t), endpoint=False)
    a_w = (a_A[1:]-a_A[:-1])/(a_t[1:]-a_t[:-1])
    #plt.plot(a_t[:-1], a_w)
    da = find_d(a_t)
    a_A += da*theta*np.append(np.tile([1, -1], int(len(a_t)/2)), [1][:len(a_t)%2])
    a_w = (a_A[1:]-a_A[:-1])/(a_t[1:]-a_t[:-1])
    
    b_t = data*np.append(np.tile([0, 1, 0], int(len(data)/3)), [0, 1, 0][:len(data)%3])
    b_t = b_t[np.nonzero(b_t)]
    b_A = np.linspace(0, theta*len(b_t), len(b_t), endpoint=False)
    db = find_d(b_t)
    b_A += db*theta*np.append(np.tile([1, -1], int(len(b_t)/2)), [1][:len(b_t)%2])
    b_w = (b_A[1:]-b_A[:-1])/(b_t[1:]-b_t[:-1])
    
    c_t = data*np.append(np.tile([0, 0, 1], int(len(data)/3)), [0, 0, 1][:len(data)%3])
    c_t = c_t[np.nonzero(c_t)]
    c_A = np.linspace(0, theta*len(c_t), len(c_t), endpoint=False)
    dc = find_d(c_t)
    c_A += dc*theta*np.append(np.tile([1, -1], int(len(c_t)/2)), [1][:len(c_t)%2])
    c_w = (c_A[1:]-c_A[:-1])/(c_t[1:]-c_t[:-1])
    
    alpha = (a_w[1:]-a_w[:-1])/(a_t[2:]-a_t[:-2])
    
    if p is True:
        plt.plot(a_t[:-1], a_w)
        plt.plot(b_t[:-1], b_w)
        plt.plot(c_t[:-1], c_w)
        #plt.plot(a_t[:-2], alpha)
        #plt.scatter(b_t, b_A)
        #plt.scatter(c_t, c_A)
    lens = [len(a_A), len(c_A), len(b_A)]
    length = min(lens)
    
    a_data = np.vstack([a_t[:length], a_A[:length]])
    b_data = np.vstack([b_t[:length], b_A[:length]])
    c_data = np.vstack([c_t[:length], c_A[:length]])
    return a_data, b_data, c_data
    
def combine_three(data, D1, D2, sav=[13, 2], p=False, alpha=True, filt=False):
    """recombines the three data streams from break_three() into 1 stream"""
    a_data, b_data, c_data = data
    a_t, a_A = a_data
    b_t, b_A = b_data
    c_t, c_A = c_data
    lens = [len(a_A), len(c_A), len(b_A)]
    length = min(lens)
    
    A = np.stack([a_A[:length], b_A[:length]+D1+theta/3., c_A[:length]+D2+2*theta/3.]).flatten('F')
    T = np.stack([a_t, b_t, c_t]).flatten('F')
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
        return interp(data, alpha=False, W=False)
    if alpha is True:
        return np.sum((a-filt_a)**2)
    return T, A, np.sum((W-filt_a)**2)
    
def interp(dataset, alpha=True, W=True, p=False):
    """combines the sensor data through interpolation\n
    dataset should be a list of numpy arrays, where each array has the form 
    [[times], [angles]].\n
    also can function as a pre-differentiation filter: thanks to the laplace 
    transform, differentiating has the transfer function G(s) = s, thereby acting
    as a high-pass filter. High-frequency noise is magnified by differentiation.
    By interpolating, the number of samples increases, effectively decreasing
    the digital frequency of data without changing its shape at all. because the 
    noise frequency is lowered, it's gain is also lowered."""
    if type(dataset[0]) is tuple:
        dataset = dataset[0]
    n = len(dataset)
    newtimes = np.array([])
    for i in dataset:
        newtimes = np.append(newtimes, i[0])
    newtimes.sort()
    newtimes = newtimes[n:-n]
    average = np.zeros(len(newtimes))
    for i in dataset:
        average += (interpolate.interp1d(i[0], i[1])(newtimes))
    average = average/n
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
