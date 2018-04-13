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

path=r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests\\"
"""575 g start and length is [167, 625, 1020], 77 (200 for full test)
        savgol parameters are 11, 3

1100 g start and length is [201, 633, 1086], 71

1667 g start and length is [179, 655, 1069], 71
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
    mass = int((path.split("\\")[-1].split()[0]))
    return file, mass

def plot_angle(data, theta=360/90):
    """plots the angle against time. doesn't account for reverse motion"""
    times = data/10**6
    times-=times[0]
    #theta = 360/90. # angle between each edge
    angles = np.linspace(0, theta*len(times), len(times), endpoint=False)
    vel = theta/(times[1:]-times[:-1])
    filt_vel = signal.savgol_filter(vel, 11, 3)
    a_times = times[1:-1]
    accel = (filt_vel[1:]-filt_vel[:-1])/(times[2:]-times[:-2])
    plt.plot(times[:-1], vel)
    peaks = signal.find_peaks_cwt(vel, [0.05, 0.5, 1, 1.5, 2], gap_thresh=0, min_length=0.2)
    plt.scatter(times[peaks], vel[peaks])
    #plt.plot(times[:-1], filt_vel)
    #plt.plot(vel)
    #plt.plot(a_times, accel)
    #plt.plot(times)

def combine(data, start, length):
    """combines the data from the trials into one"""
    s = np.zeros(length)
    for i in start:
        s+=data[i:i+length]
    return s/len(start)
    

def plot_three(times, pc=False, p=False):
    """plots the three sensors individually. accepts offsets of each sensor to
    account for non-identical pulse angles\n"""
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
    
def combine_three(data, D1, D2, sav=[13, 2], p=False):
    """recombines the three data streams from plot_three() into 1 stream"""
    a_data, b_data, c_data = data
    a_t, a_A = a_data
    b_t, b_A = b_data
    c_t, c_A = c_data
    lens = [len(a_A), len(c_A), len(b_A)]
    length = min(lens)
    
    #D1, D2 = find_D(data, [a_A, b_A, c_A])
    #D2 = find_D(data, [a_A, b_A, c_A])[1]
    A = np.stack([a_A[:length], b_A[:length]+D1+theta/3., c_A[:length]+D2+2*theta/3.]).flatten('F')
    T = np.stack([a_t, b_t, c_t]).flatten('F')
    W = (A[1:]-A[:-1])/(T[1:]-T[:-1])
    #plt.plot(T, A)
    if p is True:
        plt.plot(T[:-1], W)
    #filt_W = signal.savgol_filter(W, sav[0], sav[1], mode="nearest")
    filt_times, filt_W = interp(data,W=True)
    if p is True:
        plt.plot(T[:-1], W)
        plt.plot(filt_times, filt_W)
    #print(D1,"\t", D2, "\t", np.sum((W-filt_W)**2))
    return np.sum((W[3:-3]-filt_W)**2)
    A = np.stack([a_A[:length], b_A[:length]+theta/3., c_A[:length]+2*theta/3.]).flatten('F')
    W = (A[1:]-A[:-1])/(T[1:]-T[:-1])
    #plt.plot(T[:-1], W)
    
def interp(dataset, W=True, p=False):
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
    newtimes = newtimes[n:-n]
    average = np.zeros(len(newtimes))
    for i in dataset:
        average += (interpolate.interp1d(i[0], i[1])(newtimes))
    average = average/n
    if W is True:
        w = (average[1:]-average[:-1])/(newtimes[1:]-newtimes[:-1])
        if p is True:
            plt.plot(newtimes[:-1], w)
        return newtimes[:-1], w
    if p is True:
        plt.plot(newtimes, average)
    return newtimes, average
    


def find_D(times, angles):
    """finds the d (the pulse offset value) by minimizing n_peaks.\n
    this function is for re-combining the 3 sensors into one dataset"""
    theta = 2*360/90
    a_A, b_A, c_A = angles
    lens = [len(a_A), len(c_A), len(b_A)]
    length = min(lens)
    #angles = np.stack([a_A[:length], b_A[:length]+theta/3., c_A[:length]+2*theta/3.]).flatten('F')
    #times = times[:len(angles)]
    angles = np.vstack((a_A[:length], b_A[:length], c_A[:length]))
    angles += np.array([[0], [1], [2]])*theta/3
    angles = angles.flatten()
    params = Parameters()
    params.add('D1', value=0) #, min=-theta, max=theta)
    params.add('D2', value=0) #, min=-theta, max=theta)
    data = np.concatenate([times, angles])
    out = minimize(square_filt_D, params, args=([data]))
    return out.params.valuesdict()['D1'], out.params.valuesdict()['D2']

def square_filt_D(param, data):
    """Returns the square of the difference between unfiltered and filtered data.\n
    param is the lmfit parameter for delta \n
    dummy is an empty variable. Has to be there for the code to work.\n
    this function is for re-combining the 3 sensors into one dataset"""
    sav=[13, 2]
    data = data.reshape((2, int(len(data)/2)))
    times, angles = data
    D1 = param.valuesdict()['D1']
    D2 = param.valuesdict()['D2']
    A = angles
    #A = (angles.reshape((3, int(len(angles))/3)) + np.array([[0], [D1], [D2]])).flatten()
    A += np.append(np.tile([0, D1, D2], int(len(A)/3)), [0, D1, D2][:len(A)%3])
    T = times[:len(A)]
    W = (A[1:]-A[:-1])/(T[1:]-T[:-1])
    filt_W = signal.savgol_filter(W, sav[0], sav[1])
    return np.sum((W-filt_W)**2), np.sum((W-filt_W)**2)


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
    alpha = (W[1:]-W[:-1])/(t[2:]-t[:-2])
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
                r = combine_three(plot_three(data), D1_list[i], D2_list[j])
                res[i, j] +=  r
                #print(D1_list[i], "\t", D2_list[j], "\t", r)
                j+=1
            i+=1
            #min_pos = res.argmin(axis=1)
            #min_vals = np.zeros(len(min_pos))
            #for i in range(len(min_pos)):
            #    min_vals[i] = res[i][min_pos[0]]
            #mins = res
        min_value = res.min()
        #print(min_value)
        min_pos = np.argwhere((res==min_value))[0]
        #print(res)
        #print(min_pos)
        D1_l = D1_list[min_pos[0]-1]
        D1_h = D1_list[min_pos[0]+1]
        D2_l = D2_list[min_pos[1]-1]
        D2_h = D2_list[min_pos[1]+1]
    return min_value, min_pos, D1_list[min_pos[0]], D2_list[min_pos[1]]

#data1 = combine(raw_data(path+"575 g.txt")[0], [167, 625, 1020], 71)
#data2 = combine(raw_data(path+"1100 g.txt")[0], [201, 633, 1086], 71)
#data3 = combine(raw_data(path+"1667 g.txt")[0], [179, 655, 1069], 71)
