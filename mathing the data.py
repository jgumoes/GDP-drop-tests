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

path=r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests\\"
"""575 g start and length is [167, 625, 1020], 77 (200 for full test)
        savgol parameters are 11, 3

1100 g start and length is [201, 633, 1086], 71

1667 g start and length is [179, 655, 1069], 71
"""

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
    

def plot_three(times, da=0):
    """plots the three sensors individually. accepts offsets of each sensor to
    account for non-identical pulse angles"""
    theta = 360/30 # degrees per pulse
    data = times/10**6
    data -= data[0]
    a_t = data*np.append(np.tile([1, 0, 0], int(len(data)/3)), [1, 0, 0][:len(data)%3])
    a_t = a_t[np.nonzero(a_t)]
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
    
    #plt.plot(a_t[:-1], a_w)
    #plt.plot(a_t[:-2], alpha)
    #plt.scatter(b_t, b_A)
    #plt.scatter(c_t, c_A)
    #peaks = signal.find_peaks_cwt(a_w, [0.05, 0.5, 1], gap_thresh=0)
    #plt.scatter(a_t[peaks], a_w[peaks])
    lens = [len(a_A), len(c_A), len(b_A)]
    length = min(lens)
    
    A = np.stack([a_A[:length], b_A[:length]+theta/3., c_A[:length]+2*theta/3.]).flatten('F')
    T = data[:len(A)]
    W = (A[1:]-A[:-1])/(T[1:]-T[:-1])
    #plt.plot(T, A)
    plt.plot(T[:-1], W)
    filt_W = signal.savgol_filter(W, 13, 2)
    plt.plot(T[:-1], filt_W)


def square_filt_D(param, t, dummy, sav=[11, 2]):
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

def find_d(times):
    """finds the d (the pulse offset value) by minimizing n_peaks."""
    params = Parameters()
    params.add('delta', value=0, min=-1, max=1)
    out = minimize(square_filt_d, params, args=(times, None), method="nelder")
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

def interp(times, angles, theta=360/90):
    """combines the sensor data through interpolation, so that the time
    increments in microseconds"""
    lens = []
    zero = data[0][0]
    for i in times:
        i-=zero
        lens.append(i[-1])
    length = min(lens)
    step = data[0][1]
    

data1 = combine(raw_data(path+"575 g.txt")[0], [167, 625, 1020], 71)
data2 = combine(raw_data(path+"1100 g.txt")[0], [201, 633, 1086], 71)
data3 = combine(raw_data(path+"1667 g.txt")[0], [179, 655, 1069], 71)
