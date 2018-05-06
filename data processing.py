# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:28:06 2018

@author: Jack
"""
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
from numba import jit
import scipy.interpolate as interpolate
import scipy.optimize as op
from scipy import signal



path=r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests" + "\\"
file = r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests\0\0.npy"


@jit(cache=True)
def make_m(weight=0, accel=0):
    """Returns to total effective mass of the experiment"""
    w_motor = 1.6 #double check this
    w_wheel = 0.620 #accurate
    g = 9.809915
    return w_motor + w_wheel + weight*(1-accel/g)

@jit(cache=True)
def make_ja():
    """calculates the moment of inertia for the testing wheel.\n
    in a function because I don't like all the variables it was making.\n
    d_ is the densit in kg/m**2; _o is the outer disks, _i is the inner disk,
    _b are the bolts, and _h are the holes"""
   # global J_A
    d_o = 0.138/(((0.29996/2))**2 - 20*0.002**2 - 0.04**2)
    d_i = 0.283/((0.288/2)**2 - 20*0.002**2 - 0.04**2)
    J_h = 20*(0.002**4) + 28*(0.002**2)*(0.137**2) + 12*(0.002**2)*(0.059**2) + (0.080)**4
    J_o = d_o/2 * ((0.29996/2)**4 - J_h)
    J_i = d_i/2 * ((0.288/2)**4 - J_h)
    J_b = 0.061/40 * (14*(0.137**2) + 6*(0.059**2))
    J_A = J_o*2 + J_i + J_b     #moment of inertia for the testing wheel
    return J_A

@jit(cache=True)
def Tf(k1, k2, u2, u3, u4, m, w):
    """returns the friction for a given velocity and mass"""
    return m*k2*np.sqrt((k1*w)**2 + 2*k1*w)/k1 + u2*w + u3 + u4*m

@jit(cache=True)
def a_free(k1, k2, u2, u3, u4, m, w):
    """returns the acceleration for a given velocity for the free-spinning tests"""
    j = make_ja()
    return -Tf(k1, k2, u2, u3, u4, m, w)/j

@jit(cache=True)
def opt_free(params, files, p=False):
    """returns the sum of the average errors between the model and actual acceleration.\n
    using the average error of each run removes the bias towards longer runs.\n
    if p is true, it will plot all runs, not just the optimised one! use with caution.\n
    the cuttoff frequency should be the same for each derivative stage. optimising 
    them independantly will give us feedback as to if it's working (i.e. if 
    cut1 = cut2, it's behaving well).\n
    However: the cutoff isn't the same for each data run, and setting them the
    same gave bad results, so 2 cut-off frequencies need to be given for each data file"""
    k1, k2, u2, u3, u4 = params[:5]
    cuts = params[5:]
    c = 0
    err = 0
    m = make_m()
    for i in files:
        data = np.load(i)
        data[1] *= 2*np.pi/360
        w, a = butt_2(cuts[c], cuts[c+1], data, wend=0, plot=False)
        am = a_free(k1, k2, u2, u3, u4, m, w)
        err += np.average((am[:-1]-a)**2)
        c+=2
        if p is True:
            plt.plot(data[0][:len(a)], a)
            plt.plot(data[0][:len(am)], am)
    return err

@jit(cache=True)
def a_mass(k1, k2, u2, u3, u4, weight, w, a):
    """returns the acceleration for a given velocity for the drop tests"""
    r = 0.288/2
    j = make_ja()
    m = make_m(weight, a/r)
    tf = Tf(k1, k2, u2, u3, u4, m, w[:-1])
    return tf/(m*(r**2) - j)

@jit(cache=True)
def opt_mass(params, files, p=False):
    k1, k2, u2, u3, u4 = params[:5]
    cuts = params[5:]
    c = 0
    err = 0
    for i in files:
        data = np.load(i)
        data[1] *= 2*np.pi/360
        Ts = 10**-3
        if data[0][1] >= 0.9:
            Ts *= 10**6
        data = resample(data, Ts)
        w, a = butt_2(cuts[c], cuts[c+1], data, wend=None, plot=False)
        c+=2
        weight = int(i.split("\\")[-2])/1000
        am = a_mass(k1, k2, u2, u3, u4, weight, w, a)
        err += np.average((am-a)**2)
        if p is True:
            plt.plot(data[0][:len(a)], a)
            plt.plot(data[0][:len(am)], am)
    return err

@jit(cache=True)
def opt_all_w(params, f_files, m_files):
    #k1, k2, u2, u3, u4 = params[5:]
    #print(type(params))
    f_params = params[:5+len(f_files)*2]
    m_params = np.append(params[:5], params[5+len(f_files)*2:])
    err = opt_free(f_params, f_files) + opt_mass(m_params, m_files)
    return err

@jit(parallel=True)
def opt_all():
    """performs optimisations on all the files"""
    f_files = ['C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\14.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\15.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\16.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\17.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\18.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\3.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\4.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\5.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\6.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\7.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\8.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\9.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\10.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\11.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\12.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\13.npy']
    m_files = ['C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\2.npy']
    params = [1, 1, 1, 1, 1]
    bounds = [(1e-12, None), (0, None), (0, None), (0, None), (0, None)]
    for i in f_files:
        params += [20, 20]
        bounds += [(1, 30), (1, 30)]
    for i in m_files:
        params += [20, 20]
        bounds += [(1, 30), (1, 30)]
    res = op.minimize(opt_all_w, params, args=(f_files, m_files), bounds=bounds, tol=10**-11)
    return res

@jit(cache=True)
def m_seq_worker(params, const, k1, k2, u2, files):
    u3 = params[0]
    u4 = (const-u3)/make_m()
    m_params = np.append([k1, k2, u2, u3, u4], params[1:])
    return opt_mass(m_params, files, False)

@jit(cache=True)
def f_seq_worker(params, files):
    k1, k2, u2, const = params[:4]
    f_params = np.append([k1, k2, u2, const, 0], params[4:])
    return opt_free(f_params, files, False)

def opt_all_seq():
    """performs optimisations on all the files. does the free spinning first,
    then uses the results as constraints for the mass runs. should be much
    faster than trying to optimize all the variables at once. Note that this
    carries the  implicit assumption that the free spinning tests gave "better"
    results, which, given that those tests were longer and had less sudden events,
    they should have better results. We do still need the drop tests for the mass
    dependancy, though"""
    f_files = ['C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\14.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\15.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\16.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\17.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\18.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\3.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\4.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\5.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\6.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\7.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\8.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\9.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\10.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\11.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\12.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\13.npy']
    m_files = ['C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\2.npy']
    f_params = [1, 1, 1, 1]
    f_bounds = [(1e-12, None), (0, None), (0, None), (0, None)]
    for i in f_files:
        f_params += [20, 20]
        f_bounds += [(1, 30), (1, 30)]
    res1 = op.minimize(f_seq_worker, f_params, args=(f_files), bounds=f_bounds, tol=10**-11)
    
    k1, k2, u2, const = res1['x'][:4]
    #const = u3 + u4*make_m()
    m_params = [const/2]
    m_bounds = [(0, const)]
    for i in m_files:
        m_params += [20, 20]
        m_bounds += [(1, 30), (1, 30)]
    res2 = op.minimize(m_seq_worker, m_params, args=(const, k1, k2, u2, m_files), bounds=m_bounds, tol=10**-11)
    return res1, res2

@jit(cache=True)
def resample(data, Ts=None):
    """resamples the data so that the sampling period is constant.\n
    if Ts is None, it tries to work out the lowest possible Ts that won't
    undersample the data"""
    t = data[0]
    d = data[1]
    interp = interpolate.interp1d(t, d)
    if Ts is None:
        dt = t[1:]-t[:-1]
        Ts = min(dt)
    N = 1 + (t[-1]-t[0])/Ts
    if N != int(N):
        N += 1
    N = int(N)
    t_new = np.linspace(t[0], t[-1], N)
    d_new = interp(t_new)
    return np.array([t_new, d_new])

@jit(cache=True)
def butt_2(cut1, cut2, data, order=1, plot=True, wend=0):
    """finds the 2nd derivative of the data, and smooths with a butterworth filter.
    20 and 20 are good first estimates for cut1 and cut2."""
    t = data[0]
    f = data[1]
    dt = (t[1:] - t[:-1])
    if dt[0] > 0.9:
        dt *= 1/(10**6)
    else:
        t *= 10**6
    w = butt(data, cut1, end=wend, plot=False)
    if plot is True:
        u = (f[1:] - f[:-1])/dt
        if wend is not None:
            u = np.append(u, wend)
        plt.plot(t[:len(u)], u)
        plt.plot(t[:len(w)], w)
    filt = butt([t[:len(w)], w], cut2, plot=plot)
    if plot is not True:
        return w, filt

@jit(cache=True)
def butt(data, cut, order=1, plot=True, end=None, opt=False):
    """finds the derivative of data, and smooths with a butterworth filter"""
    t = data[0]
    f = data[1]
    dt = (t[1:] - t[:-1]) /(10**6)
    u = (f[1:]-f[:-1])/dt
    if end is not None:
        u = np.append(u, end)
    else:
        end = np.average(u[-10:])
    init = np.average(u[:10])
    cut = 2*cut/len(u)
    if opt is True:
        init = op.minimize(butt_err, init, args=(u, cut, order))['x']
    uf = np.append(init*np.ones(10), u)
    uf = np.append(uf, end*np.ones(10))
    b, a = signal.butter(order, cut, btype="low")
    filt = signal.filtfilt(b, a, uf-init) + init
    filt = filt[10:-10]
    if plot is True:
        plt.plot(t[:len(filt)], filt)
    else:
        return filt

def butt_err(init, u, cut, order):
    """worker function for butt(). finds the error between the filtered and 
    unfiltered data"""
    b, a = signal.butter(order, cut, btype="low")
    uf = np.append(init*np.ones(10), u)
    filt = signal.filtfilt(b, a, uf-init) + init
    filt = filt[10:]
    return np.sum(np.abs(filt[:10]-u[:10]))

def plot_f(params, files):
    """plots the results of the free spinning tests only"""
    k1, k2, u2, u3, u4 = params[:5]
    cuts = params[5:]
    c = 0
    for i in files:
        data = np.load(i)
        data[1] *= 2*np.pi/360
        Ts = 10**-6
        if data[0][1] >= 0.9:
            Ts *= 10**6
        w, a = butt_2(cuts[c], cuts[c+1], data, wend=None, plot=False)
        m = make_m()
        am = a_free(k1, k2, u2, u3, u4, m, w)
        c+=2
        plt.plot(data[0][:len(a)], a)
        plt.plot(data[0][:len(am)], am)

def plot_res(params, files):
    k1, k2, u2, u3, u4, cut1, cut2 = params
    for i in files:
        data = np.load(i)
        data[1] *= 2*np.pi/360
        Ts = 10**-6
        if data[0][1] >= 0.9:
            Ts *= 10**6
        weight = int(i.split("\\")[-2])/1000
        if weight != 0:
            data = resample(data, Ts)
        w, a = butt_2(cut1, cut2, data, wend=None, plot=False)
        if weight == 0:
            am = a_mass(k1, k2, u2, u3, u4, weight, w, a)
        else:
            m = make_m()
            am = a_free(k1, k2, u2, u3, u4, m, w)
        plt.plot(data[0][:len(a)], a)
        plt.plot(data[0][:len(am)], am)