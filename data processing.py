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

#J_A = None
#print(raw_data(path+"free spinning.txt"))

# =============================================================================
# from sympy import *
# init_printing(use_unicode=True)
# mu1, mu2, mu3, mu4, mu5, n, j, m, t = symbols("mu1, mu2, mu3, mu4, mu5, n, j, m, t")
# o = Function("theta")
# w = Function("omega")
# a = diff(w(t), t)
# Tf = mu1*m*w(t) + mu2*w(t) + mu3*(w(t)**n) + mu4 + mu5*m
# omega = lambdify((mu1, mu2, mu3, mu4, mu5, n, j, m, t), dsolve(Eq(-Tf.subs(n, 0)/j, a)))
# =============================================================================

def solve_theta(t, mu1, mu2, mu4, mu5, j, m, w0):
    """analytical solution for theta, assuming mu3 is 0.\n
    function is not reveresed"""
    k1 = mu1*m + mu2
    k2 = mu4 + mu5*m
    c2 = j*(w0 + k2/k1)/k1
    return -k2*t/k1 -c2*np.exp(-k1*t/j) + c2

def accel_free(t, w, k1, k2, mu3, n, j, m):
    """The equation for the acceleration of the free-spinning wheel"""
    Tf = k1*w + mu3*(w**n) + k2
    return -Tf/j

def theta_free(t, k1, k2, j, w0):
    """analytical solution for theta, assuming mu3 is 0.\n
    the constants make it suitable for the free spinning data.\n
    function is not reveresed."""
    c2 = j*(w0 + k2/k1)/k1
    return -k2*t/k1 -c2*np.exp(-k1*t/j) + c2

def opt_theta_free(params, t, th, j, w0):
    """The lmfit model for theta_free"""
    k1 = params['k1']
    k2 = params['k2']
    #m = make_m(0, 0)
    model = theta_free(t, k1, k2, j, w0)
    return np.abs(model**2 - th**2)

def fit_theta_free(data):
    """Fits the curve for theta_free. Data should be the times and angles"""
    t = data[0]
    th = data[1]
    w0 = (th[1]-th[0])/(t[1]-t[0])
    params = lm.Parameters()
    params.add("k1", value = 0.5, min=0)
    params.add("k2", value = 0.5, min=0)
    J_A = make_ja()
    minner = lm.Minimizer(opt_theta_free, params, fcn_args=(t, th, J_A, w0))
    return minner.minimize()
    
    

# =============================================================================
# def fit(data, m=None, func=solve_theta):
#     """fit the data to curve. j, w0 and m should be and the end of the paramaters"""
#     curve = lm.Model(theta_free)
#     params = curve.make_params()
#     global J_A
#     if J_A is None:
#         J_A = make_ja()
#     params["j"].set(min=J_A, max=J_A)
#     params["m"].set(min=m, max=m)
#     params["w0"].set(min=data[1][0], max=data[1][0])
#     for p in curve.param_names[:-3]:
#         params[p].set(value=0.5, min=0)
#     minim = lm.Minimizer(curve, params, fcn_args=(curve.independent_vars))
#     res = minim.minimize()
#     return res
# =============================================================================
    

@jit
def solve_w(t, k1, k2, mu3, n, j, m, w0=0, res=10, plot=False):
    """differentiates the acceleration equation to find w(t).\n
    uses euler's method to sub-microsecond precision (1us/res).\n
    res is the number of points between each microsecond"""
    times = t
    times -= times[0]
    if times[1]<1:
        times*=10**6
    times_out = np.linspace(times[0], times[-1], int((times[-1]-times[0])*res))
    w_out = np.zeros(len(times_out))
    w_out[0] = w0
    ind = 0
    delta=1/res
    while ind < len(w_out)-1:
        time = times_out[ind]
        w_out[ind+1] = w_out[ind] + delta*accel_free(time, w_out[ind], k1, k2, mu3, n, j, m)
        ind+=1
        #if (ind>100000) and (ind%100000 == 0):
        #    print(ind)
    np.save(np.array(times_out, w_out), "w, euler forwards, res=%d" % res)
    if plot is True:
        plt.plot(times_out, w_out)
        
def make_th_simple(t, k1, k2, mu3, n, j, m, w0=0, res=100):
    """finds theta by taking the second derivative of negative acceleration 
    using eulers method. uses constant-mass simplification for the long 
    free-spinning tests.\n
    t should be in microseconds"""
    times = np.append(0, np.linspace(0, t, t*res))
    thetas = np.zeros(t*res + 1)
    p = 0
    #while p<
    
    

def find_w(t, a, end=False, plot=False):
    """finds the angular velocity.\n
    if end is given, it adds end to the end of the data i.e. end=0 for the 
    free spinning data"""
    w = (a[1:]-a[:-1])/(t[1:]-t[:-1])
    if end is False:
        t = t[:-1]
    else:
        w = np.append(w, end)
    if plot is True:
        plt.plot(t, w)
    return t, w

def invert(t, w, plot=False):
    """inverts the data to make it backwards. useful for optimising the
    free spinning data against euler's method.\n
    the time inversion conserves the interval spacing, so the change in angle
    is valid."""
    #c = np.median(w)
    w_inv = w #2*c - w
    ct = np.median(t)
    times = ct*2 - t
    times -= min(times)
    if plot is True:
        plt.plot(times, w_inv)
    return times, w_inv

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
    k1, k2, u2, u3, u4, cut1, cut2 = params
    err = 0
    m = make_m()
    for i in files:
        data = np.load(i)
        data[1] *= 2*np.pi/360
        w, a = butt_2(cut1, cut2, data, wend=0, plot=False)
        am = a_free(k1, k2, u2, u3, u4, m, w)
        err += np.average((am[:-1]-a)**2)
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
    k1, k2, u2, u3, u4, cut1, cut2 = params
    err = 0
    for i in files:
        data = np.load(i)
        data[1] *= 2*np.pi/360
        Ts = 10**-6
        if data[0][1] >= 0.9:
            Ts *= 10**6
        data = resample(data, Ts)
        w, a = butt_2(cut1, cut2, data, wend=None, plot=False)
        weight = int(i.split("\\")[-2])/1000
        am = a_mass(k1, k2, u2, u3, u4, weight, w, a)
        err += np.average((am-a)**2)
        if p is True:
            plt.plot(data[0][:len(a)], a)
            plt.plot(data[0][:len(am)], am)
    return err

@jit(cache=True)
def opt_all_w(params):
    k1, k2, u2, u3, u4, cut1, cut2, cut3, cut4 = params
    f_files = ['C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\14.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\15.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\16.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\17.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\18.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\3.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\4.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\5.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\6.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\7.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\8.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\9.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\10.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\11.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\12.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\0\\13.npy']
    m_files = ['C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\575\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\2.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1100\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\0.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\1.npy', 'C:\\Users\\Jack\\Documents\\Uni\\GDP\\Drop Tests\\1667\\2.npy']
    f_params = [k1, k2, u2, u3, u4, cut1, cut2]
    m_params = [k1, k2, u2, u3, u4, cut3, cut4]
    err = opt_free(f_params, f_files) + opt_mass(m_params, m_files)
    return err

@jit(parallel=True)
def opt_all():
    """performs optimisations on all the files"""
    params = [1, 1, 1, 1, 1, 20, 20, 20, 20]
    bounds = [(1e-06, None), (0, None), (0, None), (0, None), (0, None), (1, 30), (1, 30), (1, 30), (1, 30)]
    res = op.minimize(opt_all_w, params, bounds=bounds, tol=10**-11)
    return res

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
        am = a_mass(k1, k2, u2, u3, u4, weight, w, a)
        plt.plot(data[0][:len(a)], a)
        plt.plot(data[0][:len(am)], am)