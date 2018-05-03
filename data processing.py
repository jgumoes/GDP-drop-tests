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

#from mathing_the_data import *

path=r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests" + "\\"
file = r"C:\Users\Jack\Documents\Uni\GDP\Drop Tests\0\0.npy"

J_A = None
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
    global J_A
    if J_A is None:
        make_ja()
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

def make_m(weight=0, accel=0):
    """Returns to total effective mass of the experiment"""
    w_motor = 1.6 #double check this
    w_wheel = 0.620 #accurate
    return w_motor + w_wheel + weight*(1-accel/9.81)

def make_ja():
    """calculates the moment of inertia for the testing wheel.\n
    in a function because I don't like all the variable it was making.\n
    d_ is the densit in kg/m**2; _o is the outer disks, _i is the inner disk,
    _b are the bolts, and _h are the holes"""
    global J_A
    d_o = 0.138/(((0.29996/2))**2 - 20*0.002**2 - 0.04**2)
    d_i = 0.283/((0.288/2)**2 - 20*0.002**2 - 0.04**2)
    J_h = 20*(0.002**4) + 28*(0.002**2)*(0.137**2) + 12*(0.002**2)*(0.059**2) + (0.080)**4
    J_o = d_o/2 * ((0.29996/2)**4 - J_h)
    J_i = d_i/2 * ((0.288/2)**4 - J_h)
    J_b = 0.061/40 * (14*(0.137**2) + 6*(0.059**2))
    J_A = J_o*2 + J_i + J_b     #moment of inertia for the testing wheel

make_ja()

@jit(parallel=True, nopython=True)
def tik_data_w(params, f, dt, As):
    """worker function for tik_diff.
    params[0] should be alpha"""
    a = params[0]
    u = params[1:]
    #dt = t[1:] - t[:-1]
    du = (u[1:] - u[:-1])/(dt[:-1])
    A_w = (u[1:] + u[:-1])*dt[:-1]/2
    Au = np.dot(As, A_w) # Au(x) is the integral of u between 0 and x
    #Au = np.zeros(len(f)-1)
    #for i in range(len(Au)):
    #    Au[i] = np.sum(A_w[:i+1])
    L2 = (Au - f[1:])**2
    return a*np.sum(np.abs(du)) + np.sum(L2)/2

@jit(parallel=True)
def tik_diff(data, a=0.5):
    """performs differentiation on the data using Tikhonov regularization.\n
    minimizing function comes from https://www.hindawi.com/journals/isrn/2011/164564/#B16"""
    t = data[0]
    t -= t[0]
    if np.average(t[1:]-t[:-1]) < 0.9:
        t*=1000000
    t = (t+0.4999).astype(int)
    f = data[1]
    f -= f[0]
    dt = (t[1:] - t[:-1])*10**6
    u0 = (f[1:] - f[:-1])/dt
    params = np.append(a, u0)
    bounds = [(0, None)] + [(None, None)]*len(u0)
    A = np.tril(np.ones((len(t)-1, len(t)-2)))
    minim = op.minimize(tik_data_w, params, args=(f, dt, A), bounds=bounds, options={'maxiter': 10**6, 'maxfun': 10**6})
    return minim

# =============================================================================
# @jit(parallel=True, nopython=True)
# def tik_data_w(params, f):
#     """worker function for tik_diff.
#     params[0] should be alpha"""
#     a = params[0]
#     u = params[1:]
#     du = u[1:] - u[:-1]
#     A_w = u[1:] + u[:-1]
#     Au = np.zeros(len(f)-1)
#     for i in range(len(Au)):
#         Au[i] = np.sum(A_w[:i+1])/2
#     L2 = (Au - f[1:])**2
#     return a*np.sum(np.abs(du)) + np.sum(L2)/2
# 
# @jit(parallel=True)
# def tik_diff(data, a=0.5):
#     """performs differentiation on the data using Tikhonov regularization.\n
#     minimizing function comes from https://www.hindawi.com/journals/isrn/2011/164564/#B16"""
#     t_old = data[0]
#     t_old -= t_old[0]
#     t_old = (t_old+0.4999).astype(int)
#     if np.average(t_old[1:]-t_old[:-1]) < 0.9:
#         t_old*=1000000
#     th = data[1]
#     t = np.arange(0, t_old[-1]+1)
#     interp = interpolate.interp1d(t_old, th)
#     f = interp(t) - th[0]
#     u0 = f[1:] - f[:-1]
#     params = np.append(a, u0)
#     bounds = [(0, None)] + [(None, None)]*len(u0)
#     minim = op.minimize(tik_data_w, params, args=(f), bounds=bounds)
#     return minim
# =============================================================================
    
    
    
    