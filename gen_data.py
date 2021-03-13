#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:25:06 2021

@author: arnovel
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
import scipy.interpolate as itp
import matplotlib.pyplot as plt
from jax import jit, grad
from typing import Collection, Union

# init
_seed = 112
num_points = 1000
xgrid = jnp.linspace(-10,10, num_points)
# seed
np.random.seed(_seed)

def get_random_spline(n_knots=15):
    xknots = np.linspace(-5,5, n_knots)
    yknots = np.random.randn(n_knots) * 1.5
    
    spl = itp.UnivariateSpline(xknots, yknots)
    
    return spl

def get_random_func(n_knots=15):
    
    splines = [
        get_random_spline(n_knots),
        get_random_spline(n_knots),
        get_random_spline(n_knots),
        get_random_spline(n_knots),
        ]
    
    def func(x):
        _base = splines[0](x)
        _sin = np.sin(splines[1](x))
        _rbf = np.exp(-np.power(splines[2](x), 2))
        _rat = 1.0 / (1 + np.power(splines[3](x), 2))
        
        return _base * _sin * _rbf - _rat
    
    return splines, func

def plot_splines(splines):
    fig, axes = plt.subplots(2,2, figsize=(20, 20))
    
    axes[0,0].plot(xgrid, splines[0](xgrid), lw=4)
    axes[0,1].plot(xgrid, splines[1](xgrid), lw=4)
    axes[1,0].plot(xgrid, splines[2](xgrid), lw=4)
    axes[1,1].plot(xgrid, splines[3](xgrid), lw=4)
    plt.tight_layout()
    plt.show()
    
def plot_bases(splines):
    _basegrid = splines[0](xgrid)
    _singrid = np.sin(splines[1](xgrid))
    _rbfgrid = np.exp(-np.power(splines[2](xgrid), 2))
    _ratgrid = 1.0 / (1 + np.power(splines[3](xgrid), 2))
    
    fig, axes = plt.subplots(4,2, figsize=(20, 20))
    
    axes[0,1].plot(xgrid, _basegrid, lw=4, c='r')
    axes[0,0].plot(xgrid, splines[0](xgrid), lw=4)
    axes[1,1].plot(xgrid, _singrid, lw=4, c='r')
    axes[1,0].plot(xgrid, splines[1](xgrid), lw=4)
    axes[2,1].plot(xgrid, _rbfgrid, lw=4, c='r')
    axes[2,0].plot(xgrid, splines[2](xgrid), lw=4)
    axes[3,1].plot(xgrid, _ratgrid, lw=4, c='r')
    axes[3,0].plot(xgrid, splines[3](xgrid), lw=4)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    splines, func = get_random_func(15)
    # plot_splines(splines)
    plot_bases(splines)
    ygrid = func(xgrid)
    plt.plot(xgrid, ygrid)
    plt.show()
    