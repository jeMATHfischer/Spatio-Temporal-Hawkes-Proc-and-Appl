#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:33:48 2018

@author: jeMATHfischer

A python class for a Simulation of nonlinear Hawkes Processes based on Ogatas thinning algorithm. 
The nonlinearities need to be monotonously increasing and the kernels either decreasing or continuous and attaining a global maximum.
"""

import numpy as np
import random as rand
from scipy.optimize import fmin

class Hawkes():

    def __init__(self, temporal, param, phi = lambda x: x + 2, mon_kernel = True):
        self.temporal = temporal
        self.Events = np.array([0])
        self.Sim_num = 0
        self.Nonlin = phi
        self.mon_kernel = mon_kernel
        self.param = param
        
        if mon_kernel is not True:
            self.ext = fmin(lambda x: -self.temporal(x, self.param), 0, disp = False);
        
    #%% ---------------
    
    
    def __bound(self, T):
        if T - self.Events[-1] < self.ext:
            # solving prediction problem h_n(t+u) < M(t)
            return self.Nonlin(self.temporal(T - self.Events, self.param).sum())+ self.temporal(self.ext, self.param)
        else:
            return self.Nonlin(self.temporal(T - self.Events, self.param).sum())
    
    def propogate_by_amount(self, k):    
        T = self.Events[-1] 
        
        for i in range(k):
            if self.mon_kernel is True:
                upper_bd = self.Nonlin(self.temporal(T - self.Events, self.param).sum())
            else:
                upper_bd = self.__bound(T)
            
            u = np.random.rand(1)
            tau = -np.log(u)/upper_bd
            T = T + tau
            s =  np.random.rand(1)

            if s <=  self.Nonlin(self.temporal(T- self.Events, self.param).sum())/upper_bd:
                self.Events = np.append(self.Events, T)
            
        if self.Sim_num == 0:    
            self.Events = np.delete(self.Events, 0,0)
            
        self.Sim_num += k 
        
    def density(self, t):
        return len(self.Events[self.Events <= t])/t
        
    def current_intensity(self, x):
        y = np.sort(np.append(x,self.Events))
        return y, self.Nonlin(np.array([np.sum(np.array([self.temporal(k-j, self.param) for j in self.Events])) for k in y]))
        


