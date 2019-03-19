#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:33:48 2018

@author: jens

Simulate a spatio-temporal Hawkes process on [0,2pi]x[0,infty) with periodic bdry conditions in space. For a detailed description see the jupyter notebook "Simulation of spatio-temporal Hawkes processes"
"""

import numpy as np
import random as rand
#import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.optimize as opt
from RandomNumberGen import draw_random_number_from_pdf

#
#rand.seed(42)

class Hawkes():

    def __init__(self, Base, spatial, temporal):
        self.Base = Base
        self.spatial = spatial
        self.temporal = temporal
        self.Events = np.array([[0],[0]])
        self.PoissEvent = np.array([])
        self.Sim_num = 0
        
        
    def propogate_by_amount(self, k, Space = [0, 2*np.pi]):        
        
        def periodized_spatial_kernel(x):
            return self.spatial(x%(2*np.pi))
        
        def flow_periodized_spatial_kernel(x):
            return periodized_spatial_kernel(np.sqrt(abs(x)))
        
        for i in range(k):
            self.PoissEvent = np.append(self.PoissEvent,rand.expovariate(1))
        
        PoissProcess = np.cumsum(self.PoissEvent)
        mu = quad(self.Base, Space[0], Space[1])[0]   
        print('new1')
        
        for time in PoissProcess[self.Sim_num :]:   
            # contained periodizer(x-j) but seems to be erroneous because why should the time be periodized? 
            # Events are drawn from self.Events[0,:], which corresponds to time
            # Rather need something that insures only x-j >= 0 contributes -> max(x-j,0)
            dist_temporal = lambda x: np.array([self.temporal(i) if i>= 0 else 0 for i in [max(x-j,0) for j in self.Events[0,:]]])
            # Integrate the periodized spatial kernel to employ inverse compensator method 
            integrated_spatial = np.array([quad(periodized_spatial_kernel, Space[0]-i, Space[1]-i)[0] for i in self.Events[1,:]])
            
            def righthandside(t):
                dydt = mu + np.multiply(integrated_spatial.T, dist_temporal(t)).sum()
                return dydt
            
            I = lambda x: quad(righthandside, 0, x)[0]  - time 
            EventTime = opt.fsolve(I, time/3)
            
            Spatial_density = lambda x: (self.Base(x) + dist_temporal(EventTime).sum()*np.array([periodized_spatial_kernel(x - i) for i in self.Events[1,:]]).sum())/(mu + dist_temporal(EventTime).sum()*integrated_spatial.sum())
            
            EventSpace = draw_random_number_from_pdf(Spatial_density, [0,2*np.pi])
            
            NewEvent = np.array([EventTime, EventSpace])
            self.Events = np.append(self.Events, NewEvent, axis = 1)
        
        self.Sim_num += k
            
