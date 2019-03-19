#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:00:48 2018

@author: jens
"""

import numpy as np
#import matplotlib.pyplot as plt


def draw_random_number_from_pdf(pdf, interval, pdfmax = 1, integers = False, max_iterations = 10000):
            for i in range(max_iterations):
                if integers == True:
                    rand_x = np.random.randint(interval[0], interval[1])
                else:
                    rand_x = (interval[1] - interval[0]) * np.random.random(1) + interval[0] #(b - a) * random_sample() + a

                    rand_y = pdfmax * np.random.random(1) 
                    calc_y = pdf(rand_x)

                    if(rand_y <= calc_y ):
                        return rand_x

            raise Exception("Could not find a matching random number within pdf in " + max_iterations + " iterations.")
