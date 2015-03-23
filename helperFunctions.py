# -*- coding: utf-8 -*-
"""
@author: Andrew
"""
from __future__ import division
from numpy import *
import numpy as np
import scipy
import pandas
from scipy.optimize import fmin_cg
import random
    
'''
The following functions are used by chillerMLR.py
'''    
def featureNormalize( data ):
    mu = mean( data, axis=0 )
    data_norm = data - mu
    sigma = std( data_norm, axis=0, ddof=1 )
    data_norm = data_norm / sigma
    return data_norm, mu, sigma
 
def computeCost( theta, X, y, lamda ):
    theta = theta.reshape( shape(X)[1], 1 )
    m 	= shape( X )[0]
    term1 = X.dot( theta ) - y 
    left_term = term1.T.dot( term1 ) / (2 * m)
    right_term = theta[1:].T.dot( theta[1:] ) * (lamda / (2*m))
    J = (left_term + right_term).flatten()[0]            
    return J

def computeGradient( theta, X, y, lamda ):
    theta = theta.reshape( shape(X)[1], 1 )
    m 	= shape( X )[0]
    grad = X.dot( theta ) - y 
    grad = X.T.dot( grad) / m
    grad[1:]	 = grad[1:] + theta[1:] * lamda / m
    return grad.flatten()

def train(X, y, lamda):
    theta = zeros( (shape(X)[1], 1) )
    result = scipy.optimize.fmin_cg( computeCost, fprime = computeGradient, x0 = theta, 
								args = (X, y, lamda), maxiter = 500, disp = True, full_output = True )
    return result[0]
    
def r_squared(predictions, actual):
    SSres = sum((actual.T - predictions)**2)
    SStot = sum((actual.T - np.mean(actual))**2)
    r_squared = 1 - (SSres/SStot)
    return r_squared
    
def predict(X, y, theta):
    m = len(y)
    actual = y
    predictions = X.dot( theta )
    errors = predictions - y.T
    MAE = abs(errors).sum() / m
    r2 = r_squared(predictions, actual)
    stdev = np.std(errors)
    return MAE, r2, stdev