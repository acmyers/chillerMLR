# -*- coding: utf-8 -*-
"""
@author: Andrew
"""

from __future__ import division
from numpy import *
import numpy as np
import scipy
from scipy.optimize import fmin_cg
from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas
import random
from dataCleaner import dataCleaner
from helperFunctions import train, predict


'''
The following code uses Multiple Linear Regression that learns from actual operations data 
to model and predict centrifugal chiller performance within a range of 0.013 +/- 0.017 
(mean absolute error +/ 1 standard deviation) or 5% error for a kW/Ton of 0.6.  Machine
learning models like this one can be used for optimizing chiller and system energy efficiency.
Those who are interested can experiment and improve on this model by adding more useful 
features, cleaning the data better, and trying new machine learning learning algorithms.

Data set:
 - Date_Time = date and time in excel serial format
 - KWperTon = measured kW per Ton of cooling
 - Teo = temperature of water at evaporator outlet (degrees F)
 - Tei = temperature of water at evaporator inlet (degress F)
 - Fevap = flow rate of water through evaporator (gpm)
 - Tci = temperature of water at condenser inlet (degrees F)
 - Tco = temperature of water at condenser outlet (degress F)
 - Fcond = flow rate of water through condenser (gpm)
 - Pei = pressure reading at evaporator inlet (psi)
 - Peo = pressure reading at evaporator outlet (psi)
 - Pco = pressure reading at condenser outlet (psi)
 - Pci = pressure reading at condenser inlet (psi)
 - A_kW = power consumption for compressor A (kW)
 - B_kW = power consumption for compressor B (kW)

For more information, check out my other repositories (chillerDataVisual, system_optimization, 
and vsdEfficiency) and the Google Whitepaper 'Machine Learning Applications for Data Center 
Optimization' by Jim Gao.

Note: look out for "Warning: Desired error not necessarily achieved due to precision loss."
This may be due to random sampling to obtain the training data set.  All you need to do is
re-run the analysis and the warning should go away.  The proper output should say
"Optimization terminated successfully."
'''
  
def main():
    ## ======== Settings ======================
    # Set rated tons (i.e. capacity) of chiller being analyzed
    ratedTons = 3700
    
    
    ## ========  Read, clean, and format data sets for machine learning =======  
    # Import the data
    filename = "chiller_data.csv"
    chillerData = pandas.read_csv(filename,low_memory=False)
    
    # Clean the data
    print 'Length before cleaning data:'
    print len(chillerData.index)
    chillerData = dataCleaner(chillerData, ratedTons)
    print 'Length after cleaning data:'
    print len(chillerData.index)
    
    # Create X and y data sets (uses 'Status' for constant)
    # try: adding and removing variables like 'Pci' and 'Fcond' to see effect on model performance
    # e.g. X = chillerData.loc[:,['Status','PER^2','PER','Tci','Fcond']]
    # note: adding another variable will invalidate the 3D data visualization below
    X = chillerData.loc[:,['Status','PER^2','PER','Tci']]
    y = chillerData.loc[:,['KWperTon']]
    resample = int(0.2*len(y))
    rows = random.sample(X.index, resample)
    X_train = X.drop(rows)
    y_train = y.drop(rows)
    
    # Keep X_train and y_train dataframes for visuals
    X_plot = X_train
    y_plot = y_train 

    # Create validation and test sets
    X = X.ix[rows]
    y = y.ix[rows]
    X_val = X.iloc[::2]
    y_val = y.iloc[::2]
    X_test = X.iloc[1::2]
    y_test = y.iloc[1::2]
    
    # Convert data to numpy format
    X_train = X_train.as_matrix()
    y_train = y_train.values
    X_val = X_val.as_matrix()
    y_val = y_val.values
    X_test = X_test.as_matrix()
    y_test = y_test.values
    
    # Option for regularization (not used when lamda = 0.0)
    lamda = 0.0
        
    # Train the model
    theta = train(X_train, y_train, lamda)
    
    
    ## ==========  Plot the data and model in 3D  ==============
    # Initialize a figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.hold(True)

    # Create a surface plot from the trained model
    x_surf=np.arange(0.25, 0.95, 0.05)            
    y_surf=np.arange(45.0, 90.0, 2.5)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = theta[0] + theta[1]*(x_surf**2) + theta[2]*x_surf + theta[3]*y_surf
    ax.plot_surface(x_surf, y_surf, z_surf, cmap=plt.cm.jet, cstride=1, rstride=1); 

    # Add the actual data as a scatter plot
    ax.scatter(X_plot['PER'], X_plot['Tci'], zs=y_plot['KWperTon'], s=10,  c='#A0A0A0')
    plt.title('Predicted vs. Actual Chiller Efficiency (rotate for better view')
    ax.set_xlabel('PER')
    ax.set_ylabel('T_ci')
    ax.set_zlabel('kW/Ton')

    plt.show()
    
    
    ## ============= Calculate training, validation, and test error ===========
    # Calculate Mean Absolute Error and R^2 for trainig, validation, and test data
    MAE_train, r2_train, stdev_train = predict(X_train, y_train, theta)
    MAE_val, r2_val, stdev_val = predict(X_val, y_val, theta)
    MAE_test, r2_test, stdev_test = predict(X_test, y_test, theta)
    
    # Print results
    print 'Theta:'
    print theta
    
    print 'R^2 (training data):'
    print r2_train
    
    print 'R^2 (validation data):'
    print r2_val
    
    print 'R^2 (test data):'
    print r2_test
    
    print 'Mean absolute error (training data):'
    print MAE_train
    
    print 'Mean absolute error (validation data):'
    print MAE_val
    
    print 'Mean absolute error (test data):'
    print MAE_test
    
    print 'Standard deviation of error (training data):'
    print stdev_train
    
    print 'Standard deviation of error (validation data):'
    print stdev_val
    
    print 'Standard deviation of error (test data):'
    print stdev_test

if __name__ == '__main__':
	main()



