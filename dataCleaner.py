# -*- coding: utf-8 -*-
"""
@author: Andrew
"""
from __future__ import division
from numpy import *
import numpy as np
import scipy
import pandas

'''
This code is for cleaning data from a centrifugal chiller.
See comments for details on why data is being removed.
'''
def dataCleaner(chillerData, ratedTons):
    
    # Create variable for "Status": if the flow rate of water through the evaporator is greater than 300 GPM, consider the chiller 'ON'
    ix = chillerData['Fevap'] > 300
    ix = ix.astype(bool)
    ix = ix.astype(int)
    chillerData['Status'] = ix

    # Create variable for 'Tons of Cooling' (1 Ton = 12000 BTU/hr)
    chillerData['Tons'] = 500*chillerData['Fevap']*(chillerData['Tei'] - chillerData['Teo'])/12000
    
    # Create variable for 'percent of full capacity' and account for 1 or 2 compressors running    
    ix1 = (chillerData['A_kW']  < (chillerData['B_kW'] - 200))
    ix1 = ix1.astype(bool)
    ix1 = ix1.astype(int)
    ix2 = (chillerData['B_kW']  < (chillerData['A_kW'] - 200))
    ix2 = ix2.astype(bool)
    ix2 = ix2.astype(int)
    ix = ix1 + ix2
    chillerData['oneCompressor'] = ix
    chillerData['PER'] = ones(len(chillerData))
    chillerData['PER'][chillerData['oneCompressor'] == 1] = chillerData['Tons']/(ratedTons/2)
    chillerData['PER'][chillerData['oneCompressor'] == 0] = chillerData['Tons']/(ratedTons)
        
    # Clean out all times when Status = 0 (i.e. chiller is off)
    chillerData = chillerData[chillerData.Status == 1]
    
    # Clean out kWperTon out of range
    chillerData = chillerData[chillerData.KWperTon > 0.15]
    chillerData = chillerData[chillerData.KWperTon < 0.9]
    
    # Clean out PER out of range
    chillerData = chillerData[chillerData.PER > 0.2]
    chillerData = chillerData[chillerData.PER < 0.95]
    
    # Clean out Fevap out of range
    chillerData = chillerData[chillerData.Fevap > 1000]
    chillerData = chillerData[chillerData.Fevap < 7000]
    
    # Clean out Teo out of range
    chillerData = chillerData[chillerData.Teo > 37]
    chillerData = chillerData[chillerData.Teo < 41]
    
    # Clean out Tei out of range
    chillerData = chillerData[chillerData.Tei > 45]
    chillerData = chillerData[chillerData.Tei < 60]

    # Clean out Fcond out of range
    chillerData = chillerData[chillerData.Fcond > 3000] 
    chillerData = chillerData[chillerData.Fcond < 11000] 

    #TODO: get rid of rows with consecutive repeat data
    
    # Create PER^2 variable
    chillerData['PER^2'] = chillerData['PER']*chillerData['PER']
    
    return chillerData