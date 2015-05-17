# chillerMLR
The code in this repository uses Multiple Linear Regression that learns from actual operations 
data to model and predict centrifugal chiller performance within a range of 0.013 +/- 0.017 
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
