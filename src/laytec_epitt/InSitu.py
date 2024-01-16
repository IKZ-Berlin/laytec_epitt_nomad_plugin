# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import sklearn
import pandas
import matplotlib.pyplot as plt
import numpy as np

csfont = {'fontname':'Times New Roman'}
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')


SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

Year_1 = 2020
Year_2 = 20
Sample = "158"

def load_Data_Ga(Year_1, Year_2, Sample):
    
    Spectrum_Data_Ga = pandas.read_excel(r"Z:\Gruppen\oxhl\Private\1Experiment_Database\Experimental_Data_%d\%d-%s\Laytec\%d_%s_Ga.xlsx" %(Year_1,Year_2, Sample, Year_2, Sample))

    return Spectrum_Data_Ga

def load_Data_Al(Year_1, Year_2, Sample):

    Spectrum_Data_Al = pandas.read_excel(r"Z:\Gruppen\oxhl\Private\1Experiment_Database\Experimental_Data_%d\%d-%s\Laytec\%d_%s_Al.xlsx" %(Year_1,Year_2, Sample, Year_2, Sample))
    
    return Spectrum_Data_Al

# In[11]:

#405nm


Start = 640
Period = 40000



#
def read_data_Ga(Year_1, Year_2, Sample, Start, Period):
    
    Data_Ga= load_Data_Ga(Year_1, Year_2, Sample)

 
    Raw_405nm_Ga = Data_Ga["DetWhite"][(Data_Ga["BEGIN"]>=Start) & (Data_Ga["BEGIN"]<Start+Period) ]

    
    Raw_633nm_Ga = Data_Ga["DetReflec"][(Data_Ga["BEGIN"]>=Start) & (Data_Ga["BEGIN"]<Start+Period) ]
   
    
    Raw_951nm_Ga = Data_Ga["RLo"][(Data_Ga["BEGIN"]>=Start) & (Data_Ga["BEGIN"]<Start+Period) ]


    Time_Domain_Ga =  Data_Ga["BEGIN"][(Data_Ga["BEGIN"]>=Start) & (Data_Ga["BEGIN"]<Start+Period)]
    
    
    Data_Read_Out_Ga= {"405nm":Raw_405nm_Ga, "633nm":Raw_633nm_Ga, "951nm":Raw_951nm_Ga, "Time_Domain":Time_Domain_Ga}
    
    return Data_Read_Out_Ga

Spectrum_Ga= read_data_Ga(Year_1, Year_2, Sample, Start, Period)



    
plt.figure(figsize=(8,6))
plt.plot(Spectrum_Ga["Time_Domain"], Spectrum_Ga["405nm"]/Spectrum_Ga["405nm"].values[0])
plt.title("Raw Reflectance Spectrum 405nm (Ga)")
plt.ylabel("Reflectance")
plt.xlabel("Time")
plt.tight_layout()


Rolling_405nm = (Spectrum_Ga["405nm"]/Spectrum_Ga["405nm"].values[0]).rolling(30).mean()


plt.plot(Spectrum_Ga["Time_Domain"],Rolling_405nm)

plt.show()
    
plt.figure(figsize=(8,6))
plt.plot(Spectrum_Ga["Time_Domain"], Spectrum_Ga["633nm"]/Spectrum_Ga["633nm"].values[0])
plt.title("Raw Reflectance Spectrum 633nm (Ga)")
plt.ylabel("Reflectance")
plt.xlabel("Time")
plt.tight_layout()
Rolling_633nm = (Spectrum_Ga["633nm"]/Spectrum_Ga["633nm"].values[0]).rolling(30).mean()


plt.plot(Spectrum_Ga["Time_Domain"],Rolling_633nm)

plt.show()
    



plt.figure(figsize=(8,6))
plt.plot(Spectrum_Ga["Time_Domain"], Spectrum_Ga["951nm"]/Spectrum_Ga["951nm"].values[0])
plt.title("Raw Reflectance Spectrum 951nm (Ga)")
plt.ylabel("Reflectance")
plt.xlabel("Time")
plt.tight_layout()


Rolling_951nm = (Spectrum_Ga["951nm"]/Spectrum_Ga["951nm"].values[0]).rolling(30).mean()


plt.plot(Spectrum_Ga["Time_Domain"],Rolling_951nm)

plt.show()







