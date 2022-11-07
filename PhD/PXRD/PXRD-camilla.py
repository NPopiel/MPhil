#Importing necessary modules
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


################################################### Plot Functions - From Liam - used as Inspo

# Imports the Rietveld data
def LoadRietveldData(filename, lambda_):
    """Loads Rietveld output txt - returns the arrays with the data points as well as x-axis in d spacings.
    In: filename, wavelength of instrument X-Ray"""
    two_theta = np.loadtxt(filename, usecols=0, unpack=True)
    Yobs = np.loadtxt(filename, usecols=1, unpack=True)
    Ycalc = np.loadtxt(filename, usecols=2, unpack=True)
    Ydiff = np.loadtxt(filename, usecols=3, unpack=True)

    d_spacing = lambda_ / (2 * np.sin(np.radians(two_theta / 2)))
    return d_spacing, Yobs, Ycalc, Ydiff, two_theta


############################################# Figure settings
SMALLS_SIZE = 12
SMALL_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLS_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLS_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLS_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#Setting instrument wavelength for d spacing conversion
wavelength = 1.5406 #Angstroms

#exporting data from XRD file
d_spacing,Yobs,Ycalc,Ydiff,two_theta = LoadRietveldData("FeSi-final-riet.txt",wavelength)

###################################   Outputting ticks

#ticks 1
offset_1 = -200
tick_1 = np.loadtxt("FeSi-1-4_486.txt",usecols=0)
tick_1_d = wavelength / (2 * np.sin(np.radians(tick_1 /2)))
y_tick_1 = np.ones(len(tick_1_d))*offset_1

#ticks 2
offset_2 = -275
tick_2 = np.loadtxt("FeSi-2-4_487.txt",usecols=0)
tick_2_d = wavelength / (2 * np.sin(np.radians(tick_2 /2)))
y_tick_2 = np.ones(len(tick_2_d))*offset_2

#determine max & min limits for plot
x_min = np.min(d_spacing)
x_max = np.max(d_spacing)

#Plot vs d
plt.figure(figsize=(16,8)) #,facecolor='w'
plt.plot(d_spacing, Yobs, 'k.', markersize=3, label='Observed')
plt.plot(d_spacing, Ycalc, 'r-', linewidth = 1.5, label='Calculated')
plt.plot(d_spacing, Ydiff, C = "#838B8B",linewidth = 0.5,  label='Difference')
plt.plot(tick_1_d,y_tick_1 , marker='|',color='b',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 1")
plt.plot(tick_2_d,y_tick_2 , marker='|',color='g',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 2")
plt.legend(loc='best')
plt.title('02C03 Rietveld Plot')
plt.xlabel('d-spacing (Å)')
plt.xlim([x_min,x_max])
plt.ylabel('Intensity Counts (a.u.)')
plt.show()
#plt.yticks([])

#plt.savefig('Riet_d.png', dp=300)

#Deterining plot limits
x_min = np.min(two_theta)
x_max = np.max(two_theta)

#Plot vs 2 theta
plt.figure(figsize=(16,8)) #,facecolor='w'
plt.plot(two_theta, Yobs, 'k.', markersize=3, label='Observed')
plt.plot(two_theta, Ycalc, 'r-', linewidth = 1.5, label='Calculated')
plt.plot(two_theta, Ydiff, C = "#838B8B", linewidth = 0.5, label='Difference')
plt.plot(tick_1,y_tick_1 , marker='|',color='b',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 1")
plt.plot(tick_2,y_tick_2 , marker='|',color='g',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 2")
plt.legend(loc='best')
plt.title('02C03 Rietveld Plot')
plt.xlabel(' 2θ (°)')
plt.xlim([x_min,x_max])
plt.ylabel('Intensity Counts (a.u.)')
#plt.yticks([])

#plt.savefig('Riet_theta.png', dpi= 300)
