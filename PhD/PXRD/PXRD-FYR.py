#Importing necessary modules
from tools.utils import *
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



#Setting instrument wavelength for d spacing conversion
wavelength = 1.5406 #Angstroms

main_path = '/Volumes/GoogleDrive/My Drive/FeSi/PXRD/Camilla/'
#exporting data from XRD file
d_spacing,Yobs,Ycalc,Ydiff,two_theta = LoadRietveldData(main_path+"FeSi-final-riet.txt",wavelength)

###################################   Outputting ticks

#ticks 1
offset_1 = -200
tick_1 = np.loadtxt(main_path+"FeSi-1-4_486.txt",usecols=0)
tick_1_d = wavelength / (2 * np.sin(np.radians(tick_1 /2)))
y_tick_1 = np.ones(len(tick_1_d))*offset_1

#ticks 2
offset_2 = -275
tick_2 = np.loadtxt(main_path+"FeSi-2-4_487.txt",usecols=0)
tick_2_d = wavelength / (2 * np.sin(np.radians(tick_2 /2)))
y_tick_2 = np.ones(len(tick_2_d))*offset_2

#determine max & min limits for plot
x_min = np.min(d_spacing)
x_max = np.max(d_spacing)

#Plot vs d
fig, (ax1, ax2) = MakePlot(figsize=(16,9), ncols=2).create()

ax= ax1

ax.plot(d_spacing, Yobs, 'k.', markersize=3, label='Observed')
ax.plot(d_spacing, Ycalc, 'r-', linewidth = 1.5, label='Calculated')
ax.plot(d_spacing, Ydiff, C = "#838B8B",linewidth = 0.5,  label='Difference')
ax.plot(tick_1_d,y_tick_1 , marker='|',color='b',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 1")
ax.plot(tick_2_d,y_tick_2 , marker='|',color='g',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 2")


plt.xlim([x_min,x_max])

publication_plot(ax, 'd-spacing (Å)', 'Intensity Counts (arb.)')



#plt.yticks([])

#plt.savefig('Riet_d.png', dp=300)

#Deterining plot limits
x_min = np.min(two_theta)
x_max = np.max(two_theta)

#Plot vs 2 theta
# fig, ax = MakePlot(figsize=(8,6)).create()

ax = ax2

ax.plot(two_theta, Yobs, 'k.', markersize=3, label='Observed')
ax.plot(two_theta, Ycalc, 'r-', linewidth = 1.5, label='Calculated')
ax.plot(two_theta, Ydiff, C = "#838B8B", linewidth = 0.5, label='Difference')
ax.plot(tick_1,y_tick_1 , marker='|',color='b',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 1")
ax.plot(tick_2,y_tick_2 , marker='|',color='g',markersize=13,linestyle= 'None'  ,label="FeSi - Phase 2")

plt.xlim([x_min,x_max])

publication_plot(ax, '2θ (°)', '')
legend = ax.legend(framealpha=0, ncol=1, loc='upper right',
              prop={'size': 24, 'family': 'arial'},  columnspacing = .5)

plt.tight_layout(pad=1)
# plt.show()
#plt.yticks([])

plt.savefig(main_path+'Riet_theta.png', dpi= 300)
