import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sounddevice as sd


colours = ['#001219',
           '#005F73',
           '#0A9396',
           '#94D2BD',
           '#E9D8A6',
           '#EE9B00',
           '#CA6702',
           '#BB3E03',
           '#AE2012',
           '#9B2226']



dat_22 = pd.read_csv('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week2_torque/All Sweeps/Cambridge_2022_Sep.026.txt',
                  delimiter='\t', skiprows=9)


def load_temps(dat, filenumber, upper_threshold=41,lower_threshold=1, up=True):

    T = np.array(dat['Ruthox_Temp_'+str(filenumber)])
    B = np.array(dat['Field_'+str(filenumber)])


    locs1 = B > lower_threshold
    locs2 = B < upper_threshold

    locs = locs1 & locs2

    T = T[locs]
    B = B[locs]

    if not up:
        T = np.flip(T)
        B = np.flip(B)
    return T, B

T_22, B_22 = load_temps(dat_22,'026')

dat_23 = pd.read_csv('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week2_torque/All Sweeps/Cambridge_2022_Sep.027.txt',
                  delimiter='\t', skiprows=9)


T_23, B_23 = load_temps(dat_23,'027',up=False)



B_big_22 = np.linspace(0,np.max(B_22), 20000)
interpd_T_22 = np.interp(B_big_22, B_22, T_22)

B_big_23 = np.linspace(0,np.max(B_22), 20000)
interpd_T_23 = np.interp(B_big_23, B_23, T_23)

diff = np.abs(interpd_T_22 - interpd_T_23)

sum_T = interpd_T_22 + interpd_T_23

fig, a = MakePlot(gs=True, figsize=(16,12)).create()

gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[:, 2])

ax1.plot(B_22, T_22, label='T-Up', c='indianred')
ax1.plot(B_23, T_23, label='T-Down', c='midnightblue')
ax2.plot(B_big_22, diff, label='Difference',c='darkslategray')
ax3.plot(B_big_22, sum_T, label='Sum',c='darkslategray')

# ax.legend()
publication_plot(ax1, 'Magnetic Field', r'$T$')
publication_plot(ax2, 'Magnetic Field', r'$\Delta T$')
publication_plot(ax3, 'Magnetic Field', r'$\Sigma T$')

handles, labels = ax1.get_legend_handles_labels()

legend = ax1.legend(handles, labels, framealpha=0,  loc='best',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0, labelspacing=2.6)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

handles, labels = ax2.get_legend_handles_labels()

legend = ax2.legend(handles, labels, framealpha=0,  loc='best',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0, labelspacing=2.6)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


handles, labels = ax3.get_legend_handles_labels()

legend = ax3.legend(handles, labels, framealpha=0,  loc='best',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0, labelspacing=2.6)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=1)
plt.show()
