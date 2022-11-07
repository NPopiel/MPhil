import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import pandas as pd
import numpy.linalg
from tools.DataFile import DataFile
from tools.MakePlot import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tools.ColorMaps import *
from tools.utils import *


main_path = '/Volumes/GoogleDrive/My Drive/Figures/TLH_08_2022/resistance_scaling/'

VT69_filename = '2020-01-29-FeSb2_v67_v68_v69_00001.dat'
VT16_filename = 'FeSb2_S19Sep_VT16_VT17_VT18.dat'
VT54_filename = '2020-01-29-FeSb2_v52_v53_v54.dat'

relevant_columns = ['Temperature (K)',
                    'Magnetic Field (Oe)',
                    'Bridge 1 Resistance (Ohms)',
                    'Bridge 1 Excitation (uA)'
                    'Bridge 2 Resistance (Ohms)',
                    'Bridge 2 Excitation (uA)']


fig, a = MakePlot(gs=True,figsize=(7,9)).create()

gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[:,:])

dat_VT16 =load_matrix(main_path+VT16_filename)

R_VT16 = np.squeeze(np.array(dat_VT16['Bridge 1 Resistance (Ohms)']))
T_VT16 = np.squeeze(np.array(dat_VT16['Temperature (K)']))

dat_VT69 =load_matrix(main_path+VT69_filename)

R_VT69 = np.squeeze(np.array(dat_VT69['Bridge 3 Resistance (Ohms)']))
T_VT69 = np.squeeze(np.array(dat_VT69['Temperature (K)']))

dat_VT54 =load_matrix(main_path+VT54_filename)

R_VT54 = np.squeeze(np.array(dat_VT54['Bridge 3 Resistance (Ohms)']))
T_VT54 = np.squeeze(np.array(dat_VT54['Temperature (K)']))


cmap = select_discrete_cmap('bulbasaur')

ax1.scatter(T_VT16, 1/R_VT16, label='VT16', s=40, c=cmap[0])
ax1.scatter(T_VT69, 1/R_VT69, label='VT69', s=40, c=cmap[7])
ax1.scatter(T_VT54, 1/R_VT54, label='VT54', s=40, c=cmap[4])


legend = ax1.legend(framealpha=0, ncol=1, loc='upper left',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(width=0.8, headlength=10, headwidth=8,color='k')



# ax1.set_ylim(1e-3, 3e7)
# ax1.set_xlim(1.5,330)

plt.tight_layout(pad=1)
publication_plot(ax1, 'Temperature (K)', r'Conductance ($\Omega^{-1}$)')



plt.tight_layout(pad=.5)
# plt.show()
plt.savefig(main_path+'VT16+VT69+VT54-GvT.png', dpi=300, bbox_inches='tight')