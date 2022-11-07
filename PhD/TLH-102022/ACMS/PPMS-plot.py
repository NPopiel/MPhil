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


main_path = '/Volumes/GoogleDrive/My Drive/Data/FeSb2/ACMS/VT154/'

filename_500 = 'MvT-VT154-001-500Oe.dat'
filename_9 = 'MvT-VT154-001-ZFC-9T.dat'


relevant_columns = ['Temperature (K)',
                    'Magnetic Field (Oe)','DC Moment (emu)']


fig, a = MakePlot(gs=True,figsize=(14,9)).create()

gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])

dat_500 =load_matrix(main_path+filename_500)

M_500 = np.squeeze(np.array(dat_500['DC Moment (emu)']))
T_500= np.squeeze(np.array(dat_500['Temperature (K)']))

dat_9 = load_matrix(main_path+filename_9)

M_9 = np.squeeze(np.array(dat_9['DC Moment (emu)']))
T_9= np.squeeze(np.array(dat_9['Temperature (K)']))



cmap = select_discrete_cmap('bulbasaur')

ax1.scatter(T_500, M_500*1e4, label='500 Oe', s=40, c=cmap[0])
ax1.scatter(T_9, M_9*1e4, label='9 T', s=40, c=cmap[7])

ax2.scatter(T_500, 1e4*M_500/0.05, label='500 Oe', s=40, c=cmap[0])
ax2.scatter(T_9, 1e4*M_9/9, label='9 T', s=40, c=cmap[7])

publication_plot(ax1, r'$T$ (K)', r'$M$ ($\times 10^{-4}$ emu)')
publication_plot(ax2, r'$T$ (K)', r'$\chi$ ($\times 10^{-4}$ emu T$^{-1}$)')

legend = ax1.legend(framealpha=0, ncol=1, loc='lower right',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())




plt.tight_layout(pad=1)
# plt.show()
plt.savefig('/Volumes/GoogleDrive/My Drive/Figures/TLH_10_2022/ACMS/VT154-PPMS.png', dpi=300, bbox_inches='tight')