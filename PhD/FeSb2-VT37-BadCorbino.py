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
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap



'''
All the paths

/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_0T.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_5deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_10deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_15deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_20deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_25deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_30deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_35deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_40deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_45deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_50deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_55deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_60deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_65deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_70deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_75deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_80deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_85deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_90deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_180deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_270deg.dat
/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T.dat
'''

main_path = '/Volumes/GoogleDrive/My Drive/Corbino/Bad/'

zero_T_IV_name = '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_0T.dat'

files = [
'/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T.dat',
'/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_5deg.dat',
'/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_10deg.dat']
#'/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_15deg.dat']
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_20deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_25deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_30deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_35deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_40deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_45deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_50deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_55deg.dat']
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_60deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_65deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_70deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_75deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_80deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_85deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_90deg.dat',
# '/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_180deg.dat']
# #'/Volumes/GoogleDrive/My Drive/Corbino/Bad/VT37_2pt_4pt_IV_1p8K_14T_270deg.dat']

relevant_columns = ['Temperature (K)',
                    'Magnetic Field (Oe)',
                    'Bridge 1 Resistance (Ohms)',
                    'Bridge 1 Excitation (uA)'
                    'Bridge 2 Resistance (Ohms)',
                    'Bridge 2 Excitation (uA)']

angles = [0, 5, 10]#, 15]#, 20, 25, 30, 35, 40, 45, 50, 55]#, 60, 65, 70, 75, 80, 85, 90, 180]#, 270]

R_2pt, R_4pt = {}, {}
I_2pt, I_4pt = {}, {}

for ind, file in enumerate(files):

    dat =  load_matrix(file)

    R_2pt[angles[ind]] = np.array(dat['Bridge 1 Resistance (Ohms)'])
    I_2pt[angles[ind]] = np.array(dat['Bridge 1 Excitation (uA)'])*1e-6

    R_4pt[angles[ind]] = np.abs(dat['Bridge 2 Resistance (Ohms)'])
    I_4pt[angles[ind]] = np.array(dat['Bridge 1 Excitation (uA)'])*1e-6


fig, axs = MakePlot(ncols=2).create()

cmap = select_discrete_cmap()

cs = [cmap[0], cmap[3], cmap[9]]

for i, angle in enumerate(angles):

    locs_2pt_a = I_2pt[angle] < 0.004
    locs_2pt_b = I_2pt[angle] > 0.000001
    locs_2pt = locs_2pt_a & locs_2pt_b
    locs_4pt_a = I_4pt[angle] < 0.004
    locs_4pt_b = I_4pt[angle] > 0.00001
    locs_4pt = locs_4pt_a & locs_4pt_b

    axs[0].plot(I_2pt[angle][locs_2pt]*1e3, 1e3*R_2pt[angle][locs_2pt]*I_2pt[angle][locs_2pt], linewidth=3, c=cs[i], label=str(angle)+'$\degree$')
    axs[1].plot(I_4pt[angle][locs_4pt]*1e3, 1e5*R_4pt[angle][locs_4pt]*I_4pt[angle][locs_4pt], linewidth=3, c=cs[i], label=str(angle)+'$\degree$')

publication_plot(axs[0], 'Current (mA)', 'Local Voltage (mV)')
publication_plot(axs[1], 'Current (mA)', r'Nonlocal Voltage ($\times 10^2$ mV)')

legend = axs[1].legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())




plt.tight_layout(pad=.5)
plt.savefig('/Volumes/GoogleDrive/My Drive/FirstYearReport/Figures/painted_corbino.png')
plt.show()
