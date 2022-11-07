import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import pandas as pd
import numpy.linalg
from tools.DataFile import DataFile
from tools.MakePlot import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tools.ColorMaps import select_discrete_cmap
from tools.utils import *

def sort_func(filepath):
  name = filepath.split('/')[-1]
  sweep_num = name.split('p')[1].split('.')[0]
  return float(sweep_num)

'''
/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/FeSi/FeSi_Feb2022_torque006.txt
'''

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/FeSi/'

angle_folders = ['Angle 1/',
                 'Angle 2/',
                 'Angle 3/',
                 'Angle 4/',
                 'Angle 5/',
                 'Angle 6/',
                 'Angle 7/',
                 'Angle 8/',
                 'Angle 9/',
                 'Angle 10/',
                 'Angle 11/'
                 ]

data_dict = {}

for angle, fold in enumerate(angle_folders):
    dat_holder = []
    i=1

    files = glob(main_path+fold+'*.csv')

    files.sort(key=sort_func)

    if len(files) > 8:
        files = np.array(files)[:7]

    for file in files:
        dat_holder.append(np.squeeze(load_matrix(file, delimeter=',')))

    data_dict[angle+1] = np.vstack(dat_holder)



fig, ax = MakePlot().create()
for ang, dat in data_dict.items():

    ax.plot(dat[:,0], dat[:,1], linewidth=2, c=select_discrete_cmap(map_name='bulbasaur')[ang-1], label='Angle '+str(ang))

ax.legend()

publication_plot(ax, 'Magnetic Field (T)', 'Capacitance (arb.)')
plt.show()


# Load in all of Angle 8 and plot as a function of time to examine noise


dat_holder = []

files = glob(main_path+'Angle 8/'+'*.csv')

files.sort(key=sort_func)

for file in files:
    dat_holder.append(np.squeeze(load_matrix(file, delimeter=',')))

all_angle8 = np.vstack(dat_holder)

fig, ax = MakePlot().create()

ax.plot(all_angle8[:,1], linewidth=2, c=select_discrete_cmap(map_name='bulbasaur')[7], label='Angle 8')

publication_plot(ax, 'Time', 'Capacitance (arb.)', title='Angle 8 Noise')
plt.show()

files = glob(main_path+'Angle 1/'+'*.csv')

files.sort(key=sort_func)

for file in files:
    dat_holder.append(np.squeeze(load_matrix(file, delimeter=',')))

all_angle1 = np.vstack(dat_holder)

fig, ax = MakePlot().create()

ax.plot(all_angle1[:,1], linewidth=2, c=select_discrete_cmap(map_name='bulbasaur')[8], label='Angle 1')

publication_plot(ax, 'Time', 'Capacitance (arb.)', title='Angle 1 Noise')
plt.show()




