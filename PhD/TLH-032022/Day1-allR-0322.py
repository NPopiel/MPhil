import numpy as np
import matplotlib.pyplot as plt
from tools.utils import *
from tools.ColorMaps import *
from tools.MakePlot import *

files = ['/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/Cambridge_March.006.txt',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/Cambridge_March.011.txt',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/Cambridge_March.013.txt',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/Cambridge_March.022.txt',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/Cambridge_March.024.txt']

delimiter = '\t'

field_column_number = 0
voltage_column_number_R = 6 # VT16 5 is signal in Y, 4 is singlan in X, 6 is R
#voltage_column_number = 8 # VT69
line_to_open_from = 8
voltage_column_number_X = 4
voltage_column_number_Y = 5

fig, axs = MakePlot(figsize=(16,6), ncols=3).create()
cmap = select_discrete_cmap('jugglypuff')

angles = [-7.5, -5,-2.5,0,2.5]
prev_angle = -7.5
for i, file in enumerate(files):


    field = np.genfromtxt(file,delimiter=delimiter,skip_header=line_to_open_from)[:,field_column_number]
    volts_R = np.genfromtxt(file,delimiter=delimiter,skip_header=line_to_open_from)[:,voltage_column_number_R]
    volts_X = np.genfromtxt(file,delimiter=delimiter,skip_header=line_to_open_from)[:,voltage_column_number_X]
    volts_Y = np.genfromtxt(file,delimiter=delimiter,skip_header=line_to_open_from)[:,voltage_column_number_Y]


    axs[0].plot(field, 1e3*volts_X, linewidth=2,c=cmap[i], label=str(angles[i]))
    axs[1].plot(field, 1e3*volts_Y, linewidth=2, c=cmap[i], label=str(angles[i]))
    axs[2].plot(field, 1e3*volts_R, linewidth=2, c=cmap[i], label=str(angles[i]))

publication_plot(axs[0], 'Magnetic Field (T)', 'Capacitance from X (mV)')
publication_plot(axs[1], 'Magnetic Field (T)', 'Capacitance from Y (mV)')
publication_plot(axs[2], 'Magnetic Field (T)', 'Capacitance from R (mV)')
legend = axs[2].legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=2)
plt.show()