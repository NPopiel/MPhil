#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:30:10 2020

@author: alexanderhickey
"""

import numpy as np
from tools.utils import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Paper_FeSb2_2020Jan/Supplementary/MPMS/'

temps0c, sus0c = np.genfromtxt(main_path+'VT54c0zmpms.csv', delimiter=',').T
temps7c, sus7c = np.genfromtxt(main_path+'VT54c7zmpms.csv', delimiter=',').T
temps0ab, sus0ab = np.genfromtxt(main_path+'VT54ab0zmpms.csv', delimiter=',').T
temps7ab, sus7ab = np.genfromtxt(main_path+'VT54ab7zmpms.csv', delimiter=',').T
 
fig, ax = MakePlot(figsize=(10,7)).create()
ax.hlines(0, 310, [0], '0.5')

ax.plot(temps0c, sus0c, color='blue',
        label='$\mathbf{H}$ $\perp$ (001) 0.05 T')
ax.plot(temps7c, sus7c, color='blue', ls='--',
        label=r'$\mathbf{H}$ $\perp$ (001)  7.0 T')
ax.plot(0, 0, color='red',
        label=r'$\mathbf{H}$ $\perp$ (110) 0.05 T')
ax.plot(0, 0, color='red', ls='--',
        label=r'$\mathbf{H}$ $\perp$ (110)  7.0 T')


ax.set_ylim(-0.8, 6.8)
ax.set_xlim(0, 305)

publication_plot(ax, 'Temperature (K)', 'Magnetic Susceptibility\n(emu T$^{-1}$ mol$^{-1}$)')


ax.legend(loc='lower right', framealpha=0, fontsize=16,
          bbox_to_anchor=(1, 0.09))


fieldc, magc, mag_errc, fitc = np.genfromtxt(main_path+'VT54cFSmpms.csv',
                                             delimiter=',').T
fieldab, magab, mag_errab, fitab = np.genfromtxt(main_path+'VT54abFSmpms.csv',
                                                 delimiter=',').T


axin = fig.add_axes([0.23, 0.44, 0.27, 0.44])

axin.hlines(0, 310, [0], '0.5')

axin.plot(temps0c, sus0c, color='blue')
axin.plot(temps7c, sus7c, color='blue', ls='--')
axin.plot(temps0ab, sus0ab, color='red')
axin.plot(temps7ab, sus7ab, color='red', ls='--')

axin.set_ylim(-0.8, 6.8)
axin.set_xlim(60, 310)
publication_plot(axin, 'Temperature (K)', 'Magnetic Susceptibility\n(emu T$^{-1}$ mol$^{-1}$)',label_fontsize=18,tick_fontsize=15)

# plt.tight_layout(pad=1)
# plt.show()

fig.savefig(main_path+'MPMSfig2.pdf', bbox_inches='tight')