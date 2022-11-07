#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:22:05 2020

@author: alexanderhickey
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from tools.utils import *

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Paper_FeSb2_2020Jan/Supplementary/Cp/HeatCapacity/'

av_data_df = pd.read_csv(main_path+'My_Average.csv')

ts = {}
t_err = {}
cps = {}
cp_err = {}

for f in range(0, 11, 2):
    (ts[f], cps[f], t_err[f], cp_err[f]
     ) = av_data_df[['Temp_{0:d}T'.format(f), 'CpT_{0:d}T'.format(f),
                  'T_err_{0:d}T'.format(f), 'CpT_err_{0:d}T'.format(f)]
                       ].values.T

fits_df = pd.read_csv(main_path+'Fits.csv')

xvals = (fits_df['Temp'].values).flatten()
                     
fig, ax1 = plt.subplots()

ax1.errorbar(ts[0], cps[0], yerr=cp_err[0],
             fmt='ko', markersize=3, capsize=2, zorder=1)


ax1.set_xlim(0, 33)
ax1.set_ylim(0, 1.99)

ax1.text(24, 0.13, '$\mu_0 H$ = 0 T',
                fontsize=18)

# ax1.minorticks_on()
# ax1.tick_params('both', which='both', direction='in', labelsize=12)
# ax1.yaxis.set_minor_locator(MultipleLocator(0.05))

publication_plot(ax1, r'$T^2$ (K$^2$)', r'$C_p/T$ (mJ mol$^{-1}$ K$^{-2}$)',label_fontsize=20, tick_fontsize=18)

# ax1.set_ylabel('$C_p/T$ (mJ mol$^{-1}$ K$^{-2}$)',
#                   fontsize=15)
# ax1.set_xlabel('$T^2$ (K$^2$)',
#                   fontsize=15)


ppmsT, ppmsCp, ppmsCp_err = pd.read_csv(main_path+'ppms_vt36.csv').values.T

axin = fig.add_axes([0.24, 0.63, 0.34, 0.25])

axin.errorbar(ppmsT, ppmsCp*1e-3, yerr=ppmsCp_err*1e-3,
             fmt='o', markersize=1.5, capsize=2, zorder=1,
             elinewidth=0.5, capthick=0.7,
             color='#0000ff')

axin.legend(frameon=False)

axin.set_ylim(-5, 49)
axin.set_xlim(0, 85)

axin.hlines([0], 0, 100, color='#888888', zorder=-1, linewidth=1)

publication_plot(axin, r'$T$ (K)', r'$C_p$ (J mol$^{-1}$ K$^{-1}$)       ', label_fontsize=14, tick_fontsize=12)

# axin.set_xlabel('$T$ (K)',
#                 fontsize=10, labelpad=1)
# axin.set_ylabel('$C_p$ (J mol$^{-1}$ K$^{-1}$)   ',
#                 fontsize=10, labelpad=0)
#
# axin.tick_params(axis='both', which='major', direction='in', labelsize=8)
# axin.yaxis.set_major_locator(MultipleLocator(10))

plt.savefig(main_path+'HeatCapacity_fig2.pdf', bbox_inches='tight')
# plt.show()

