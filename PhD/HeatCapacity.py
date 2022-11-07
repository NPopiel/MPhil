#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:22:05 2020

@author: alexanderhickey
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

fit_mean = {}
fit_low = {}
fit_high = {}

for f in range(0, 11, 2):
    (fit_mean[f], fit_low[f], fit_high[f]
     ) = fits_df[['CpT_mean_{0:d}T'.format(f),'CpT_low_{0:d}T'.format(f),
                     'CpT_High_{0:d}T'.format(f)]
                       ].values.T

fig, big_ax = MakePlot(figsize=(8, 11)).create()

big_ax.set_frame_on(False)
big_ax.set_yticks([])
big_ax.set_xticks([])
big_ax.set_ylabel('Specific Heat per Temperature (mJ mol$^{-1}$ K$^{-2}$)',
                  fontsize=18, labelpad=23)
big_ax.set_xlabel('Temperature Squared (K$^2$)',
                  fontsize=18, labelpad=20)

ax1 = fig.add_subplot(4, 1, 1)

ax1.errorbar(ts[0], cps[0], xerr=t_err[0], yerr=cp_err[0],
             fmt='ko', markersize=3, capsize=2, zorder=1)
ax1.plot(xvals, fit_mean[0], 'r-', zorder=2)
ax1.fill_between(xvals, fit_low[0], fit_high[0],
                 color='r', alpha=0.5, zorder=2, linewidth=0)

ax1.set_xlim(0, 33)
ax1.set_ylim(0, 1.95)

ax1.text(2, 1.6, '$\mu_0 H_0$ = 0 T',
                fontsize=17)

ax1.minorticks_on()
ax1.tick_params('both', which='both', direction='in', labelsize=12)

axes = {}

for n, f in enumerate(range(0, 11, 2)):
    axes[n] = fig.add_subplot(4, 2, n+3)
    
    axes[n].errorbar(ts[f], cps[f], yerr=cp_err[f],
                 fmt='ko', markersize=3, capsize=2, zorder=1)
    axes[n].plot(xvals, fit_mean[f], 'r-', zorder=2)
    axes[n].fill_between(xvals, fit_low[f], fit_high[f],
                     color='r', alpha=0.5, zorder=2, linewidth=0)

    axes[n].set_xlim(0, 11.5)
    axes[n].set_ylim(0, 1.19)
    
    axes[n].text(1, 1.03, '$\mu_0 H_0$ = {} T'.format(f),
                fontsize=14)
    
    axes[n].minorticks_on()
    axes[n].tick_params('both', which='both', direction='in', labelsize=12)

fig.tight_layout()    

plt.savefig(main_path+'HeatCapacity_fig2.pdf', bbox_inches='tight')
plt.show()

