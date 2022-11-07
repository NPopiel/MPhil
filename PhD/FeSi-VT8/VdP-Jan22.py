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
from tools.ColorMaps import *



'''
All the paths
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FeSi_FSVT8_VdP_Rxx_0T_1p8K_dcIV.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FeSi_FSVT8_VdP_Rxx_Rxy_cooldown.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FeSi_FSVT8_VdP_Rxx_Rxy_cooldown2-justRxx.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FeSi_FSVT8_VdP_Rxx_Rxy_cooldown2.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_-5Tcooldown.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_-7Tcooldown.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_-9Tcooldown.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_0Twarmup.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_5Twarmup.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_7Twarmup.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_9T-IV.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_9Twarmup.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT8-Rxx_Rxy_cooldown.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT9-3K-Rxx-Rxy-FS.dat
/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/FSVT9-4K-Rxx-Rxy-FS.dat

Plan

Plot the IV

Plot all of the cooldowns + warmups with different fields


'''

main_path = '/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/VdP/'

relevant_columns = ['Temperature (K)',
                    'Magnetic Field (Oe)',
                    'Bridge 1 Resistance (Ohms)',
                    'Bridge 1 Excitation (uA)'
                    'Bridge 2 Resistance (Ohms)',
                    'Bridge 2 Excitation (uA)']

filename_iv2 = 'FSVT8-Rxx_Rxy_9T-IV.dat'
filename_iv1 = 'FeSi_FSVT8_VdP_Rxx_0T_1p8K_dcIV.dat'

Rxx_iv1 = np.array(load_matrix(main_path+filename_iv1)['Bridge 1 Resistance (Ohms)'])
Rxx_iv2 = np.array(load_matrix(main_path+filename_iv2)['Bridge 1 Resistance (Ohms)'])
Rxy_iv2 = np.array(load_matrix(main_path+filename_iv2)['Bridge 3 Resistance (Ohms)'])

current1 = np.array(load_matrix(main_path+filename_iv1)['Bridge 1 Excitation (uA)'])
current2 = np.array(load_matrix(main_path+filename_iv2)['Bridge 1 Excitation (uA)'])


fig, ax = MakePlot(figsize=(6,9)).create()


cmap = select_discrete_cmap('bulbasaur')
ax.plot(current1, Rxx_iv1,label="R$_{xx}$ 0 T",c=cmap[0])
ax.plot(current2, Rxx_iv2,label="R$_{xx}$ 9 T",c=cmap[4])
ax.plot(current2, Rxy_iv2,label="R$_{xy}$ 9 T",c=cmap[7])

publication_plot(ax, 'Current ($\mu$A)', 'Resistance ($\Omega$)')


legend = ax.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())




plt.tight_layout(pad=.5)
plt.show()

# Make the RvT graphs

filenames = ['FSVT8-Rxx_Rxy_cooldown.dat',
             'FSVT8-Rxx_Rxy_7Twarmup.dat',
             'FSVT8-Rxx_Rxy_9Twarmup.dat',
             'FSVT8-Rxx_Rxy_-7Tcooldown.dat',
             'FSVT8-Rxx_Rxy_-9Tcooldown.dat']

labels = ['0 T',
          '7 T',
          '9 T',
          '-7 T',
          '-9 T']




fig, ax = MakePlot(figsize=(6,9)).create()

cm = select_discrete_cmap('blastoise')

for i, file in enumerate(filenames):

    Rxx = np.array(load_matrix(main_path+file)['Bridge 1 Resistance (Ohms)'])
    Rxy = np.array(load_matrix(main_path+file)['Bridge 3 Resistance (Ohms)'])
    T = np.array(load_matrix(main_path+file)['Temperature (K)'])

    ax.plot(T,Rxx,label=labels[i],c=cm[i])
    ax.plot(T,Rxy,label=labels[i], c=cm[i],linestyle='dashed')

publication_plot(ax, 'Temperature (K)', 'Resistance ($\Omega$)',x_ax_log=True,y_ax_log=True)


legend = ax.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())




plt.tight_layout(pad=.5)
plt.show()


filename_iv = 'FSVT8_NL_L_IV_1p8K_0T.dat'

R_iv_NL = load_matrix(main_path+filename_iv)['Bridge 1 Resistance (Ohms)']
R_iv_L = load_matrix(main_path+filename_iv)['Bridge 2 Resistance (Ohms)']

i_NL = load_matrix(main_path+filename_iv)['Bridge 1 Excitation (uA)']
i_L = load_matrix(main_path+filename_iv)['Bridge 2 Excitation (uA)']

V_NL = R_iv_NL * i_NL /1e6
V_L = R_iv_L * i_L /1e6

fig, ax = MakePlot(figsize=(9,9)).create()

ax.plot(i_L, V_L,label="Local",c='indianred', linewidth=2)
ax.plot(i_NL, V_NL,label="Non-local",c='midnightblue', linewidth=2)

publication_plot(ax, 'Current (A)', 'Effective Voltage (V)')


legend = ax.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())




plt.tight_layout(pad=.5)
plt.show()
#
# # COmbne the IV and R(T) into a nice graph
#
#
# fig, axs = MakePlot(ncols=3, figsize=(18,9)).create()
# gs = axs[2].get_gridspec()
# # remove the underlying axes
# for ax in axs[1:]:
#     ax.remove()
# axbig = fig.add_subplot(gs[1:])
#
# ax = axs[0]
# ax.plot(temps_cooldown, R_cooldown_L,label="Local",c='indianred')
# ax.plot(temps[range2], R_warmup_NL[range2],c='midnightblue')
# ax.plot(temps[range1], R_warmup_NL[range1],label="Non-local",c='midnightblue')
# ax.plot(temps[locs_negative], R_warmup_NL[locs_negative], label='Non-local (negative)', c='darkslategrey')
#
#
# publication_plot(ax, 'Temperature (K)', 'Resistance ($\Omega$)',x_ax_log=True)
#
#
# legend = ax.legend(framealpha=0, ncol=1, loc='lower left',
#               prop={'size': 24, 'family': 'arial'}, handlelength=0)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())
#
#
# ax = axbig
# ax.plot(i_L, V_L,label="Local",c='indianred', linewidth=2)
# ax.plot(i_NL, V_NL,label="Non-local",c='midnightblue', linewidth=2)
#
# publication_plot(ax, 'Current ($\mu$A)', 'Effective Voltage (V)')
#
#
# legend = ax.legend(framealpha=0, ncol=1, loc='best',
#               prop={'size': 24, 'family': 'arial'}, handlelength=0)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())
#
# ax.annotate('FeSi VT8',xy=(500,-0.2),fontname='arial',fontsize=24,ha='center',va='center')
# ax.annotate('1.8 K',xy=(500,-0.3),fontname='arial',fontsize=24,ha='center',va='center')
#
#
# plt.tight_layout(pad=.5)
# plt.savefig(main_path+'FeSi_RvT_IV_logged.png',dpi=200)
# plt.show()


#
# # Plot the 1.8 K field sweeps
#
# filename_10 = 'FSVT8_NL_L_10uA_1p8K_FS.dat'
# filename_100 = 'FSVT8_NL_L_100uA_1p8K_FS.dat'
# filename_750 = 'FSVT8_NL_L_750uA_1.8K_FS-try2.dat'
# filename_1000 = 'FSVT8_NL_L_1000uA_1p8K_FS.dat'
#
# filenames = [filename_10,filename_100,filename_750,filename_1000]
#
# ig, axs = MakePlot(figsize=(9,9), ncols=2).create()
#
# colors = ['#2E4C6D',
#           '#396EB0',
#           '#DADDFC',
#           '#FC997C']
#
# labels = ['10 $\mu$A','100 $\mu$A','750 $\mu$A','1000 $\mu$A']
#
# for ind, name in enumerate(filenames):
#
#     R_NL = load_matrix(main_path + name)['Bridge 1 Resistance (Ohms)']
#     R_L = load_matrix(main_path + name)['Bridge 2 Resistance (Ohms)']
#     B = load_matrix(main_path + name)['Magnetic Field (Oe)']/1e4
#
#     axs[0].plot(B, R_L, label=labels[ind], c=colors[ind], linewidth=2)
#     axs[1].plot(B, R_NL, label=labels[ind], c=colors[ind], linewidth=2)
#
# publication_plot(axs[0], 'Magnetic Field (T)', 'Resistance ($\Omega$)', title='Local Contacts')
# publication_plot(axs[1], 'Magnetic Field (T)', 'Resistance ($\Omega$)', title='Non-local Contacts')
#
# # handles, labels = ax.get_legend_handles_labels()
# # line1 = Line2D([], [],label=r'Non-local contact',
# #           linestyle='dashed', color='k')
# #
# # line2 = Line2D([], [],label=r'Local contact', color='k')
# # handles.extend([line1, line2])
#
#
# legend = axs[0].legend(framealpha=0, ncol=1, loc='best',
#               title='Current', prop={'size': 16, 'family': 'arial'})
# plt.setp(legend.get_title(), fontsize=20, fontname='arial')
#
# plt.tight_layout(pad=.5)
# plt.show()



# # Explore the 750 field sweeps at different T
#
#
#
#
# filenames = ['/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/FSVT8_NL_L_750uA_1.8K_FS-try2.dat',
# '/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/FSVT8_NL_L_750uA_2.5K_FS.dat',
# '/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/FSVT8_NL_L_750uA_3.5K_FS.dat',
# '/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/FSVT8_NL_L_750uA_4K_FS.dat',
# '/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/FSVT8_NL_L_750uA_4.5K_FS.dat',
# '/Volumes/GoogleDrive/My Drive/FeSi/FSVT8/FSVT8_NL_L_750uA_5K_FS.dat'
# ]
#
# ig, axs = MakePlot(figsize=(12,12), ncols=2).create()
#
# colors = ['#161853',
#           '#00A19D',
#           '#3DB2FF',
#           '#FFB830',
#           '#DB6400',
#           '#FF2442']
#
# labels = ['1.8 K', '2.5 K', '3.5 K', '4.0 K', '4.5 K', '5.0 K']
#
# colors2 = []
#
# for ind, name in enumerate(filenames):
#
#     R_NL = load_matrix(name)['Bridge 1 Resistance (Ohms)']
#     R_L = load_matrix(name)['Bridge 2 Resistance (Ohms)']
#     B = load_matrix(name)['Magnetic Field (Oe)']/1e4
#
#     axs[0].plot(B, R_L, label=labels[ind], c=colors[ind], linewidth=2)
#     axs[1].plot(B, R_NL, label=labels[ind], c=colors[ind], linewidth=2)
#
# publication_plot(axs[0], 'Magnetic Field (T)', 'Resistance ($\Omega$)', title='Local Contacts')
# publication_plot(axs[1], 'Magnetic Field (T)', 'Resistance ($\Omega$)', title='Non-local Contacts')
#
# # handles, labels = ax.get_legend_handles_labels()
# # line1 = Line2D([], [],label=r'Non-local contact',
# #           linestyle='dashed', color='k')
# #
# # line2 = Line2D([], [],label=r'Local contact', color='k')
# # handles.extend([line1, line2])
#
#
#
# legend = axs[0].legend(framealpha=0, ncol=1, loc='best',
#               prop={'size': 20, 'family': 'arial'}, handlelength=0)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())
#
# axs[1].annotate('FeSi VT8',xy=(-10,74),fontname='arial',fontsize=20,ha='center',va='center')
# axs[1].annotate(r'$\mathbf{j}$ = 750 $\mu$A $\parallel$ [100]',xy=(-6.5,72.8),fontname='arial',fontsize=20,ha='center',va='center')
#
# plt.tight_layout(pad=.5)
#
# plt.savefig(main_path+'FeSiVT8_750uA_Fieldsweeps.png')
# plt.show()


# Plot R v I and GvI


filename_iv = 'FSVT8_NL_L_IV_1p8K_0T.dat'

R_iv_NL = load_matrix(main_path+filename_iv)['Bridge 1 Resistance (Ohms)']
R_iv_L = load_matrix(main_path+filename_iv)['Bridge 2 Resistance (Ohms)']

flux_quantum = 7.748091729 * 10 **-5

G_NL = 1 / R_iv_NL / flux_quantum
G_L = 1 / R_iv_L / flux_quantum

i_NL = load_matrix(main_path+filename_iv)['Bridge 1 Excitation (uA)']
i_L = load_matrix(main_path+filename_iv)['Bridge 2 Excitation (uA)']

V_NL = R_iv_NL * i_NL /1e6
V_L = R_iv_L * i_L /1e6

fig, ax = MakePlot(figsize=(9,9)).create()

ax.plot(i_L, R_iv_L,label="Local",c='indianred', linewidth=2)
ax.plot(i_NL, R_iv_NL,label="Non-local",c='midnightblue', linewidth=2)

publication_plot(ax, 'Current ($\mu$A)', 'Resistance ($\Omega$)')


legend = ax.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=.5)
plt.show()

# Plot the GvI

fig, ax = MakePlot(figsize=(9,9)).create()

ax.plot(i_L, G_L,label="Local",c='indianred', linewidth=2)
#ax.plot(i_NL[:1332], G_NL[:1332],label="Non-local",c='midnightblue', linewidth=2)
#ax.plot(i_NL[1358:], G_NL[1358:],c='midnightblue', linewidth=2)

publication_plot(ax, 'Current ($\mu$A)', r'Conductance ($\frac{2 e^2}{h}$)')


legend = ax.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=.5)
plt.show()
