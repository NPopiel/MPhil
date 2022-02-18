import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

main_path = '/Users/npopiel/Documents/MPhil/Data/'

save_path = '/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/'

samples = ['VT26']
temps = [2.0, 5.0]#,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0)#,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0)
temps2 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0)#,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0)

temp_lab1 = [2, 5]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]#,22,23,24,25,26,27,28,29]#,19,20,21]#,22]
temp_lab2 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]#,22,23,24,25,26,27,28,29]#,19,20,21]#,22]


possible_currents1 = np.array([10,20,50,100,200,500,1000,1500])

possible_currents2 = np.array([50,1000])

fig, axs = MakePlot(figsize=(32,18),nrows=2, ncols=2).create()

axs_temp = [axs[0,0], axs[0,1]]

axs_curr = [axs[1,0], axs[1,1]]

for i, temp in enumerate(temps):

    temp_path = main_path + 'VT26/' + str(temp) + '/'

    ax = axs_temp[i]

    if i == 1:

        possible_currents1 = possible_currents1[:-1]

    for ind, current in enumerate(possible_currents1):

        resistance, field = load_r_and_h(temp_path, current)



        lab = str(current) + r' $\mu \mathrm{A}$'

        ax.plot(field,resistance/resistance[0],linewidth=1.8,c=plt.cm.viridis(ind/len(possible_currents1)))


        max_r = np.max(resistance)
        min_r = np.min(resistance)



    ax.set_title(str(temp_lab1[i]) + r' K',fontname='arial',fontsize=44)
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    ax.minorticks_on()

    ax.tick_params('both', which='major', direction='in', length=6, width=2,
                       bottom=True, top=True, left=True, right=True)

    ax.tick_params('both', which='minor', direction='in', length=4, width=1.5,
                       bottom=True, top=True, left=True, right=True)
    ax.yaxis.offsetText.set_fontsize(27)


    plt.setp(ax.get_xticklabels(), fontsize=36, fontname='arial')
    plt.setp(ax.get_yticklabels(), fontsize=36, fontname='arial')
    ax.locator_params(axis='y', nbins=2)

    ax.set_xticklabels([])

    if i < 1:
        cbar_ax = fig.add_axes([0.91, 0.565, 0.02, 0.335])

        cbar_tick_locs = np.arange(len(possible_currents1)) / len(possible_currents1)

        sm = plt.cm.ScalarMappable(cmap='viridis')
        cbar = fig.colorbar(sm, ticks=cbar_tick_locs, cax=cbar_ax, pad=15, aspect=30)
        cbar.ax.set_yticklabels(possible_currents1)
        cbar.ax.set_title(r'Current $(\mu \mathrm{A})$', fontname='arial', fontsize=36, pad=10.0)
        cbar.ax.tick_params(labelsize=28)



for i, current in enumerate(possible_currents2):



    ax = axs_curr[i]

    for ind, temp in enumerate(temps2):

        temp_path = main_path + 'VT26/' + str(temp) + '/'

        resistance, field = load_r_and_h(temp_path, current)

        lab = str(temp_lab2[ind]) + r' K'

        ax.plot(field,resistance/resistance[0],linewidth=1.8,c=plt.cm.coolwarm(ind/len(temps2)))



    ax.set_title(str(current) + r' $\mu \mathrm{A}$',fontname='arial',fontsize=44)
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    ax.minorticks_on()

    ax.tick_params('both', which='major', direction='in', length=6, width=2,
                       bottom=True, top=True, left=True, right=True)

    ax.tick_params('both', which='minor', direction='in', length=4, width=1.5,
                       bottom=True, top=True, left=True, right=True)
    ax.yaxis.offsetText.set_fontsize(27)


    plt.setp(ax.get_xticklabels(), fontsize=36, fontname='arial')
    plt.setp(ax.get_yticklabels(), fontsize=36, fontname='arial')

    ax.locator_params(axis='y', nbins=2)

    if i < 1:

        # handles, labels = ax.get_legend_handles_labels()
        # legend = fig.legend(handles, labels, framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
        #                        title='Temperature (k)', prop={"size": 18})
        # plt.setp(legend.get_title(), fontsize=20, fontname='arial')

        cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.335])

        cbar_tick_locs = np.arange(len(temp_lab2))[::2] / len(temp_lab2)

        sm = plt.cm.ScalarMappable(cmap='coolwarm')
        cbar = fig.colorbar(sm, ticks=cbar_tick_locs, cax=cbar_ax, pad=0.04, aspect=30)
        cbar.ax.set_yticklabels(temp_lab2[::2])
        cbar.ax.set_title(r'Temperature (K)', fontname='arial', fontsize=36, pad=10.0)
        cbar.ax.tick_params(labelsize=28)

plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1,  top=0.9, wspace=0.2, hspace=0.4)

# handles, labels = ax.get_legend_handles_labels()
# legend = fig.legend(handles, labels, framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
#                        title='Current (mA)', prop={"size": 18})
# plt.setp(legend.get_title(), fontsize=20, fontname='arial')

# cbar_ax = fig.add_axes([0.91, 0.09, 0.02, 0.84])
#
# cbar_tick_locs = np.arange(len(possible_currents1))/len(possible_currents1)
#
# sm = plt.cm.ScalarMappable(cmap='viridis')
# cbar = fig.colorbar(sm, ticks=cbar_tick_locs, cax=cbar_ax,pad=0.04, aspect = 30)
# cbar.ax.set_yticklabels(possible_currents1)
# cbar.ax.set_title(r'Current $(\mu \mathrm{A})$',fontname='arial', fontsize=36,pad=10.0)
# cbar.ax.tick_params(labelsize=28)

fig.text(0.4,0.02, 'Magnetic Field (T)',fontname='arial',fontsize=58)
fig.text(0.04,0.5, 'Magnetoresistance\n Ratio',fontname='arial',ha='center',va='center',fontsize=58,rotation=90)
#fig.suptitle('10 Kelvin',fontname='arial',fontsize=69)
#plt.tight_layout()
#plt.show()
fig.savefig('/Users/npopiel/Desktop/VT26-2.pdf',dpi=400)

