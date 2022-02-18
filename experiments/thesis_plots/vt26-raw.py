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
temps = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0)#,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0)

temp_lab1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]#,22,23,24,25,26,27,28,29]#,19,20,21]#,22]


possible_currents1 = np.array([10,20,50,100,200,500,1000,1500])


#currents_log = np.logspace(-3,np.log10(1.5),10000)
currents_log = np.linspace(0.01,1.5,10000)

temperoos, cureentoos = [], []

temp_lst = []

xx, yy = np.meshgrid(temps, possible_currents1 / 1000)
data_grid_og = np.array([np.ravel(xx), np.ravel(yy)]).T
temperatures = np.linspace(2, temps[-1], 10000)

grid_x, grid_y = np.meshgrid(temperatures, currents_log)

sample_lst = []

fig, axs = MakePlot(figsize=(32,18),nrows=5, ncols=4).create()

axs = [axs[0,0], axs[0,1], axs[0,2],axs[0,3],
       axs[1,0], axs[1,1], axs[1,2],axs[1,3],
       axs[2,0], axs[2,1], axs[2,2],axs[2,3],
       axs[3,0], axs[3,1], axs[3,2],axs[3,3],
       axs[4,0], axs[4,1], axs[4,2],axs[4,3]]

for i, temp in enumerate(temps):

    temp_path = main_path + 'VT26/' + str(temp) + '/'

    current_lst = []

    ax = axs[i]

    max_r_list, min_r_list = [], []

    for ind, current in enumerate(possible_currents1):
        resistance, field = load_r_and_h(temp_path, current)

        max_r_list.append(np.max(resistance))
        min_r_list.append(np.min(resistance))

        lab = str(current) + r' $\mu \mathrm{A}$'

        ax.plot(field,resistance,linewidth=1.8,c=plt.cm.viridis(ind/len(possible_currents1)), label=lab)


    max_r = np.max(max_r_list)
    min_r = np.min(min_r_list)



    ax.set_title(str(temp_lab1[i]) + ' K',fontname='arial',fontsize=44)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    #ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.minorticks_on()

    ax.tick_params('both', which='major', direction='in', length=6, width=2,
                   bottom=True, top=True, left=True, right=True)

    ax.tick_params('both', which='minor', direction='in', length=4, width=1.5,
                   bottom=True, top=True, left=True, right=True)
    ax.yaxis.offsetText.set_fontsize(27)
    # plt.legend(title=r'Temperature $(K)$', loc='best',frameon=False, fancybox=False, framealpha=0, borderpad=1)
    # ax.set_title('Magnetization in Positive Field Region',fontsize=title_size,fontname='arial', pad=30)
    #
    # ax.annotate(r'FeSb$_2$',xy=(4,3.2e-2),fontname='arial',fontsize=24,va='center',ha='center')
    # ax.annotate(r'VT66',xy=(4,2.9e-2),fontname='arial',fontsize=24,va='center',ha='center')
    # ax.annotate(r'$ \vec H \parallel c $',xy=(4,2.6e-2), fontname='arial',fontsize=24,va='center',ha='center')


    plt.setp(ax.get_xticklabels(), fontsize=36, fontname='arial')
    plt.setp(ax.get_yticklabels(), fontsize=36, fontname='arial')
    ax.set_ylim((min_r, max_r))
    ax.locator_params(axis='y', nbins=2)

    if i <=15:
        ax.set_xticklabels([])


plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1,  top=0.9, wspace=0.2, hspace=0.4)

# handles, labels = ax.get_legend_handles_labels()
# legend = fig.legend(handles, labels, framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
#                        title='Current (mA)', prop={"size": 18})
# plt.setp(legend.get_title(), fontsize=20, fontname='arial')

cbar_ax = fig.add_axes([0.91, 0.09, 0.02, 0.84])

cbar_tick_locs = np.arange(len(possible_currents1))/len(possible_currents1)

sm = plt.cm.ScalarMappable(cmap='viridis')
cbar = fig.colorbar(sm, ticks=cbar_tick_locs, cax=cbar_ax,pad=0.04, aspect = 30)
cbar.ax.set_yticklabels(possible_currents1)
cbar.ax.set_title(r'Current $(\mu \mathrm{A})$',fontname='arial', fontsize=36,pad=10.0)
cbar.ax.tick_params(labelsize=28)

fig.text(0.4,0.04, 'Magnetic Field (T)',fontname='arial',fontsize=58)
fig.text(0.04,0.4, r'Resistance $(\Omega)$',fontname='arial',fontsize=58,rotation=90)
#plt.tight_layout()
#plt.show()
fig.savefig(save_path+'raw_VT26-draft1.png', bbox_inches='tight',dpi=400)

