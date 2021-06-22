import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage.filters
import seaborn as sns
import matplotlib.colors

flux_quantum = 7.748091729 * 10 **-5

main_path = '/Users/npopiel/Documents/MPhil/Data/step_data/'

curr_500 = ['VT64_1p75K500uA.csv',
            'VT64_2p0K500uA.csv',
            'VT64_2p25K500uA.csv',
            'VT64_2p5K500uA.csv',
            'VT64_2p75K500uA.csv',
            'VT64_3p0K500uA.csv',
            'VT64_3p25K500uA.csv',
            'VT64_3p5K500uA.csv',
            'VT64_3p75K500uA.csv',
            'VT64_4p0K500uA.csv']
curr_600 = ['VT64_1p75K600uA.csv',
            'VT64_2p0K600uA.csv',
            'VT64_2p25K600uA.csv',
            'VT64_2p5K600uA.csv']
curr_700 = ['VT64_1p75K700uA.csv',
            'VT64_2p0K700uA.csv',
            'VT64_2p25K700uA.csv']
curr_800 = ['VT64_1p75K800uA.csv',
            'VT64_2p0K800uA.csv',
            'VT64_2p25K800uA.csv',
            'VT64_2p5K800uA.csv',
            'VT64_2p75K800uA.csv',
            'VT64_3p0K800uA.csv',
            'VT64_3p25K800uA.csv',
            'VT64_3p5K800uA.csv',
            'VT64_3p75K800uA.csv',
            'VT64_4p0K800uA.csv']
curr_900 = ['VT64_1p75K900uA.csv',
            'VT64_2p0K900uA.csv',
            'VT64_2p25K900uA.csv',
            'VT64_2p5K900uA.csv',
            'VT64_2p75K900uA.csv',
            'VT64_3p0K900uA.csv',
            'VT64_3p25K900uA.csv',
            'VT64_3p5K900uA.csv',
            'VT64_3p75K900uA.csv',
            'VT64_4p0K900uA.csv']
curr_1000 = ['VT64_1p75K1000uA.csv',
            'VT64_2p0K1000uA.csv',
            'VT64_2p25K1000uA.csv',
            'VT64_2p5K1000uA.csv',
            'VT64_2p75K1000uA.csv',
            'VT64_3p0K1000uA.csv',
            'VT64_3p25K1000uA.csv',
            'VT64_3p5K1000uA.csv',
            'VT64_3p75K1000uA.csv',
            'VT64_4p0K1000uA.csv']
curr_1100 = ['VT64_1p75K1100uA.csv',
            'VT64_2p0K1100uA.csv',
            'VT64_2p25K1100uA.csv',
            'VT64_2p5K1100uA.csv',
            'VT64_2p75K1100uA.csv',
            'VT64_3p0K1100uA.csv',
            'VT64_3p25K1100uA.csv',
            'VT64_3p5K1100uA.csv',
            'VT64_3p75K1100uA.csv',
            'VT64_4p0K1100uA.csv']
curr_1200 = ['VT64_1p75K1200uA.csv',
            'VT64_2p0K1200uA.csv',
            'VT64_2p25K1200uA.csv',
            'VT64_2p5K1200uA.csv',
            'VT64_2p75K1200uA.csv',
            'VT64_3p0K1200uA.csv',
            'VT64_3p25K1200uA.csv',
            'VT64_3p5K1200uA.csv',
            'VT64_3p75K1200uA.csv',
            'VT64_4p0K1200uA.csv']
curr_1500 = ['VT64_1p75K1500uA.csv',
            'VT64_2p0K1500uA.csv',
            'VT64_2p25K1500uA.csv',
            'VT64_2p5K1500uA.csv',
            'VT64_2p75K1500uA.csv',
            'VT64_3p0K1500uA.csv',
            'VT64_3p25K1500uA.csv',
            'VT64_3p5K1500uA.csv',
            'VT64_3p75K1500uA.csv']

temps_a = ['1.75 K +', '1.75 K -', '2.00 K +', '2.00 K -', '2.25 K +','2.25 K -',
           '2.50 K +', '2.50 K -', '2.75 K +', '2.75 K -', '3.00 K +', '3.00 K -',
           '3.25 K +', '3.25 K -', '3.5 K +', '3.5 K -', '3.75 K +','3.75 K -', '4.00 K +', '4.00 K -']
temps_b = ['1.75 K +', '1.75 K -', '2.00 K +', '2.00 K -', '2.25 K +','2.25 K -',
           '2.50 K +', '2.50 K -', '2.75 K +', '2.75 K -', '3.00 K +', '3.00 K -',
           '3.25 K +', '3.25 K -', '3.5 K +', '3.5 K -', '3.75 K +','3.75 K -']

temps_c = ['1.75 K +', '1.75 K -', '2.00 K +', '2.00 K -', '2.25 K +','2.25 K -',
           '2.50 K +', '2.50 K -']
temps_d = ['1.75 K +', '1.75 K -', '2.00 K +', '2.00 K -', '2.25 K +','2.25 K -']
currents = [r'500 $\mu A$', r'600 $\mu A$',r'700 $\mu A$',r'800 $\mu A$', r'900 $\mu A$', r'1000 $\mu A$', r'1100 $\mu A$', r'1200 $\mu A$', r'1500 $\mu A$']


data_sets = [curr_500, curr_600, curr_700, curr_800, curr_900, curr_1000, curr_1100, curr_1200, curr_1500]

sample = 'VT64'


res_up_plus, res_down_plus, res_up_minus, res_down_minus = [], [], [], []
b_up_plus, b_down_plus, b_up_minus, b_down_minus = [], [], [], []

fig, axs = MakePlot(nrows=3,ncols=3).create()

axes = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2], axs[2,0], axs[2,1], axs[2,2]]



for ind, curr_data_name in enumerate(data_sets):
    if ind == 0 or 2<ind<=5:
        temps = temps_a

    elif ind == 1 :
        temps = temps_c
    elif ind == 2:
        temps = temps_d
    else:
        temps = temps_b
    res_lst, field_lst = [], []
    label_lst = []

    c = 0

    for idx, current_temp_data_name in enumerate(curr_data_name):
        dat = load_matrix(main_path+current_temp_data_name).T
        res_lst.append(dat[0])
        field_lst.append(dat[1])
        label_lst.append([temps[idx]]*len(dat[0]))

        start_loc = 0
        max_loc = np.argmax(dat[1])
        min_loc = np.argmin(dat[1])
        mid_loc = (min_loc-max_loc)/2
        mid_loc = int(mid_loc.round(0))
        end_loc = dat[1].shape[0]

        sweep_up_locs_pos_field = np.arange(start_loc, max_loc)
        sweep_down_locs_pos_field = np.arange(max_loc, mid_loc-1)
        sweep_up_locs_neg_field = np.arange(mid_loc+1, min_loc)
        sweep_down_locs_neg_field = np.arange(min_loc, end_loc)

        # res_up_plus.append(dat[0][sweep_up_locs_pos_field])
        axes[ind].plot(dat[1][[sweep_up_locs_pos_field]],
               dat[0][[sweep_up_locs_pos_field]] / flux_quantum,
                label=temps[c],color=plt.cm.jet(c/10),linewidth=1.2)
        axes[ind].plot(dat[1][[sweep_down_locs_pos_field]],
                dat[0][[sweep_down_locs_pos_field]] / flux_quantum,
                linestyle='dashed', label=temps[c+1],color=plt.cm.jet(c/10),linewidth=1.2)

        axes[ind].plot(dat[1][[sweep_up_locs_neg_field]],
                dat[0][[sweep_up_locs_neg_field]] / flux_quantum,
                label=temps[c],color=plt.cm.jet(c/10),linewidth=1.2)
        axes[ind].plot(dat[1][[sweep_down_locs_neg_field]],
                dat[0][[sweep_down_locs_neg_field]] / flux_quantum,
                linestyle='dashed', label=temps[c+1],color=plt.cm.jet(c/10),linewidth=1.2)

        c+=1

    axes[ind].ticklabel_format(style='sci', axis='y',useMathText=True) #, scilimits=(0, 0)

    if ind == 6 :
        axes[ind].set_xlabel(r'Magnetic Field $(T)$', fontsize=14, fontname='arial')
    elif ind == 7:
        axes[ind].set_xlabel(r'Magnetic Field $(T)$', fontsize=14, fontname='arial')
    elif ind == 8:
        axes[ind].set_xlabel(r'Magnetic Field $(T)$', fontsize=14, fontname='arial')
    else:
        axes[ind].set_xlabel(r'', fontsize=14, fontname='arial')

    if ind == 0:
        axes[ind].set_ylabel(r'Resistance $(\frac{h}{2e^2})$', fontsize=14, fontname='arial')
    elif ind == 3:
        axes[ind].set_ylabel(r'Resistance $(\frac{h}{2e^2})$', fontsize=14, fontname='arial')
    elif ind == 6:
        axes[ind].set_ylabel(r'Resistance $(\frac{h}{2e^2})$', fontsize=14, fontname='arial')
    else:
        axes[ind].set_ylabel(r'', fontsize=14, fontname='arial')
    axes[ind].set_xlim()
    axes[ind].set_ylim()
    axes[ind].minorticks_on()
    axes[ind].tick_params('both', which='both', direction='in',
                    bottom=True, top=True, left=True, right=True)
    axes[ind].set_title(currents[ind])


handles, labels = axes[6].get_legend_handles_labels()

labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]

plt.subplots_adjust(hspace=0.25)
fig.legend(handles, labels, title='Temperature', loc="center right",# mode='expand',  # Position of legend
           borderaxespad=0.1, framealpha=0)
           #bbox_to_anchor=(0.5, 0),bbox_transform = plt.gcf().transFigure )


#sns.lineplot(x=r'Magnetic Field $(T)$', y=r'Resistance $(\Omega)$', data = df, hue=r'Temperature')
plt.suptitle('Magnetoresistance of VT64', fontsize=18, fontname='arial')
#ax.legend(title='Temperature',loc='right', fontsize=12)#,bbox_to_anchor=(1, 1), borderaxespad=0.)
#plt.tight_layout()
#plt.savefig('/Users/npopiel/Documents/MPhil/Data/step_data/Grouped by Current/VT64_conductance_'+ currents[ind]+'.png', dpi=600)
plt.show()
plt.close()
