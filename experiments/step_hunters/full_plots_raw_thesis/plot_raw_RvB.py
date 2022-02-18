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

fig, axs = MakePlot(nrows=3,ncols=3,figsize=(16,9)).create()

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
               dat[0][[sweep_up_locs_pos_field]],
                label=temps[c],color=plt.cm.jet(c/10),linewidth=1.2)
        axes[ind].plot(dat[1][[sweep_down_locs_pos_field]],
                dat[0][[sweep_down_locs_pos_field]],
                linestyle='dashed', label=temps[c+1],color=plt.cm.jet(c/10),linewidth=1.2)

        axes[ind].plot(dat[1][[sweep_up_locs_neg_field]],
                dat[0][[sweep_up_locs_neg_field]],
                label=temps[c],color=plt.cm.jet(c/10),linewidth=1.2)
        axes[ind].plot(dat[1][[sweep_down_locs_neg_field]],
                dat[0][[sweep_down_locs_neg_field]],
                linestyle='dashed', label=temps[c+1],color=plt.cm.jet(c/10),linewidth=1.2)

        c+=1


    if ind == 6 :
        #axes[ind].set_xlabel(r'Magnetic Field $(T)$', fontsize=22, fontname='arial')
        plt.setp(axes[ind].get_xticklabels(), fontsize=18, fontname='arial')
    elif ind == 7:
        #axes[ind].set_xlabel(r'Magnetic Field $(T)$', fontsize=22, fontname='arial')
        plt.setp(axes[ind].get_xticklabels(), fontsize=18, fontname='arial')
    elif ind == 8:
        #axes[ind].set_xlabel(r'Magnetic Field $(T)$', fontsize=22, fontname='arial')
        plt.setp(axes[ind].get_xticklabels(), fontsize=18, fontname='arial')
    else:
        axes[ind].set_xlabel(r'', fontsize=22, fontname='arial')
        axes[ind].set_xticklabels([])

    if ind == 0:
        #axes[ind].set_ylabel(r'Resistance $(\Omega)$', fontsize=22, fontname='arial')
        print('h')

    elif ind == 3:
        print('h')
        #axes[ind].set_ylabel(r'Resistance $(\Omega)$', fontsize=22, fontname='arial')
    elif ind == 6:
        print('h')
        #axes[ind].set_ylabel(r'Resistance $(\Omega)$', fontsize=22, fontname='arial')
    else:
        axes[ind].set_ylabel(r'', fontsize=22, fontname='arial')
    plt.setp(axes[ind].get_yticklabels(), fontsize=18, fontname='arial')
    axes[ind].set_xlim()
    axes[ind].set_ylim()
    axes[ind].minorticks_on()
    axes[ind].tick_params('both', which='major', direction='in', length=6, width=2,
                   bottom=True, top=True, left=True, right=True)

    axes[ind].tick_params('both', which='minor', direction='in', length=4, width=1.5,
                   bottom=True, top=True, left=True, right=True)
    axes[ind].set_title(currents[ind])

#
# handles, labels = axes[8].get_legend_handles_labels()
#
# labels, ids = np.unique(labels, return_index=True)
# handles = [handles[i] for i in ids]
#
#
#
# plt.subplots_adjust(hspace=0.25)
# legend = fig.legend(handles, labels, title='Temperature', loc='upper right',# mode='expand',  # Position of legend
#             framealpha=0,prop={"size": 16})
#
# plt.setp(legend.get_title(), fontsize=20, fontname='arial')

           #bbox_to_anchor=(0.5, 0),bbox_transform = plt.gcf().transFigure )


plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1,  top=0.9, wspace=0.2, hspace=0.4)

# handles, labels = ax.get_legend_handles_labels()
# legend = fig.legend(handles, labels, framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
#                        title='Current (mA)', prop={"size": 18})
# plt.setp(legend.get_title(), fontsize=20, fontname='arial')

cbar_ax = fig.add_axes([0.91, 0.09, 0.02, 0.84])

cbar_tick_locs = np.arange(10)/10


temps = [1.75,2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00]

sm = plt.cm.ScalarMappable(cmap='jet')
cbar = fig.colorbar(sm, ticks=cbar_tick_locs, cax=cbar_ax,pad=0.04, aspect = 30)
cbar.ax.set_yticklabels(temps)
cbar.ax.set_title(r'Temperature (K)',fontname='arial', fontsize=20,pad=10.0)
cbar.ax.tick_params(labelsize=18)

fig.text(0.4,0.03, 'Magnetic Field (T)',fontname='arial',fontsize=28)
fig.text(0.03,0.4, r'Resistance $(\Omega)$',fontname='arial',fontsize=28,rotation=90)


#sns.lineplot(x=r'Magnetic Field $(T)$', y=r'Resistance $(\Omega)$', data = df, hue=r'Temperature')
#plt.suptitle('Magnetoresistance of VT64', fontsize=18, fontname='arial')
#ax.legend(title='Temperature',loc='right', fontsize=12)#,bbox_to_anchor=(1, 1), borderaxespad=0.)
#plt.tight_layout()
plt.savefig('/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/VT64_RvB_Ohms-draft2.png', bbox_inches='tight',dpi=400)
#plt.show()
#plt.close()
