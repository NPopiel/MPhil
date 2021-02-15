import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage.filters
import seaborn as sns

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
currents = [r'500 $\mu A$', r'800 $\mu A$', r'900 $\mu A$', r'1000 $\mu A$', r'1100 $\mu A$', r'1200 $\mu A$', r'1500 $\mu A$']


data_sets = [curr_500, curr_800, curr_900, curr_1000, curr_1100, curr_1200, curr_1500]

sample = 'VT64'

colours = sns.color_palette('muted')


for ind, curr_data_name in enumerate(data_sets):
    if ind <=5:
        temps = temps_a
    else:
        temps = temps_b
    res_lst, field_lst = [], []
    label_lst = []

    fig, ax = MakePlot().create()

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
        sweep_down_locs_neg_field = np.arange(mid_loc+1, min_loc)
        sweep_up_locs_neg_field = np.arange(min_loc, end_loc)

        ax.plot(dat[1][[sweep_up_locs_pos_field]],
                1 / dat[0][[sweep_up_locs_pos_field]] / flux_quantum,
                label=temps[c],color=colours[c],linewidth=2.0)
        ax.plot(dat[1][[sweep_down_locs_pos_field]],
                1 / dat[0][[sweep_down_locs_pos_field]] / flux_quantum,
                linestyle='dashed', label=temps[c+1],color=colours[c],linewidth=2.0)

        ax.plot(dat[1][[sweep_up_locs_neg_field]],
                1 / dat[0][[sweep_up_locs_neg_field]] / flux_quantum,
                label=temps[c],color=colours[c],linewidth=2.0)
        ax.plot(dat[1][[sweep_down_locs_neg_field]],
                1 / dat[0][[sweep_down_locs_neg_field]] / flux_quantum,
                linestyle='dashed', label=temps[c+1],color=colours[c],linewidth=2.0)

        c+=1

    ax.ticklabel_format(style='sci', axis='y') #, scilimits=(0, 0)

    handles, labels = plt.gca().get_legend_handles_labels()

    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, title='Temperature', loc='best')


    #sns.lineplot(x=r'Magnetic Field $(T)$', y=r'Resistance $(\Omega)$', data = df, hue=r'Temperature')
    plt.title('Conductance of VT64 at '+ currents[ind], fontsize=18)
    #ax.legend(title='Temperature',loc='right', fontsize=12)#,bbox_to_anchor=(1, 1), borderaxespad=0.)
    ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
    ax.set_ylabel(r'Conductance $(\frac{2e^2}{h})$', fontsize=14)
    plt.savefig('/Users/npopiel/Documents/MPhil/Data/step_data/Grouped by Current/VT64_conductance_'+ currents[ind]+'.png', dpi=600)
    #plt.show()
    plt.close()
