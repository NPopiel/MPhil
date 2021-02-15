import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage.filters
import seaborn as sns

main_path = '/Users/npopiel/Documents/MPhil/Data/step_data/'


files = ['VT64_NYE1p75K600uA.csv',
'VT64_1p75K500uA.csv',
'VT64_1p75K800uA.csv',
'VT64_1p75K900uA.csv',
'VT64_1p75K1000uA.csv',
'VT64_1p75K1100uA.csv',
'VT64_1p75K1200uA.csv',
'VT64_1p75K1500uA.csv',
'VT64_2p0K500uA.csv',
'VT64_2p0K800uA.csv',
'VT64_2p0K900uA.csv',
'VT64_2p0K1000uA.csv',
'VT64_2p0K1100uA.csv',
'VT64_2p0K1200uA.csv',
'VT64_2p0K1500uA.csv',
'VT64_2p5K500uA.csv',
'VT64_2p5K800uA.csv',
'VT64_2p5K900uA.csv',
'VT64_2p5K1000uA.csv',
'VT64_2p5K1100uA.csv',
'VT64_2p5K1200uA.csv',
'VT64_2p5K1500uA.csv',
'VT64_2p25K500uA.csv',
'VT64_2p25K800uA.csv',
'VT64_2p25K900uA.csv',
'VT64_2p25K1000uA.csv',
'VT64_2p25K1100uA.csv',
'VT64_2p25K1200uA.csv',
'VT64_2p25K1500uA.csv',
'VT64_2p75K500uA.csv',
'VT64_2p75K800uA.csv',
'VT64_2p75K900uA.csv',
'VT64_2p75K1000uA.csv',
'VT64_2p75K1100uA.csv',
'VT64_2p75K1200uA.csv',
'VT64_2p75K1500uA.csv',
'VT64_3p0K500uA.csv',
'VT64_3p0K800uA.csv',
'VT64_3p0K900uA.csv',
'VT64_3p0K1000uA.csv',
'VT64_3p0K1100uA.csv',
'VT64_3p0K1200uA.csv',
'VT64_3p0K1500uA.csv',
'VT64_3p5K500uA.csv',
'VT64_3p5K800uA.csv',
'VT64_3p5K900uA.csv',
'VT64_3p5K1000uA.csv',
'VT64_3p5K1100uA.csv',
'VT64_3p5K1200uA.csv',
'VT64_3p5K1500uA.csv',
'VT64_3p25K500uA.csv',
'VT64_3p25K800uA.csv',
'VT64_3p25K900uA.csv',
'VT64_3p25K1000uA.csv',
'VT64_3p25K1100uA.csv',
'VT64_3p25K1200uA.csv',
'VT64_3p25K1500uA.csv',
'VT64_3p75K500uA.csv',
'VT64_3p75K800uA.csv',
'VT64_3p75K900uA.csv',
'VT64_3p75K1000uA.csv',
'VT64_3p75K1100uA.csv',
'VT64_3p75K1200uA.csv',
'VT64_3p75K1500uA.csv',
'VT64_4p0K500uA.csv',
'VT64_4p0K800uA.csv',
'VT64_4p0K900uA.csv',
'VT64_4p0K1000uA.csv',
'VT64_4p0K1100uA.csv',
'VT64_4p0K1200uA.csv',
'VT64_NYE1p75K700uA.csv',
'VT64_NYE2p0K600uA.csv',
'VT64_NYE2p0K700uA.csv',
'VT64_NYE2p5K600uA.csv',
'VT64_NYE2p25K600uA.csv',
'VT64_NYE2p25K700uA.csv',
'VT64_NYE11p8K600uA.csv']

one_p_75 = ['VT64_1p75K500uA.csv',
            'VT64_NYE1p75K600uA.csv',
            'VT64_NYE1p75K700uA.csv',
            'VT64_1p75K800uA.csv',
            'VT64_1p75K900uA.csv',
            'VT64_1p75K1000uA.csv',
            'VT64_1p75K1100uA.csv',
            'VT64_1p75K1200uA.csv',
            'VT64_1p75K1500uA.csv']
two_p_0 = ['VT64_2p0K500uA.csv',
            'VT64_NYE2p0K600uA.csv',
            'VT64_NYE2p0K700uA.csv',
            'VT64_2p0K800uA.csv',
            'VT64_2p0K900uA.csv',
            'VT64_2p0K1000uA.csv',
            'VT64_2p0K1100uA.csv',
            'VT64_2p0K1200uA.csv',
            'VT64_2p0K1500uA.csv']
two_p_25 = ['VT64_2p25K500uA.csv',
            'VT64_NYE2p25K600uA.csv',
            'VT64_NYE2p25K700uA.csv',
            'VT64_2p25K800uA.csv',
            'VT64_2p25K900uA.csv',
            'VT64_2p25K1000uA.csv',
            'VT64_2p25K1100uA.csv',
            'VT64_2p25K1200uA.csv',
            'VT64_2p25K1500uA.csv']
two_p_5 = ['VT64_2p5K500uA.csv',
            #'VT64_NYE2p5K600uA.csv',
            #'VT64_NYE2p5K700uA.csv',
            'VT64_2p5K800uA.csv',
            'VT64_2p5K900uA.csv',
            'VT64_2p5K1000uA.csv',
            'VT64_2p5K1100uA.csv',
            'VT64_2p5K1200uA.csv',
            'VT64_2p5K1500uA.csv']
two_p_75 = ['VT64_2p75K500uA.csv',
            #'VT64_NYE2p75K600uA.csv',
            #'VT64_NYE2p75K700uA.csv',
            'VT64_2p75K800uA.csv',
            'VT64_2p75K900uA.csv',
            'VT64_2p75K1000uA.csv',
            'VT64_2p75K1100uA.csv',
            'VT64_2p75K1200uA.csv',
            'VT64_2p75K1500uA.csv']
three_p_0 = ['VT64_3p0K500uA.csv',
            #'VT64_NYE3p0K600uA.csv',
            #'VT64_NYE3p0K700uA.csv',
            'VT64_3p0K800uA.csv',
            'VT64_3p0K900uA.csv',
            'VT64_3p0K1000uA.csv',
            'VT64_3p0K1100uA.csv',
            'VT64_3p0K1200uA.csv',
            'VT64_3p0K1500uA.csv']
three_p_25 = ['VT64_3p25K500uA.csv',
            #'VT64_NYE3p25K600uA.csv',
            #'VT64_NYE3p25K700uA.csv',
            'VT64_3p25K800uA.csv',
            'VT64_3p25K900uA.csv',
            'VT64_3p25K1000uA.csv',
            'VT64_3p25K1100uA.csv',
            'VT64_3p25K1200uA.csv',
            'VT64_3p25K1500uA.csv']
three_p_5 = ['VT64_3p5K500uA.csv',
            #'VT64_NYE3p5K600uA.csv',
            #'VT64_NYE3p5K700uA.csv',
            'VT64_3p5K800uA.csv',
            'VT64_3p5K900uA.csv',
            'VT64_3p5K1000uA.csv',
            'VT64_3p5K1100uA.csv',
            'VT64_3p5K1200uA.csv',
            'VT64_3p5K1500uA.csv']
three_p_75 = ['VT64_3p75K500uA.csv',
            #'VT64_NYE3p75K600uA.csv',
            #'VT64_NYE3p75K700uA.csv',
            'VT64_3p75K800uA.csv',
            'VT64_3p75K900uA.csv',
            'VT64_3p75K1000uA.csv',
            'VT64_3p75K1100uA.csv',
            'VT64_3p75K1200uA.csv',
            'VT64_3p75K1500uA.csv']
four_p_0 = ['VT64_4p0K500uA.csv',
            #'VT64_NYE4p0K600uA.csv',
            #'VT64_NYE4p0K700uA.csv',
            'VT64_4p0K800uA.csv',
            'VT64_4p0K900uA.csv',
            'VT64_4p0K1000uA.csv',
            'VT64_4p0K1100uA.csv',
            'VT64_4p0K1200uA.csv']
            #'VT64_4p0K1500uA.csv']

temps = ['1.75 K', '2.00 K', '2.25 K', '2.50 K', '2.75 K', '3.00 K', '3.25 K', '3.5 K', '3.75 K', '4.00 K']
currents_a = [r'500 $\mu A$',r'600 $\mu A$',r'700 $\mu A$',r'800 $\mu A$',r'900 $\mu A$',r'1000 $\mu A$',r'1100 $\mu A$',r'1200 $\mu A$',r'1500 $\mu A$']
currents_b = [r'500 $\mu A$',r'800 $\mu A$',r'900 $\mu A$',r'1000 $\mu A$',r'1100 $\mu A$',r'1200 $\mu A$',r'1500 $\mu A$']


data_sets = [one_p_75,
             two_p_0,
             two_p_25,
             two_p_5,
             two_p_75,
             three_p_0,
             three_p_25,
             three_p_5,
             three_p_75,
             four_p_0]

sample = 'VT64'


for ind, temp_data_name in enumerate(data_sets):
    if ind > 2:
        current = currents_b
    else:
        currents = currents_a
    res_lst, field_lst = [], []
    label_lst = []
    for idx, current_temp_data_name in enumerate(temp_data_name):
        dat = load_matrix(main_path+current_temp_data_name)
        res_lst.append(dat[0])
        field_lst.append(dat[1])
        label_lst.append([currents[idx]]*len(dat[0]))

    df = pd.DataFrame({r'Magnetic Field $(T)$': flatten(field_lst),
                       r'Resistance $(\Omega)$': flatten(res_lst),
                       r'Current': flatten(label_lst)})

    fig, ax = MakePlot().create()
    sns.lineplot(x=r'Magnetic Field $(T)$', y=r'Resistance $(\Omega)$', data = df, hue=r'Current')
    plt.title('VT64 at '+ temps[ind], fontsize=18)
    ax.legend(title='Current',loc='right', fontsize=12)#,bbox_to_anchor=(1, 1), borderaxespad=0.)
    ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
    ax.set_ylabel(r'Resistance $(\Omega)$', fontsize=14)
    plt.savefig('/Users/npopiel/Documents/MPhil/Data/step_data/Grouped by Temp/VT64_'+ temps[ind]+'.png', dpi=600)
    plt.close()
