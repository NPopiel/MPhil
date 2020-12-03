import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.colors import DivergingNorm

main_path = '/Users/npopiel/Documents/MPhil/Data/'

samples = ['VT11', 'VT1', 'VT51', 'SBF25', 'VT26','VT49']#
temps_1 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0)#,23.0)
temps_2 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0)
temps_3 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0)#,10.0)
temp_ranges = [temps_1,
               temps_2,
               temps_3,
               temps_1,
               temps_2,
               temps_3]

temp_lab1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
temp_lab2 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
temp_lab3 = [2,3,4,5,6,7,8,9]
temp_labs = [temp_lab1,temp_lab2,temp_lab3,temp_lab1,temp_lab2,temp_lab3]

possible_currents = np.array([1,5,10,20,50,100,200,500,1000,1500])

sample_lst = []

dict = {}

for ind, sample in enumerate(samples):

    temp_lst = []

    temps = temp_ranges[ind]
    resistance_temp_lst, field_temp_lst, temp_names = [], [], []
    for temp in temps:

        temp_path = main_path + sample + '/' + str(temp) +'/'

        resistance_current_lst, field_current_lst, current_names  = [], [], []

        for current in possible_currents:

            resistance, field = load_r_and_h(temp_path, current)

            resistance_current_lst.append(resistance)
            field_current_lst.append(field)
            current_names.append([str(current/1000)+ ' mA']*len(field))

        resistance_temp_lst.append(flatten(resistance_current_lst))
        field_temp_lst.append(flatten(field_current_lst))
        temp_names.append([str(temp)+' K'] * len(flatten(field_current_lst)))

    dict['Resistance'] = flatten(re)

    array = np.array(temp_lst).T[:,:5]

    rdgn = sns.color_palette("viridis", as_cmap=True)
    #divnorm = DivergingNorm(vmin=array.min(), vcenter=1, vmax=array.max()) #shiftedColorMap(get_cmap('RdBu'),start=np.min(array),midpoint=1,stop=np.max(array))


    fig, ax = MakePlot().create()
    sns.heatmap(array[:,:5],cmap=rdgn)#, norm=divnorm)
    plt.xticks(np.arange(array[:,:5].shape[1])+0.5,temp_labs[ind][:5])
    plt.yticks(np.arange(array[:,:5].shape[0])+0.5,possible_currents/1000)
    plt.xlabel('Temperature (K)',fontsize=16)
    plt.ylabel('Current (mA)',fontsize=16)
    #plt.title(r'$\frac{R_{B=14}}{R_{B=0}}$ ('+sample+')',fontsize=22,usetex=True)
    plt.title(r'Max Field Resistance over Min Field (' + sample + ')', fontsize=22)
    plt.savefig(main_path+sample+'_ratio_short2.png',dpi=200)
    plt.close()
    print(array.shape)


