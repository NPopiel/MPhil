import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt

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

possible_currents = [1,5,10,20,50,100,200,500,1000,1500]

sample_lst = []

for ind, sample in enumerate(samples):

    temp_lst = []

    temps = temp_ranges[ind]

    for temp in temps:

        temp_path = main_path + sample + '/' + str(temp) +'/'

        current_lst = []

        for current in possible_currents:

            resistance, field = load_r_and_h(temp_path, current)

            z = np.polyfit(field, resistance, 1)

            current_lst.append(z[0])

        temp_lst.append(current_lst)

    sns.set_palette('viridis')

    array = np.array(temp_lst).T
    fig, ax = MakePlot().create()
    sns.heatmap(array,cmap='viridis')
    plt.xticks(np.arange(array.shape[1])+0.5,temp_labs[ind])
    plt.yticks(np.arange(array.shape[0])+0.5,possible_currents)
    plt.xlabel('Temperature (K)',fontsize=16)
    plt.ylabel('Current (mA)',fontsize=16)
    plt.title('Slope of RvB ('+sample+')',fontsize=22)
    plt.savefig(main_path+sample+'.png',dpi=200)
    plt.close()
    print(array.shape)

