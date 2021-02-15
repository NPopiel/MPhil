import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import DivergingNorm

main_path = '/Users/npopiel/Documents/MPhil/Data/'

samples = ['VT11', 'VT1', 'VT51', 'SBF25', 'VT26','VT49']#
temps_1 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0)#,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0)#,23.0)
temps_2 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0)#,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0)
temps_3 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0)#,10.0)
temp_ranges = [temps_1,
               temps_2,
               temps_3,
               temps_1,
               temps_2,
               temps_3]


possible_currents1 = np.array([1,5,10,20,50,100,200,500,1000,1500])
possible_currents2 = np.array([1500,1000,500,200,100,50,20,10,5,1])



integer_fields = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]


sample_lst = []

for ind, sample in enumerate(samples):

    temperoos, cureentoos = [], []

    temp_lst, fields_lst = [], []

    temps = temp_ranges[ind]

    for temp in temps:

        temp_path = main_path + sample + '/' + str(temp) +'/'

        resistance_lst, current_lst, field_curr, tempatu_lst = [], [], [], []

        for current in possible_currents1:

            resistance, field = load_r_and_h(temp_path, current)

            #nans, x = nan_helper(resistance)
            #resistance[nans] = np.interp(x(nans), x(~nans), resistance[~nans])

            min_field_loc = 0
            max_field_loc = np.argmax(field)

            if min_field_loc > max_field_loc:
                range1, range2 = max_field_loc, min_field_loc
            else:
                range2, range1 = max_field_loc, min_field_loc

            clipped_arr = resistance[range1:range2]

            field_clipped = field[range1:range2]

            field_ind_lst = []

            for field_val in integer_fields:
                close_val ,idx = find_nearest(field_clipped,field_val,return_idx=True)
                field_ind_lst.append(idx)

            ratio = clipped_arr[field_ind_lst]/resistance[min_field_loc]

            resistance_lst.append(ratio)
            field_curr.append(integer_fields)
            tempatu_lst.append([temp]*len(ratio))
            current_lst.append([current/1000]*len(ratio))


        temp_lst.append(flatten(resistance_lst))
        fields_lst.append(flatten(field_curr))
        temperoos.append(flatten(tempatu_lst))
        cureentoos.append(flatten(current_lst))


    df = pd.DataFrame({'Magnetoresistive Ratio':flatten(temp_lst),
                       'Temperature (K)': flatten(temperoos),
                       'Current (mA)': flatten(cureentoos),
                       'Magnetic Field (T)': flatten(fields_lst)})

    groupers = df.groupby('Current (mA)')


    for ind, key in enumerate(groupers.groups.keys()):

        df1 = df[df['Current (mA)']==key]

        group2 = df1.groupby('Magnetic Field (T)')

        fig, ax = MakePlot(nrows=3, ncols=5).create()
        axs = [ax[0, 0], ax[0, 1], ax[0, 2], ax[0, 3], ax[0, 4],
               ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3], ax[1, 4],
               ax[2, 0], ax[2, 1], ax[2, 2], ax[2, 3], ax[2, 4]]

        for ind2, key2 in enumerate(group2.groups.keys()):
            df2 = df1[df1['Magnetic Field (T)']==key2]

            if ind2!=12:
                sns.scatterplot(x='Temperature (K)', y='Magnetoresistive Ratio', data=df2, palette="bright", legend=False, ax=axs[ind2]) #style='Temperature (K)',
            if ind2 == 12:
                sns.scatterplot(x='Temperature (K)', y='Magnetoresistive Ratio',
                                data=df2, palette="bright", legend=True,
                                ax=axs[ind2])
                axs[ind2].legend(loc='center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=15, frameon=False)

            if not (ind2 == 0 or ind2 == 5 or ind2 == 10):
                axs[ind2].set_ylabel('')

            if ind2 < 10:
                axs[ind2].set_xlabel('')

            axs[ind2].set_title(str(key2)+ 'T', fontsize=14)


        plt.suptitle('Low Temperature Magnetoresistive Ratio for '+ sample + 'Current (' + str(key) + ' mA)',fontsize=22)
        fig.tight_layout(pad=2.0)
        #plt.show()
        plt.savefig('/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/MR_by_Field/'+ sample +'mr_field_curr_'+str(key)+'.png',dpi=300)
        plt.close()