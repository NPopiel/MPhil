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

            min_field_loc = np.where(field == np.amin(field))[0][0]
            max_field_loc = np.where(field == np.amax(field))[0][0]

            if min_field_loc > max_field_loc:
                range1, range2 = max_field_loc, min_field_loc
            else:
                range2, range1 = max_field_loc, min_field_loc

            clipped_arr = resistance[range1:range2]

            ratio = clipped_arr/resistance[min_field_loc]

            resistance_lst.append(ratio)
            field_curr.append(field[range1:range2])
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

    fig, ax = MakePlot(nrows=2,ncols=5).create()
    axs = [ax[0,0],ax[0,1],ax[0,2],ax[0,3],ax[0,4],ax[1,0],ax[1,1],ax[1,2],ax[1,3],ax[1,4]]
    for ind, key in enumerate(groupers.groups.keys()):


        if ind!=7:
            sns.scatterplot(x='Magnetic Field (T)', y='Magnetoresistive Ratio', hue='Temperature (K)', data=df[df['Current (mA)']==key], palette="bright", legend=False, ax=axs[ind]) #style='Temperature (K)',
        if ind == 7:
            sns.scatterplot(x='Magnetic Field (T)', y='Magnetoresistive Ratio', hue='Temperature (K)',
                            data=df[df['Current (mA)'] == key], palette="bright", legend=True,
                            ax=axs[ind])
            axs[ind].legend(loc='center', title='Temperature (K)', bbox_to_anchor=(0.5, -0.25), shadow=False, ncol=8, frameon=False)

        if ind < 5:
            axs[ind].set_xlabel('')
            if ind != 0:
                axs[ind].set_ylabel('')
        if ind >= 5:
            if ind != 5:
                axs[ind].set_ylabel('')
        axs[ind].set_title(str(key)+ 'mA', fontsize=14)


    # legend = ax[1,4].legend()
    # ax[1,4].get_legend().remove()

    fig.suptitle('Resistance (T = ' + str(temp) + ' K)', fontsize=22)
    # plt.figlegend(frameon=True,
    #               loc='upper right',
    #               title='Temperature (K)' )  # ,labels=np.unique(df.current_sweep_ch2.values), frameon=True)

    plt.suptitle('Low Temperature Magnetoresistive Ratio for '+ sample,fontsize=22)
    fig.tight_layout(pad=4.0)
    plt.show()

    #
    # for curr, inds in groupers.groups.items():
    #
    #     df_curr = df[df['Current (mA)'] == curr]
    #
    #     fig, ax = MakePlot().create()
    #     sns.scatterplot(x='Magnetic Field (T)', y='Magnetoresistive Ratio', hue='Temperature (K)', data=df_curr, palette="bright", legend=True) #style='Temperature (K)',
    #     plt.title('Low Temperature Magnetoresistive Ratio ('+str(curr)+'(mA) ) for '+ sample, fontsize=22)
    #     ax.set_xlabel('Magnetic Field (T)', fontsize=16)
    #     ax.set_ylabel('Magnetoresistive Ratio', fontsize=16)
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # #plot here