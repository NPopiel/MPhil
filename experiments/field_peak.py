import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import DivergingNorm

main_path = '/Users/npopiel/Documents/MPhil/Data/'

samples = ['VT11',
           'VT1', 'VT51', 'SBF25', 'VT26','VT49']#
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


            max_resistance_loc = np.argmax(resistance)

            fig, ax = MakePlot().create()
            plt.axvline(field[max_resistance_loc])
            plt.scatter(x=field, y=resistance)
            plt.title('Temp = '+str(temp)+ 'current' + str(current) + 'sample:' + sample)
            plt.show()

            field_curr.append(field[max_resistance_loc])
            tempatu_lst.append(temp)
            current_lst.append(current/1000)


        fields_lst.append(field_curr)
        temperoos.append(tempatu_lst)
        cureentoos.append(current_lst)


    df = pd.DataFrame({'Temperature (K)': flatten(temperoos),
                       'Current (mA)': flatten(cureentoos),
                       r'$\mu_o H_{Peak} (T)$': flatten(fields_lst)})

    groupers = df.groupby('Temperature (K)')

    for ind, key in enumerate(groupers.groups.keys()):

        fig, ax = MakePlot().create()

        # g = sns.FacetGrid(df, col="time", row="sex")
        # g.map(sns.scatterplot, "total_bill", "tip")

        sns.scatterplot(x='Current (mA)', y=r'$\mu_o H_{Peak} (T)$',
                        data=df[df['Temperature (K)'] == key], palette="bright") #style='Temperature (K)', column='Temperature (K)'


        plt.suptitle('Low Temperature Peak Field for '+ sample + '(T = '+ str(key) + ' K)',fontsize=22)
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