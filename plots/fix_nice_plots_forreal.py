import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage.filters
from matplotlib.colors import DivergingNorm

main_path = '/Users/npopiel/Documents/MPhil/Data/'
save_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/Raw/'

samples = ['VT11','VT1', 'VT51', 'SBF25', 'VT26','VT49']
temps_1 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0)#,23.0)
temps_2 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0)
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

            #resistance, locs_2_drop = remove_noise(resistance, 5, eps=1.5)

            #field = [e for i, e in enumerate(field) if i not in locs_2_drop]

            resistance = scipy.ndimage.filters.median_filter(resistance, size=3)
            resistance_lst.append(resistance)
            field_curr.append(field)
            tempatu_lst.append([temp]*len(resistance))
            current_lst.append([current/1000]*len(resistance))


        temp_lst.append(flatten(resistance_lst))
        fields_lst.append(flatten(field_curr))
        temperoos.append(flatten(tempatu_lst))
        cureentoos.append(flatten(current_lst))


    df = pd.DataFrame({r'Resistance $(\Omega)$':flatten(temp_lst),
                       'Temperature (K)': flatten(temperoos),
                       'Current (mA)': flatten(cureentoos),
                       'Magnetic Field (T)': flatten(fields_lst)})

    groupers = df.groupby('Temperature (K)')

    for ind, key in enumerate(groupers.groups.keys()):
        df1 = df[df['Temperature (K)'] == key]

        fig, axs = MakePlot(figsize=(16, 9)).create()
        sns.scatterplot(x='Magnetic Field (T)', y=r'Resistance $(\Omega)$', hue='Current (mA)',data=df1, palette="bright", legend=True, s=80, alpha=0.7,
                                ax=axs)
        plt.title(r'Resistance by Magnetic Field $(T = ' + str(key) + r' K)$ for ' + sample, fontsize=22)
        axs.set_xlabel('Magnetic Field (T)',fontsize=16)
        axs.set_ylabel(r'Resistance $(\Omega)$',fontsize=16)
        axs.semilogy()
        plt.legend(title='Current (mA)', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.tight_layout()
        # ,labels=np.unique(df.current_sweep_ch2.values), frameon=True)
        name = save_path + sample + 'resistance_t_' + str(key) + '.pdf'
        #plt.show()
        plt.savefig(name, dpi=200)
        print('Done!', name)
        plt.close()


