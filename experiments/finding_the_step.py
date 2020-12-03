import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import DivergingNorm

main_path = '/Users/npopiel/Documents/MPhil/Data/'

samples = ['VT49']#
temps_3 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0)#,10.0)
temp_ranges = [temps_3]


possible_currents1 = np.array([1,5,10,20,50,100,200,500,1000,1500])

sample_lst = []

load_path = main_path + 'VT49/2.0/'

resistances, fields, currents = [], [], []

for current in possible_currents1:

    resistance, field = load_r_and_h(load_path, current)
    resistances.append(resistance)
    fields.append(field)
    currents.append([current]*len(resistance))

df = pd.DataFrame({r'Resistance $(\Omega)$': flatten(resistances),
                   'Current (mA)': flatten(currents),
                   'Magnetic Field (T)': flatten(fields)})

groupers = df.groupby('Current (mA)')

fig, ax = MakePlot(nrows=2, ncols=5).create()
axs = [ax[0, 0], ax[0, 1], ax[0, 2], ax[0, 3], ax[0, 4], ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3], ax[1, 4]]
for ind, key in enumerate(groupers.groups.keys()):

    sns.scatterplot(x='Magnetic Field (T)', y=r'Resistance $(\Omega)$',
                        data=df[df['Current (mA)'] == key], legend=False,
                        ax=axs[ind])  # style='Temperature (K)',

    if ind < 5:
        axs[ind].set_xlabel('')
        if ind != 0:
            axs[ind].set_ylabel('')
    if ind >= 5:
        if ind != 5:
            axs[ind].set_ylabel('')
    axs[ind].set_title(str(key) + 'mA', fontsize=14)


plt.suptitle('2K Resistance VT49 ', fontsize=22)
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