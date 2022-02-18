import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import median_filter
import seaborn as sns

main_path = '/Users/npopiel/Documents/MPhil/Data/step_data/'

df = pd.read_csv(main_path+'steplocs.csv')

# groupe data by temp to loop over and extract current changes
groupers = df.groupby('temp')



for constant_temp, inds in groupers.groups.items():

    df_T = df[df.temp == constant_temp]

    # if you want to verify it is done correctly uncomment the line with plot=True

    groupers2 = df_T.groupby('current')

    for current, indxs in groupers2.groups.items():
        # select data only with constant current, and constant temperature

        subsection = df_T[df_T['current'] == current]

        # get resistance and field values (after converting to tesla)

        dG = subsection['max_delta_g']
        field = subsection['field']

        # filename for saving, converting to an array with resistance in first col, field in second then saving!

sns.set_context('paper')

fig, ax = MakePlot().create()

#

sns.scatterplot(x='temp',y='max_delta_g',hue='current',style='sweep',data=df)
plt.show()


'''
sns.set_context('paper')

fig, ax = MakePlot().create()

clrs = sns.color_palette('husl', n_colors=10)

line_up_plus = np.poly1d(np.polyfit(max_up_plus_field, max_up_plus, 1))
line_down_plus = np.poly1d(np.polyfit(max_down_plus_field, max_down_plus, 1))
line_up_minus = np.poly1d(np.polyfit(max_up_minus_field, max_up_minus, 1))
line_down_minus = np.poly1d(np.polyfit(max_down_minus_field, max_down_minus, 1))

for i in range(len(up_minus)):
    plt.plot(vec_up_plus[i], up_plus[i], label=temps[i],
             color=clrs[i], linewidth=2.0)
    plt.plot(vec_down_plus[i], down_plus[i], label=temps[i], linestyle='dashed',
             color=clrs[i], linewidth=2.0)
    plt.plot(vec_up_minus[i], up_minus[i], label=temps[i],
             color=clrs[i], linewidth=2.0)
    plt.plot(vec_down_minus[i], down_minus[i], label=temps[i], linestyle='dashed',
             color=clrs[i], linewidth=2.0)
plt.plot(linear_vector, line_up_plus(linear_vector), c='k')
plt.plot(linear_vector, line_down_plus(linear_vector), c='k')
plt.plot(linear_vector_2, line_up_minus(linear_vector_2), c='k')
plt.plot(linear_vector_2, line_down_minus(linear_vector_2), c='k')
plt.title(r'Difference in Conductance (900 $\mu A$)', fontsize=18)
plt.axhline(0, -14, 16, color='k', alpha=0.75)
# plt.axhline(.25, -14, 16, color='k', alpha=0.75)
# plt.axhline(0.5, -14, 16, color='k', alpha=0.75)
# plt.axhline(1.0, -14, 16, color='k', alpha=0.75)
ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
ax.set_ylabel(r'$\delta G $', fontsize=14)
ax.set_ylim((0, 1.3))
ax.set_xlim()
ax.set_ylim()
ax.minorticks_on()
ax.tick_params('both', which='both', direction='in',
               bottom=True, top=True, left=True, right=True)

ax.ticklabel_format(style='sci', axis='y')  # , scilimits=(0, 0)

handles, labels = plt.gca().get_legend_handles_labels()

labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, title='Temperature', loc='best')

plt.show()
'''