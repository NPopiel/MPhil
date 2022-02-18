import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage.filters
import seaborn as sns
import matplotlib.colors

flux_quantum = 7.748091729 * 10 **-5

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/Step/step_data/'

filenames_all = ['down_neg_fit_1p75K800uA.csv',
                 'down_neg_fit_1p75K900uA.csv',
                 'down_neg_fit_2p0K800uA.csv',
                 'down_neg_fit_2p0K900uA.csv',
                 'down_neg_fit_2p5K800uA.csv',
                 'down_neg_fit_2p5K900uA.csv',
                 'down_neg_fit_2p25K800uA.csv',
                 'down_neg_fit_2p25K900uA.csv',
                 'down_neg_fit_2p75K800uA.csv',
                 'down_neg_fit_3p0K800uA.csv',
                 'down_pos_fit_1p75K800uA.csv',
                 'down_pos_fit_1p75K900uA.csv',
                 'down_pos_fit_2p0K800uA.csv',
                 'down_pos_fit_2p0K900uA.csv',
                 'down_pos_fit_2p5K800uA.csv',
                 'down_pos_fit_2p5K900uA.csv',
                 'down_pos_fit_2p25K800uA.csv',
                 'down_pos_fit_2p25K900uA.csv',
                 'down_pos_fit_2p75K800uA.csv',
                 'down_pos_fit_3p0K800uA.csv',
                 'up_neg_fit_1p75K800uA.csv',
                 'up_neg_fit_1p75K900uA.csv',
                 'up_neg_fit_2p0K800uA.csv',
                 'up_neg_fit_2p0K900uA.csv',
                 'up_neg_fit_2p5K800uA.csv',
                 'up_neg_fit_2p5K900uA.csv',
                 'up_neg_fit_2p25K800uA.csv',
                 'up_neg_fit_2p25K900uA.csv',
                 'up_neg_fit_2p75K800uA.csv',
                 'up_neg_fit_3p0K800uA.csv',
                 'up_pos_fit_1p75K800uA.csv',
                 'up_pos_fit_1p75K900uA.csv',
                 'up_pos_fit_2p0K700uA.csv',
                 'up_pos_fit_2p0K800uA.csv',
                 'up_pos_fit_2p0K900uA.csv',
                 'up_pos_fit_2p5K800uA.csv',
                 'up_pos_fit_2p5K900uA.csv',
                 'up_pos_fit_2p25K800uA.csv',
                 'up_pos_fit_2p25K900uA.csv',
                 'up_pos_fit_2p75K800uA.csv',
                 'up_pos_fit_3p0K800uA.csv']


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
curr_600 = ['VT64_1p75K600uA.csv',
            'VT64_2p0K600uA.csv',
            'VT64_2p25K600uA.csv',
            'VT64_2p5K600uA.csv']
curr_700 = [#'VT64_1p75K700uA.csv', # starts in middle of the step
            'VT64_2p0K700uA.csv',  # starts in middle of the step
            'VT64_2p25K700uA.csv']
curr_800 = ['VT64_1p75K800uA.csv',
            'VT64_2p0K800uA.csv',
            'VT64_2p25K800uA.csv',
            'VT64_2p5K800uA.csv',
            'VT64_2p75K800uA.csv',
            'VT64_3p0K800uA.csv']#,
            #'VT64_3p25K800uA.csv',
            #'VT64_3p5K800uA.csv',
            #'VT64_3p75K800uA.csv',
            #'VT64_4p0K800uA.csv']
curr_900 = ['VT64_1p75K900uA.csv',
            'VT64_2p0K900uA.csv',
            'VT64_2p25K900uA.csv',
            'VT64_2p5K900uA.csv',
            'VT64_2p75K900uA.csv',
            'VT64_3p0K900uA.csv']#, I think fit starts breaking here.. do by hand
            #'VT64_3p25K900uA.csv',
            #'VT64_3p5K900uA.csv',
            #'VT64_3p75K900uA.csv',
            #'VT64_4p0K900uA.csv']
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

temps_c = ['1.75 K +', '1.75 K -', '2.00 K +', '2.00 K -', '2.25 K +','2.25 K -',
           '2.50 K +', '2.50 K -']
temps_d = ['1.75 K +', '1.75 K -', '2.00 K +', '2.00 K -', '2.25 K +','2.25 K -']
currents = [r'500 $\mu A$', r'600 $\mu A$',r'700 $\mu A$',r'800 $\mu A$', r'900 $\mu A$', r'1000 $\mu A$', r'1100 $\mu A$', r'1200 $\mu A$', r'1500 $\mu A$']


#data_sets = [curr_500, curr_600, curr_700, curr_800, curr_900, curr_1000, curr_1100, curr_1200, curr_1500]

data_sets=[curr_800]

sample = 'VT64'

sweeps = ['Positive Field Up-Sweep',
          'Positive Field Down-Sweep',
          'Negative Field Up-Sweep',
          'Negative Field Down-Sweep']

sweeps_save = ['up_pos_fit_', 'down_pos_fit_', 'up_neg_fit_', 'down_neg_fit_']


for ind, curr_data_name in enumerate(data_sets):

    up_plus, up_minus, down_plus, down_minus = [], [], [], []
    field_up_plus, field_up_minus, field_down_plus, field_down_minus = [], [], [], []

    max_up_plus, max_up_minus, max_down_plus, max_down_minus = [],[],[],[]
    max_up_plus_field, max_up_minus_field, max_down_plus_field, max_down_minus_field = [], [], [], []


    for idx, current_temp_data_name in enumerate(curr_data_name):
        root_name = current_temp_data_name.split('_')[1]
        for ind, sweep in enumerate(sweeps_save):
          dat = np.loadtxt(main_path + 'fits/' + sweep + root_name, delimiter=',')
          if ind == 0:
            field_up_plus.append(dat[:,0])
            up_plus.append(dat[:,1])
            max_loc = np.argmax(dat[:,1])
            max_up_plus.append(dat[:,1][max_loc])
            max_up_plus_field.append((dat[:,0][max_loc]))
          if ind == 1:
            field_down_plus.append(dat[:,0])
            down_plus.append(dat[:,1])
            max_loc = np.argmax(dat[:,1])
            max_down_plus.append(dat[:,1][max_loc])
            max_down_plus_field.append((dat[:,0][max_loc]))
          if ind == 2:
            field_up_minus.append(dat[:,0])
            up_minus.append(dat[:,1])
            max_loc = np.argmax(dat[:,1])
            max_up_minus.append(dat[:,1][max_loc])
            max_up_minus_field.append((dat[:,0][max_loc]))
          if ind == 3:
            field_down_minus.append(dat[:,0])
            down_minus.append(dat[:,1])
            max_loc = np.argmax(dat[:,1])
            max_down_minus.append(dat[:,1][max_loc])
            max_down_minus_field.append((dat[:,0][max_loc]))


fig, ax = MakePlot(figsize=(16,9)).create()

temps = ['1.75 K', '2.00 K',  '2.25 K',
           '2.50 K',  '2.75 K',  '3.00 K']



line_up_plus = np.poly1d(np.polyfit(max_up_plus_field,max_up_plus,1))
line_down_plus = np.poly1d(np.polyfit(max_down_plus_field,max_down_plus,1))
line_up_minus = np.poly1d(np.polyfit(max_up_minus_field,max_up_minus,1))
line_down_minus = np.poly1d(np.polyfit(max_down_minus_field,max_down_minus,1))

linear_vector = np.arange(0,14,200)
linear_vector_2 = np.arange(-14,0,200)

for i in range(len(up_minus)):
    ax.plot(field_up_plus[i], up_plus[i], label=temps[i],
                 linewidth=2.0, color=plt.cm.jet(i/10))
    ax.plot(field_down_plus[i], down_plus[i], label=temps[i], linestyle='dashed',
                 linewidth=2.0, color=plt.cm.jet(i/10))
    ax.plot(field_up_minus[i], up_minus[i], label=temps[i],
                 linewidth=2.0, color=plt.cm.jet(i/10))
    ax.plot(field_down_minus[i], down_minus[i], label=temps[i], linestyle='dashed',
                 linewidth=2.0, color=plt.cm.jet(i/10))
#ax.plot(linear_vector, line_up_plus(linear_vector),c='k')
plt.plot(linear_vector, line_down_plus(linear_vector),c='k')
#ax.plot(linear_vector_2, line_up_minus(linear_vector_2),c='k')
plt.plot(linear_vector_2, line_down_minus(linear_vector_2),c='k')
ax.set_title(r'800 $\mu A$', fontsize=32,fontname='arial')
ax.axhline(0, -14, 16, color='k', alpha=0.75)
#plt.axhline(.25, -14, 16, color='k', alpha=0.75)
#plt.axhline(0.5, -14, 16, color='k', alpha=0.75)
#plt.axhline(1.0, -14, 16, color='k', alpha=0.75)
ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=22,fontname='arial')
ax.set_ylabel(r'$|\Delta G |$ $(\frac{2 e^2}{h})$', fontsize=22,fontname='arial')
ax.set_ylim((0, 1.3))
ax.set_xlim()
ax.set_ylim()
ax.set_xlim()
ax.set_ylim()
ax.minorticks_on()
ax.tick_params('both', which='major', direction='in', length=8, width=4/3,
                      bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=6, width=2,
                      bottom=True, top=True, left=True, right=True)
plt.setp(ax.get_yticklabels(), fontsize=18, fontname='arial')
plt.setp(ax.get_xticklabels(), fontsize=18, fontname='arial')

ax.ticklabel_format(style='sci', axis='y')  # , scilimits=(0, 0)
handles, labels = ax.get_legend_handles_labels()

labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]



plt.subplots_adjust(hspace=0.25)
legend = ax.legend(handles, labels, title='Temperature', loc='best',   # Position of legend
            framealpha=0,prop={"size": 16})
plt.setp(legend.get_title(), fontsize=18, fontname='arial')
plt.tight_layout()
#plt.show()
plt.savefig('/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/VT64_delta-G-800uA-draft1.png', bbox_inches='tight',dpi=400)