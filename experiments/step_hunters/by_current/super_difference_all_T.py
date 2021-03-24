import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage.filters
import seaborn as sns

main_path = '/Users/npopiel/Documents/MPhil/Data/step_data/'

conductance_quantum = 7.748091729 * 10 ** -5

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
curr_800 = ['VT64_1p75K800uA.csv',
            'VT64_2p0K800uA.csv',
            'VT64_2p25K800uA.csv',
            'VT64_2p5K800uA.csv',
            'VT64_2p75K800uA.csv',
            'VT64_3p0K800uA.csv']#,
            #'VT64_3p25K800uA.csv']#,
            #'VT64_3p5K800uA.csv']  # ,
# 'VT64_3p75K800uA.csv',
# 'VT64_4p0K800uA.csv']
curr_900 = ['VT64_1p75K900uA.csv',
            'VT64_2p0K900uA.csv',
            'VT64_2p25K900uA.csv',
            'VT64_2p5K900uA.csv',
            'VT64_2p75K900uA.csv',
            'VT64_3p0K900uA.csv']
            # 'VT64_3p25K900uA.csv',
            # 'VT64_3p5K900uA.csv',
            # 'VT64_3p75K900uA.csv',
            # 'VT64_4p0K900uA.csv']
curr_1000 = ['VT64_1p75K1000uA.csv',
             'VT64_2p0K1000uA.csv',
             'VT64_2p25K1000uA.csv',
             'VT64_2p5K1000uA.csv',
             'VT64_2p75K1000uA.csv',
             'VT64_3p0K1000uA.csv']#,
             # 'VT64_3p25K1000uA.csv',
             # 'VT64_3p5K1000uA.csv',
             # 'VT64_3p75K1000uA.csv',
             # 'VT64_4p0K1000uA.csv']
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

temps_a = ['1.75 K', '2.00 K', '2.25 K', '2.50 K', '2.75 K', '3.00 K', '3.25 K']#, '3.5 K']  # , '3.75 K', '4.00 K']
temps_b = ['1.75 K', '2.00 K', '2.25 K', '2.50 K', '2.75 K', '3.00 K']#, '3.25 K']#, '3.5 K']  # , '3.75 K']
currents = [r'500 $\mu A$', r'800 $\mu A$', r'900 $\mu A$', r'1000 $\mu A$', r'1100 $\mu A$', r'1200 $\mu A$',
            r'1500 $\mu A$']

data_sets = [curr_800]  # , curr_900, curr_1000, curr_1100, curr_1200, curr_1500] #curr_500,

sample = 'VT64'

sweeps = ['Positive Field Up-Sweep',
          'Negative Field Up-Sweep',
          'Positive Field Down-Sweep',
          'Negative Field Down-Sweep']


def fit_wo_step(resistance, field, conductance, window_length=5, polyorder=3, deg=3):
    resistance_smooth = scipy.signal.savgol_filter(x=resistance, window_length=window_length, polyorder=polyorder)

    first_diff = np.diff(resistance_smooth)

    step_locs = np.arange(np.argmax(first_diff) - 1, np.argmin(first_diff) + 1)

    region_to_fit = np.setdiff1d(np.arange(resistance.shape[0]), step_locs)

    fitted_line = np.polyfit(x=field[region_to_fit], y=conductance[region_to_fit], deg=deg)

    return np.poly1d(fitted_line)


for ind, curr_data_name in enumerate(data_sets):
    temps = temps_b
    #
    # if ind <=5:
    #     temps = temps_a
    # else:
    #     temps = temps_b
    res_lst, field_lst = [], []
    label_lst = []

    up_plus, up_minus, down_plus, down_minus = [], [], [], []
    vec_up_plus, vec_up_minus, vec_down_plus, vec_down_minus = [], [], [], []

    for idx, current_temp_data_name in enumerate(curr_data_name):
        dat = load_matrix(main_path + current_temp_data_name)
        res_lst.append(dat[0])
        field_lst.append(dat[1])
        #label_lst.append([temps[idx]] * len(dat[0]))

        resistance = dat.T[0]
        field = dat.T[1]

        start_loc_up_pos = np.where(field > 1.2)[0][0]  # 0
        max_loc = np.argmax(field)
        min_loc = np.argmin(field)
        mid_loc = (min_loc - max_loc) / 2
        mid_loc = max_loc + int(mid_loc.round(0))
        end_loc = field.shape[0]

        sweep_up_locs_pos_field = np.arange(start_loc_up_pos, max_loc)
        sweep_down_locs_pos_field = np.arange(max_loc, mid_loc - 4)
        sweep_down_locs_neg_field = np.arange(mid_loc + 4, min_loc)
        sweep_up_locs_neg_field = np.arange(min_loc, end_loc - 4)

        resistance_up_pos = resistance[sweep_up_locs_pos_field]
        field_up_pos = field[sweep_up_locs_pos_field]
        conductance_up_pos = 1 / resistance_up_pos / conductance_quantum

        ffit_up_pos = fit_wo_step(resistance_up_pos, field_up_pos, conductance_up_pos)

        linear_vector = np.linspace(0, 14, 100)

        # Test Method

        fig, ax = MakePlot().create()

        first_diff = np.diff(resistance)
        resistance_smooth = scipy.signal.savgol_filter(x=resistance, window_length=5, polyorder=3)

        ax.plot(field[sweep_up_locs_pos_field[:-1]],first_diff[sweep_up_locs_pos_field[:-1]], label=temps[idx])
        ax.axhline(np.mean(first_diff[sweep_up_locs_pos_field[:-1]]))
        #ax.axvline(field[np.argmin(first_diff)+1])
        #ax.axvline(field[np.argmax(first_diff)-1])
        ax.plot(field[
        sweep_up_locs_pos_field], resistance_smooth[sweep_up_locs_pos_field], label=temps[idx])


        # Up-Sweep Positive
        '''
        fig, ax = MakePlot().create()
        plt.title(r'800 $\mu A$ at ' + temps[idx] + ' Up-Sweep +', fontsize=18)
        ax.plot(field_up_pos, conductance_up_pos, label=temps[idx])
        ax.plot(linear_vector, ffit_up_pos(linear_vector), label='Fit w/o Step')
        plt.legend(title='Line', loc='best')
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize = 14)
        ax.set_ylabel(r'Conductance $(\frac{2e^2}{h})$', fontsize=14)
        plt.savefig(main_path+'fit.pdf')
        '''
        diff_up_pos = ffit_up_pos(field_up_pos) - conductance_up_pos
        up_plus.append(diff_up_pos)
        vec_up_plus.append(field_up_pos)

        '''

        fig, ax = MakePlot().create()
        plt.title(r'Difference in Conductance (800 $\mu A$ @ '+ temps[idx] + ') Up-Sweep +', fontsize=18)
        plt.plot(field_up_pos, diff_up_pos)
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
        ax.set_ylabel(r'$\delta G $', fontsize=14)
        ax.set_ylim((-0.5, 1.4))

        plt.show()
        '''
        # Down Sweep Positive

        resistance_down_pos = resistance[sweep_down_locs_pos_field]
        field_down_pos = field[sweep_down_locs_pos_field]
        conductance_down_pos = 1 / resistance_down_pos / conductance_quantum

        ffit_down_pos = fit_wo_step(resistance_down_pos, field_down_pos, conductance_down_pos)

        linear_vector = np.linspace(0, 14, 100)

        '''
        fig, ax = MakePlot().create()
        # ax.plot(field[sweep_up_locs_pos_field[:-1]],first_diff[sweep_up_locs_pos_field[:-1]], label=temps[idx])
        # ax.axhline(np.mean(first_diff[sweep_up_locs_pos_field[:-1]]))
        # ax.axvline(field[np.argmin(first_diff)+1])
        # ax.axvline(field[np.argmax(first_diff)-1])
        plt.title(r'800 $\mu A$ at ' + temps[idx] + ' Down-Sweep +', fontsize=18)
        ax.plot(field_down_pos, conductance_down_pos, label=temps[idx])
        ax.plot(linear_vector, ffit_down_pos(linear_vector), label='Fit w/o Step')
        plt.legend(title='Line', loc='best')
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
        ax.set_ylabel(r'Conductance $(\frac{2e^2}{h})$', fontsize=14)
        # ax.plot(field[
        # sweep_up_locs_pos_field], resistance_smooth[sweep_up_locs_pos_field], label=temps[idx])
        plt.show()
        '''

        diff_down_pos = ffit_down_pos(field_down_pos) - conductance_down_pos
        down_plus.append(diff_down_pos)
        vec_down_plus.append(field_down_pos)
        '''

        fig, ax = MakePlot().create()
        plt.title(r'Difference in Conductance (800 $\mu A$ @ ' + temps[idx] + ') Down-Sweep +', fontsize=18)
        plt.plot(field_down_pos, diff_down_pos)
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
        ax.set_ylabel(r'$\delta G $', fontsize=14)
        ax.set_ylim((-0.5, 1.4))

        plt.show()
        '''

        # Up-Sweep Negative

        resistance_up_neg = resistance[sweep_up_locs_neg_field]
        field_up_neg = field[sweep_up_locs_neg_field]
        conductance_up_neg = 1 / resistance_up_neg / conductance_quantum

        ffit_up_neg = fit_wo_step(resistance_up_neg, field_up_neg, conductance_up_neg)

        linear_vector_2 = np.linspace(-14, 0, 100)

        '''
        fig, ax = MakePlot().create()
        #ax.plot(field[sweep_up_locs_pos_field[:-1]],first_diff[sweep_up_locs_pos_field[:-1]], label=temps[idx])
        #ax.axhline(np.mean(first_diff[sweep_up_locs_pos_field[:-1]]))
        #ax.axvline(field[np.argmin(first_diff)+1])
        #ax.axvline(field[np.argmax(first_diff)-1])
        plt.title(r'800 $\mu A$ at ' + temps[idx]+ ' Up-Sweep -', fontsize=18)
        ax.plot(field_up_neg, conductance_up_neg, label=temps[idx])
        ax.plot(linear_vector_2, ffit_up_neg(linear_vector_2), label='Fit w/o Step')
        plt.legend(title='Line', loc='best')
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize = 14)
        ax.set_ylabel(r'Conductance $(\frac{2e^2}{h})$', fontsize=14)
        #ax.plot(field[
        # sweep_up_locs_pos_field], resistance_smooth[sweep_up_locs_pos_field], label=temps[idx])
        plt.show()
        '''

        diff_up_neg = ffit_up_neg(field_up_neg) - conductance_up_neg
        up_minus.append(diff_up_neg)
        vec_up_minus.append(field_up_neg)
        '''

        fig, ax = MakePlot().create()
        plt.title(r'Difference in Conductance (800 $\mu A$ @ '+ temps[idx] + ') Up-Sweep -', fontsize=18)
        plt.plot(field_up_neg, diff_up_neg)
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
        ax.set_ylabel(r'$\delta G $', fontsize=14)
        ax.set_ylim((-0.5, 1.4))

        plt.show()
        '''
        # Down Sweep Negative

        resistance_down_neg = resistance[sweep_down_locs_neg_field]
        field_down_neg = field[sweep_down_locs_neg_field]
        conductance_down_neg = 1 / resistance_down_neg / conductance_quantum

        ffit_down_neg = fit_wo_step(resistance_down_neg, field_down_neg, conductance_down_neg)
        '''

        fig, ax = MakePlot().create()

        plt.title(r'800 $\mu A$ at ' + temps[idx] + ' Down-Sweep -', fontsize=18)
        ax.plot(field_down_neg, conductance_down_neg, label=temps[idx])
        ax.plot(linear_vector_2, ffit_down_neg(linear_vector_2), label='Fit w/o Step')
        plt.legend(title='Line', loc='best')
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
        ax.set_ylabel(r'Conductance $(\frac{2e^2}{h})$', fontsize=14)

        plt.show()
        '''

        diff_down_neg = ffit_down_neg(field_down_neg) - conductance_down_neg
        down_minus.append(diff_down_neg)
        vec_down_minus.append(field_down_neg)
        '''
        fig, ax = MakePlot().create()
        plt.title(r'Difference in Conductance (800 $\mu A$ @ ' + temps[idx] + ') Down-Sweep -', fontsize=18)
        plt.plot(field_down_neg, diff_down_neg)
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
        ax.set_ylabel(r'$\delta G $', fontsize=14)
        ax.set_ylim((-0.5, 1.4))

        plt.show()
        ''''''

        sns.set_context('paper')


        fig, ax = MakePlot().create()
        plt.title(r'Difference in Conductance (800 $\mu A$ @ ' + temps[idx] + ')', fontsize=18)
        plt.plot(field_up_pos,diff_up_pos, label='Positive Field Up-Sweep',
                 color=sns.color_palette('muted')[0],linewidth=2.0)
        plt.plot(field_down_pos, diff_down_pos, label='Positive Field Down-Sweep',linestyle='dashed',
                 color=sns.color_palette('muted')[0],linewidth=2.0)
        plt.plot(field_up_neg, diff_up_neg, label='Negative Field Up-Sweep',
                 color=sns.color_palette('muted')[1],linewidth=2.0)
        plt.plot(field_down_neg, diff_down_neg, label='Negative Field Down-Sweep',linestyle='dashed',
                 color=sns.color_palette('muted')[1],linewidth=2.0)
        plt.axhline(0,-14,16, color='k',alpha=0.75)
        plt.axhline(.25,-14,16, color='k',alpha=0.75)
        plt.axhline(0.5,-14,16, color='k',alpha=0.75)
        plt.axhline(1.0,-14,16, color='k',alpha=0.75)
        ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
        ax.set_ylabel(r'$\delta G $', fontsize=14)
        ax.set_ylim((-0.5, 1.4))
        plt.legend(title='Field Sweep')

        plt.show()
        '''

    sns.set_context('paper')

    fig, ax = MakePlot().create()

    clrs = sns.color_palette('husl', n_colors=10)

    for i in range(len(up_minus)):
        plt.plot(vec_up_plus[i], up_plus[i], label=temps[i],
                 color=clrs[i], linewidth=2.0)
        plt.plot(vec_down_plus[i], down_plus[i], label=temps[i], linestyle='dashed',
                 color=clrs[i], linewidth=2.0)
        plt.plot(vec_up_minus[i], up_minus[i], label=temps[i],
                 color=clrs[i], linewidth=2.0)
        plt.plot(vec_down_minus[i], down_minus[i], label=temps[i], linestyle='dashed',
                 color=clrs[i], linewidth=2.0)
    plt.title(r'Difference in Conductance (800 $\mu A$)', fontsize=18)
    plt.axhline(0, -14, 16, color='k', alpha=0.75)
    plt.axhline(.25, -14, 16, color='k', alpha=0.75)
    plt.axhline(0.5, -14, 16, color='k', alpha=0.75)
    plt.axhline(1.0, -14, 16, color='k', alpha=0.75)
    ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
    ax.set_ylabel(r'$\delta G $', fontsize=14)
    ax.set_ylim((-0.5, 1.4))

    ax.ticklabel_format(style='sci', axis='y')  # , scilimits=(0, 0)

    handles, labels = plt.gca().get_legend_handles_labels()

    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, title='Temperature', loc='best')

    plt.show()
