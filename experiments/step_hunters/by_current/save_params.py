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

conductance_quantum = (7.748091729 * 10 ** -5)

curr_500 = [#'VT64_1p75K500uA.csv',
            #'VT64_2p0K500uA.csv',
            #'VT64_2p25K500uA.csv',
            #'VT64_2p5K500uA.csv',
            #'VT64_2p75K500uA.csv',
            'VT64_3p0K500uA.csv']
            # 'VT64_3p25K500uA.csv',
            # 'VT64_3p5K500uA.csv',
            # 'VT64_3p75K500uA.csv',
            # 'VT64_4p0K500uA.csv']
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
            'VT64_2p25K900uA.csv']#,
            # 'VT64_2p5K900uA.csv',
            # 'VT64_2p75K900uA.csv',
            # 'VT64_3p0K900uA.csv']#,
            # 'VT64_3p25K900uA.csv',
            # 'VT64_3p5K900uA.csv',
            # 'VT64_3p75K900uA.csv',
            # 'VT64_4p0K900uA.csv']
curr_1000 = ['VT64_1p75K1000uA.csv',
             'VT64_2p0K1000uA.csv',
             'VT64_2p25K1000uA.csv',
             'VT64_2p5K1000uA.csv',
             'VT64_2p75K1000uA.csv',
             'VT64_3p0K1000uA.csv']
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
temps_b = ['1.75 K', '2.00 K', '2.25 K']#, '2.50 K', '2.75 K', '3.00 K', '3.25 K']#, '3.5 K']  # , '3.75 K']
currents = [r'500 $\mu A$', r'800 $\mu A$', r'900 $\mu A$', r'1000 $\mu A$', r'1100 $\mu A$', r'1200 $\mu A$',
            r'1500 $\mu A$']

temperatures = [1.75,2.00,2.25]#,2.5,2.75,3.00]
data_sets = [curr_900]#[curr_900, curr_1000]#, curr_1100, curr_1200, curr_1500] #curr_500,

sample = 'VT64'

sweeps = ['Positive Field Up-Sweep',
          'Negative Field Up-Sweep',
          'Positive Field Down-Sweep',
          'Negative Field Down-Sweep']

def integrate_the_step(field, fitted_function, real_resistance, dx=100):

    y_diff = np.abs(fitted_function(field) - real_resistance)

    area = np.trapz(y_diff,dx=dx)

    return area

def get_B1_and_B2(first_diff):

    B1, B2 = np.argmax(first_diff) - 1, np.argmin(first_diff) + 1

    return B1, B2

def fit_wo_step(resistance, field, window_length=5, polyorder=3, deg=3, return_field = False, plot=False,
                num_pts=6,return_area=False, return_shortened_locs=False,plt_check=False):
    resistance_smooth = scipy.signal.savgol_filter(x=resistance, window_length=window_length, polyorder=polyorder)

    first_diff = np.diff(resistance_smooth)

    B1, B2 = get_B1_and_B2(first_diff)

    if plt_check:
        fig, ax = MakePlot().create()
        ax.plot(field, 1 / resistance)
        plt.show()

        inpt = input('Continue with fit? Y/N')
        if inpt.lower() == 'n':
            raise UserWarning



    step_locs = np.arange(B1, B2)

    # select a region with 6 points before and 6 points after step

    # add in functionality here to ensure num_pts not greater/less than total
    num_left, num_right = num_pts, num_pts

    if B1-num_left < 0:
        num_left = 0
    if B2+num_right>field.shape[0]:
        num_right = field.shape[0]-1

    smaller_region = np.arange(B1-num_left, B2+num_right)

    B1 = field[B1]
    B2 = field[B2]

    region_to_fit = np.setdiff1d(smaller_region, step_locs)

    fitted_line = np.polyfit(x=field[region_to_fit], y=resistance[region_to_fit], deg=deg)

    if plot:
        fig, ax = MakePlot().create()
        ax.plot(field, 1/resistance)
        ax.plot(np.linspace(0,14,200),1/np.poly1d(fitted_line)(np.linspace(0,14,200)))
        plt.show()
    if return_shortened_locs:
        step_locs = smaller_region
        if return_area:

            area = integrate_the_step(field[smaller_region],np.poly1d(fitted_line),resistance[smaller_region])

            if return_field:
                return np.poly1d(fitted_line), B1, B2, area, step_locs
            else:
                return np.poly1d(fitted_line), area, step_locs

        else:

            if return_field:
                return np.poly1d(fitted_line), B1, B2, step_locs
            else:
                return np.poly1d(fitted_line), step_locs
    else:
        if return_area:

            area = integrate_the_step(field[smaller_region], np.poly1d(fitted_line), resistance[smaller_region])

            if return_field:
                return np.poly1d(fitted_line), B1, B2, area
            else:
                return np.poly1d(fitted_line), area

        else:

            if return_field:
                return np.poly1d(fitted_line), B1, B2
            else:
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

    b1s_upplus, b1s_upminus, b1s_downplus, b1s_downminus = [], [], [], []
    b2s_upplus, b2s_upminus, b2s_downplus, b2s_downminus = [], [], [], []

    up_plus, up_minus, down_plus, down_minus = [], [], [], []
    vec_up_plus, vec_up_minus, vec_down_plus, vec_down_minus = [], [], [], []

    max_up_plus, max_up_minus, max_down_plus, max_down_minus = [],[],[],[]
    max_up_plus_field, max_up_minus_field, max_down_plus_field, max_down_minus_field = [], [], [], []

    for idx, current_temp_data_name in enumerate(curr_data_name):
        dat = load_matrix(main_path + current_temp_data_name)
        res_lst.append(dat[0])
        field_lst.append(dat[1])
        label_lst.append([temps[idx]] * len(dat[0]))

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

        ffit_up_pos, B1, B2, area, small_locs = fit_wo_step(resistance_up_pos, field_up_pos,
                                                            return_field=True, return_area=True,
                                                            return_shortened_locs=True,plt_check=False)

        #fit_up_pos = ffit_up_pos/conductance_quantumf

        print(area)

        b1s_upplus.append(B1)
        b2s_upplus.append(B2)
        print('Delta B', np.abs(B2-B1))

        linear_vector = np.linspace(0, 14, 100)


        # Up-Sweep Positive

        conductance_fit = 1/ffit_up_pos/conductance_quantum

        diff_up_pos = median_filter(np.abs(1/ffit_up_pos(field_up_pos[small_locs])/conductance_quantum - 1/resistance_up_pos[small_locs]/conductance_quantum),4)
        up_plus.append(diff_up_pos)
        vec_up_plus.append(field_up_pos[small_locs])
        max_up_plus.append(np.max(diff_up_pos))
        max_up_plus_field.append(field_up_pos[small_locs][np.argmax(diff_up_pos)])


        # Down Sweep Positive

        resistance_down_pos = resistance[sweep_down_locs_pos_field]
        field_down_pos = field[sweep_down_locs_pos_field]
        conductance_down_pos = 1 / resistance_down_pos / conductance_quantum

        ffit_down_pos, B1, B2, area, small_locs = fit_wo_step(resistance_down_pos, field_down_pos,return_field=True, return_area=True,return_shortened_locs=True)

        linear_vector = np.linspace(0, 14, 100)


        diff_down_pos = median_filter(np.abs(1/ffit_down_pos(field_down_pos[small_locs])/conductance_quantum - 1/resistance_down_pos[small_locs]/conductance_quantum),4)
        down_plus.append(diff_down_pos)
        vec_down_plus.append(field_down_pos[small_locs])
        max_down_plus.append(np.max(diff_down_pos))
        max_down_plus_field.append(field_down_pos[small_locs][np.argmax(diff_down_pos)])
        b1s_downplus.append(B1)
        b2s_downplus.append(B2)


        # Up-Sweep Negative

        resistance_up_neg = resistance[sweep_up_locs_neg_field]
        field_up_neg = field[sweep_up_locs_neg_field]
        conductance_up_neg = 1 / resistance_up_neg / conductance_quantum

        ffit_up_neg, B1, B2, area, small_locs = fit_wo_step(resistance_up_neg, field_up_neg, return_field=True, return_area=True, return_shortened_locs=True)

        linear_vector_2 = np.linspace(-14, 0, 100)



        diff_up_neg = median_filter(np.abs(1/ffit_up_neg(field_up_neg[small_locs])/conductance_quantum - 1/resistance_up_neg[small_locs]/conductance_quantum),4)
        up_minus.append(diff_up_neg)
        vec_up_minus.append(field_up_neg[small_locs])
        max_up_minus.append(np.max(diff_up_neg))
        max_up_minus_field.append(field_up_neg[small_locs][np.argmax(diff_up_neg)])
        b1s_upminus.append(B1)
        b2s_upminus.append(B2)

        # Down Sweep Negative

        resistance_down_neg = resistance[sweep_down_locs_neg_field]
        field_down_neg = field[sweep_down_locs_neg_field]
        conductance_down_neg = 1 / resistance_down_neg / conductance_quantum

        ffit_down_neg, B1, B2, area, small_locs = fit_wo_step(resistance_down_neg, field_down_neg, return_field=True, return_area=True, return_shortened_locs=True)


        diff_down_neg = median_filter(np.abs(1/ffit_down_neg(field_down_neg[small_locs])/conductance_quantum - 1/resistance_down_neg[small_locs]/conductance_quantum),4)
        down_minus.append(diff_down_neg)
        vec_down_minus.append(field_down_neg[small_locs])
        max_down_minus.append(np.max(diff_down_neg))
        max_down_minus_field.append(field_down_neg[small_locs][np.argmax(diff_down_neg)])
        b1s_downminus.append(B1)
        b2s_downminus.append(B2)

    upsweep_pos_field = pd.DataFrame({'field':max_up_plus_field,'max_delta_g':max_up_plus,
                         'temp':temperatures,'B1':b1s_upplus,'B2':b2s_upplus,
                         'sweep':[sweeps[0]]*len(max_up_plus)})
    upsweep_minus_field = pd.DataFrame({'field':max_up_minus_field,'max_delta_g':max_up_minus,
                         'temp':temperatures,'B1':b1s_upminus,'B2':b2s_upminus,
                         'sweep':[sweeps[1]]*len(max_down_plus)})
    downsweep_pos_field = pd.DataFrame({'field':max_down_plus_field,'max_delta_g':max_down_plus,
                         'temp':temperatures,'B1':b1s_downplus,'B2':b2s_downplus,
                         'sweep':[sweeps[2]]*len(max_down_plus)})
    downsweep_minus_field = pd.DataFrame({'field':max_down_minus_field,'max_delta_g':max_down_minus,
                         'temp':temperatures,'B1':b1s_downminus,'B2':b2s_downminus,
                         'sweep':[sweeps[3]]*len(max_down_minus)})

    df = pd.concat([upsweep_pos_field,downsweep_pos_field,upsweep_minus_field,downsweep_minus_field])

    df.to_csv(main_path+'steplocs_900uA.csv',index=False)