import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.colors as mcolors

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp





hex_list = ['#0091ad', '#d6f6eb', '#fdf1d2', '#faaaae', '#ff57bb']
# remove x axis ticks

main_path = '/Users/npopiel/Documents/MPhil/Data/'

samples = [#'VT11',
           'VT1', 'VT51', 'SBF25', 'VT26','VT49']#
temps_1 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0)#,23.0)
temps_2 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0)
temps_3 = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0)#,10.0)
temp_ranges = [#temps_1,
               temps_2,
               temps_3,
               temps_1,
               temps_2,
               temps_3]

temp_lab1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
temp_lab2 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
temp_lab3 = [2,3,4,5,6,7,8,9]
temp_labs = [#temp_lab1,
             temp_lab2,temp_lab3,temp_lab1,temp_lab2,temp_lab3]

possible_currents1 = np.array([1,5,10,20,50,100,200,500,1000,1500])
possible_currents2 = np.array([1500,1000,500,200,100,50,20,10,5,1])

#currents_log = np.logspace(-3,np.log10(1.5),10000)
currents_log = np.linspace(0.001,1.5,10000)



sample_lst = []

for ind, sample in enumerate(samples):

    temperoos, cureentoos = [], []

    temp_lst = []

    temps = temp_ranges[ind]
    xx, yy = np.meshgrid(temps,possible_currents1/1000)
    data_grid_og = np.array([np.ravel(xx),np.ravel(yy)]).T
    temperatures = np.linspace(2, temps[-1],10000)

    grid_x, grid_y = np.meshgrid(temperatures,currents_log)

    for temp in temps:

        temp_path = main_path + sample + '/' + str(temp) +'/'

        current_lst = []

        for current in possible_currents1:

            resistance, field = load_r_and_h(temp_path, current)

            min_field_loc = np.where(field == np.amin(field))[0][0]
            max_field_loc = np.where(field == np.amax(field))[0][0]

            ratio = resistance[max_field_loc]/resistance[min_field_loc]

            current_lst.append(ratio)
            temperoos.append(temp)
            cureentoos.append(current/1000)

            fig, ax = MakePlot().create()
            plt.scatter(x=field,y=resistance)
            plt.title('Temp' + str(temp) + 'current' + str(current))
            plt.show()

        current_arr = np.array(current_lst)
        nans, x = nan_helper(current_arr)
        current_arr[nans] = np.interp(x(nans), x(~nans), current_arr[~nans])

        temp_lst.append(current_arr)

    array = np.array(temp_lst).T

    other_grid = np.array([temperoos,cureentoos])

    interpd = scipy.interpolate.griddata(data_grid_og,np.ravel(array),(grid_x,grid_y),method='cubic')

    rdgn = sns.color_palette("coolwarm", as_cmap=True, n_colors=1024)
    divnorm = mcolors.TwoSlopeNorm(vmin=interpd.min(),vcenter=1, vmax=interpd.max())

    fig, ax = MakePlot().create()
    plt.imshow(interpd, cmap=rdgn, aspect='auto',extent=[2 , temps[-1], 0.001 , 1.5],origin='lower', norm=divnorm)
    plt.xlabel('Temperature (K)', fontsize=16)
    plt.ylabel('Current (mA)', fontsize=16)
    #ax.set_yscale('log')
    plt.colorbar()
    # plt.title(r'$\frac{R_{B=14}}{R_{B=0}}$ ('+sample+')',fontsize=22,usetex=True)
    plt.title(r'Magentoresistive Ratio for ' + sample + r'$(\frac{R_{14T}}{R_{0T}})$ ', fontsize=22)
    # plt.savefig(main_path+sample+'_interpd.png',dpi=200)
    # plt.close()
    plt.show()
    print(array.shape)

