import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage
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

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



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

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap



hex_list = ['#0091ad', '#d6f6eb', '#fdf1d2', '#faaaae', '#ff57bb']
# remove x axis ticks

main_path = '/Users/npopiel/Documents/MPhil/Data/'

save_path = '/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/'

samples = ['VT26']
temps = (2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0)

temp_lab1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]#,21,22,23,24,25,26,27,28,29]#,19,20,21]#,22]


possible_currents1 = np.array([10,20,50,100,200,500,1000,1500])


#currents_log = np.logspace(-3,np.log10(1.5),10000)
currents_log = np.linspace(0.01,1.5,10000)

temperoos, cureentoos = [], []

temp_lst = []

xx, yy = np.meshgrid(temps, possible_currents1 / 1000)
data_grid_og = np.array([np.ravel(xx), np.ravel(yy)]).T
temperatures = np.linspace(2, temps[-1], 10000)

grid_x, grid_y = np.meshgrid(temperatures, currents_log)

sample_lst = []

for temp in temps:

    temp_path = main_path + 'VT26/' + str(temp) + '/'

    current_lst = []

    for current in possible_currents1:
        resistance, field = load_r_and_h(temp_path, current)

        min_field_loc = 0
        max_field_loc = np.argmax(field)

        ratio = resistance[max_field_loc] / resistance[min_field_loc]

        current_lst.append(ratio)
        temperoos.append(temp)
        cureentoos.append(current / 1000)

        # fig, ax = MakePlot().create()
        # plt.plot(field,resistance)
        # plt.title('Temp' + str(temp) + 'current' + str(current))
        # plt.show()

    current_arr = np.array(current_lst)
    nans, x = nan_helper(current_arr)
    current_arr[nans] = np.interp(x(nans), x(~nans), current_arr[~nans])

    temp_lst.append(current_arr)

array = np.array(temp_lst).T

other_grid = np.array([temperoos, cureentoos])

interpd = scipy.interpolate.griddata(data_grid_og, np.ravel(array), (grid_x, grid_y), method='cubic', rescale=True)
interpd = scipy.ndimage.median_filter(interpd,size=25)

#rdgn =mcolors.Colormap("gnuplot2_r", N=512)
rdgn = plt.get_cmap("gnuplot2_r")  # sns.color_palette('viridis',as_cmap=True, n_colors=1024)#
# if sample != 'SBF25':

# else:
#    divnorm = mcolors.TwoSlopeNorm(vmin=interpd.min(), vcenter=interpd.median(), vmax=interpd.max())
#interpd -=1
fig, ax = MakePlot().create()
vmin=interpd.min()
vmax=interpd.max()
vcenter = 0.2
divnorm = mcolors.TwoSlopeNorm(vmin=0.9999999, vcenter=1, vmax=interpd.max())  # interpd.max())
#divnorm = MidpointNormalize(vmin=interpd.min(), vcenter=0, vmax=1.5)
plt.imshow(interpd, cmap=rdgn, aspect='auto', interpolation='gaussian',extent=[2, temps[-1], 0.01, 1.5], origin='lower', norm=divnorm)
plt.xlabel('Temperature (K)', fontsize=24,fontname='arial')
plt.ylabel('Current (mA)', fontsize=24,fontname='arial')
ax.set_xscale('log')
plt.setp(ax.get_xticklabels(), fontsize=20, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=20, fontname='arial')

cbar = plt.colorbar()
cbar.ax.set_title(r'$\frac{R(14T)}{R(0T)}$ ',fontname='arial', fontsize=28,pad=10.0)
cbar.ax.tick_params(labelsize=18)
# plt.title(r'$\frac{R_{B=14}}{R_{B=0}}$ ('+sample+')',fontsize=22,usetex=True)
#plt.title(r'Magentoresistive Ratio for VT1 ' + r'$(\frac{R_{14T}}{R_{0T}})$ ', fontsize=16,fontname='arial')
#save_path = '/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/'
plt.savefig('/Users/npopiel/Desktop/heatmap-VT26b.png', dpi=400)
#plt.close()
#plt.show()

