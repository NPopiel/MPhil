from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter
import seaborn as sns
from lmfit import Model
# Get each angle
# get sweep for each angle
# choose grid of fields (maybe 1.5 T window, ie (1,1.5,2) (1.5,2,2.5) middle of range is the 'value of field' for exponent
# get sparse grid of field, exponent pairs
# interp to smooth
# make heatmap

# fiddle with spacing to try and get this info
# alternatively you can use some order of derivative

# or raw-subtracted polynomial


f1 = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT16/day1/1.5K_-7.5deg_sweep1.csv'



def extract_moving_indices(field, window_size = 1.5, n=.1):

    min_field = np.min(field)
    max_field = np.max(field)

    inds_list = []

    last_field = min_field

    end_field = np.round(window_size + last_field,1)
    while end_field <= max_field:
        end_field = np.round(window_size + last_field,1)

        inds1 = field > last_field
        inds2 = field < end_field

        inds = inds1 & inds2

        inds_list.append(inds)

        last_field += n

    return inds_list

def power_law(x,a,b,c):
    return a * np.power(x,b) + c
def power_law2(x,a=1,b=2,c=.1):
    return a * np.power(x,b) + c

def exp(x,a,b,c):

    return a*np.exp(x*b) + c

def fit_power_as_function(inds_list, field, volts, plot=False):
    lst = []
    eps = 5e-7
    popt=(1,1,1)

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        # a0, b0, c0 = popt
        a0, b0, c0 = (.1,2,.1)
        a0 += 1
        b0+=1
        c0+=1

        popt, pcov = curve_fit(power_law, B, v, p0=(a0,b0,c0),maxfev=150000)

        a, b, c = popt

        # print(np.mean(B))
        if plot:
            fig, ax = MakePlot().create()

            ax.plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            ax.plot(B, power_law(B,a,b, c), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
            print(b)
            plt.show()

        lst.append([np.mean(B), a, b, c])

    return np.array(lst)



files2 = [
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_2deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_5deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_9deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_13deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_16deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_19deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_23deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_30deg_sweep1.csv',
# # '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.36K_-10.5deg_sweep1.csv',
# # '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.36K_-7.5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.36K_45deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.36K_55deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/6K_65deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/6K_75deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/6K_85deg_sweep1.csv']

files = ['/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_60deg_sweep1.csv']
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_64deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_67.5deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_72deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_75.5deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_79deg_sweep1.csv',
# '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_82.5deg_sweep1.csv',
#          '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.3K_90deg_sweep1.csv']

angles2 = [45,55,65,75,85]
angles = [60, 64, 67.5, 72, 75.5, 79, 82.5, 90]

fits_by_files = {}
fields = []
lambdas = []
As, Cs = [],[]

B_thresh = .75
for i, f1 in enumerate(files):
    print(f1)

    field = np.genfromtxt(f1, delimiter=',')[5:, 0]
    volts = medfilt(np.genfromtxt(f1, delimiter=',')[5:, 1],59)
    # fig, ax = MakePlot().create()
    # #
    # ax.plot(field, volts, lw=2, c=select_discrete_cmap('venasaur')[0])
    # # ax.plot(B, power_law(B, a, b, c), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
    # # print(b)
    # plt.show()
    # # remove area where field < 2

    low_B_inds = field >= B_thresh



    field = field[low_B_inds]
    volts = volts[low_B_inds]
    volts -= volts[0]

    upper_inds = field < 27

    field = field[upper_inds]
    volts = volts[upper_inds]

    inds_list = np.arange(len(field))

    popt, pcov = curve_fit(exp, field, volts)


    fig, ax = MakePlot().create()

    ax.plot(field, volts, linewidth=2, c='midnightblue')
    ax.plot(field, popt[0]*np.exp(field*popt[1]), linewidth=2, c='indianred')

    # ax.set_ylim(0,.1)

    plt.show()

