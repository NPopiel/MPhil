from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter
import seaborn as sns
from scipy.ndimage import median_filter

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

def fit_poly_as_function(inds_list, field, volts, plot=False):
    lst = []
    eps = 5e-7
    popt=(1,1,1)

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        # a0, b0, c0 = popt
        a0, b0, c0 = (.1,2,.1)
        a0 += eps
        b0+=.001
        c0+=eps

        f = np.poly1d(np.polyfit(B, v, deg=2))



        lst.append([B, v, f])

    return np.array(lst)



# Fit 2nd order poly as a sliding window
# On that window, find the angle as a function of field between the fit and the data

def extract_angle2(B, v, f, plot=False):

        inds = np.arange(len(B))

        thetas = []

        for (i1, i2) in zip(inds[:-1], inds[1:]):


            delta_B = B[i2] - B[i1]
            delta_v = v[i2] - v[i1]
            delta_f = f(B[i2]) - f(B[i1])

            B_lin = [B[i1], B[i2]]
            V_lin = [v[i1], v[i2]]
            f_lin = [f(B[i1]), f(B[i2])]

            lin_dat = np.polyfit(B_lin, V_lin, 1)[0]
            lin_f = np.polyfit(B_lin, f_lin, 1)[0]


            vec1 = np.array([delta_B, delta_v])
            vec2 = np.array([delta_B, delta_f])

            norm_v1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
            norm_v2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

            theta = np.degrees(np.arccos(((np.dot(vec1, vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))))

            # fig, ax = MakePlot().create()
            #
            # ax.plot(10*vec1)
            # ax.plot(10 * vec2)
            # plt.show()



            #
            # # theta = np.rad2deg(np.arctan(np.abs((delta_v/delta_B - delta_f/delta_B) / (1 + delta_v*delta_f/delta_B/delta_B))))
            # theta = np.degrees(np.arctan(
            #     (lin_f - lin_dat) / (1 + lin_dat*lin_f)))


            thetas.append(theta)

        if plot:
            fig, ax = MakePlot().create()

            ax.plot()



        thetas = np.array(thetas)
        return median_filter(thetas,45)


def extract_angle3(B, v, f):


    diff_B = np.diff(B)
    diff_v = np.diff(v)
    diff_f = np.diff(f(B))

    thetas = np.rad2deg(
        np.arctan((diff_v/diff_B - diff_f/diff_B) / (1 + diff_v*diff_f/diff_B/diff_B)))


    return median_filter(thetas, 25)


def extract_angle4(B, v, f):


    diff_B = np.diff(B)
    diff_v = np.diff(v)
    diff_f = np.diff(f(B))

    thetas = np.rad2deg(np.arctan((diff_v/diff_B)) - np.arctan(diff_f/diff_B))


    return median_filter(thetas, 25)


files = [
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_2deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_9deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_13deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_16deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_19deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_23deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.35K_30deg_sweep1.csv',
# # '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.36K_-10.5deg_sweep1.csv',
# # '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.36K_-7.5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.36K_45deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/0.36K_55deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/6K_65deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/6K_75deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/DataSet/6K_85deg_sweep1.csv']

angles = [2,5,9,13,16,19,23,30,45,55,65,75,85]

fits_by_files = {}
fields = []
lambdas = []
As, Cs = [],[]

def g(x, c, p, q, w):
    return c + p*x**2 + q * np.exp(x/w)

B_thresh = 10
for i, f1 in enumerate(files):
    print(f1)

    field = np.genfromtxt(f1, delimiter=',')[5:, 0]
    volts = medfilt(np.genfromtxt(f1, delimiter=',')[5:, 1],31)

    field = field[:len(field) - 15]
    volts = volts[:len(volts) - 15]

    low_B_inds = field >= B_thresh

    field = field[low_B_inds]
    volts = volts[low_B_inds]

    fitting_poly_inds = field <= 17

    field_fit = field[fitting_poly_inds]
    volts_fit = volts[fitting_poly_inds]

    f = np.poly1d(np.polyfit(field_fit, volts_fit, 2))

    # here get the angle

    popt, pcov = curve_fit(g, field_fit, volts_fit, maxfev=15000)

    c = popt[0]
    p = popt[1]
    q = popt[2]
    w = popt[3]

    print('Params: ', popt)
    # find where theta is greater than X times the first value





    fig, axs = MakePlot(nrows=2).create()
    axs[0].plot(field, volts)
    axs[0].plot(field_fit, g(field_fit, c, p, q, w), c='r', linestyle='dashed')
    #axs[0].plot(field, f(field), c='r', linestyle='dashed')
    axs[1].plot(field_fit, (q*np.exp(field_fit/w))+p*field_fit**2, c='midnightblue', label='Ratio')
    axs[1].plot(field_fit, p * field_fit ** 2 , c='purple', linestyle='dashed', alpha=0.6, label='Quadratic')
    axs[1].plot(field_fit, (q * np.exp(field_fit/w)), c='forestgreen', linestyle='dashed', alpha=0.6, label='Exponential')

    publication_plot(axs[0], '','Torque (arb,)')
    publication_plot(axs[1], 'Magnetic FIeld (T)', 'Fit Torque (arb,)')
    axs[1].legend()
    plt.show()


