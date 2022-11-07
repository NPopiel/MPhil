from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter
import seaborn as sns
from scipy.signal import argrelmin
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

def fit_g_as_function(inds_list, field, volts, plot=False):
    lst = []
    eps = 5e-7
    popt=(1,1,1)

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        # a0, b0, c0 = popt
        c0, p0, q0, w0 = (.1,2,.1,.1)

        c0+=eps

        popt, pcov = curve_fit(g, B, v, p0=(c0, p0, q0, w0),maxfev=150000)

        c, p, q, w = popt

        # print(np.mean(B))
        if plot:
            fig, ax = MakePlot().create()

            ax.plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            ax.plot(B, g(B,c, p, q, w), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
            plt.show()

        lst.append([np.mean(B), c, p, q, w])

    return np.array(lst)



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
cs, ps, qs, ws = [], [], [], []

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
    volts-=volts[0]

    fitting_poly_inds = field <= 45

    field_fit = field[fitting_poly_inds]
    volts_fit = volts[fitting_poly_inds]

    f = np.poly1d(np.polyfit(field_fit, volts_fit, 2))


    inds_list = extract_moving_indices(field_fit, n=3, window_size=3)

    param_array = fit_g_as_function(inds_list, field_fit, volts_fit, plot=True)

    fits_by_files[angles[i]] = param_array

    fields.append(param_array[:, 0])
    cs.append(param_array[:, 1])
    ps.append(param_array[:, 2])
    qs.append(param_array[:, 3])
    ws.append(param_array[: ,4])

    fig, ax = MakePlot().create()

    for j in range(len(param_array[:,1])):

        ax.scatter(np.mean(param_array[j,0]), param_array[j,2]/param_array[j,3], c=plt.cm.jet(j/len(param_array[:,0])))

    publication_plot(ax, 'Magnetic Field (T)', r'\frac{p}{q}')

    plt.show()





    # fig, axs = MakePlot(nrows=2).create()
    # axs[0].plot(field, volts)
    # axs[0].plot(field_fit, g(field_fit, c, p, q, w), c='r', linestyle='dashed')
    # #axs[0].plot(field, f(field), c='r', linestyle='dashed')
    # axs[1].plot(field_fit, (q*np.exp(field_fit/w))+p*field_fit**2, c='midnightblue', label='Ratio')
    # axs[1].plot(field_fit, p * field_fit ** 2 , c='purple', linestyle='dashed', alpha=0.6, label='Quadratic')
    # axs[1].plot(field_fit, (q * np.exp(field_fit/w)), c='forestgreen', linestyle='dashed', alpha=0.6, label='Exponential')
    #
    # publication_plot(axs[0], '','Torque (arb,)')
    # publication_plot(axs[1], 'Magnetic FIeld (T)', 'Fit Torque (arb,)')
    # axs[1].legend()
    # plt.show()

N = 500

possible_fields = np.array(fields[0])


xx, yy = np.meshgrid(angles, possible_fields)
data_grid_og = np.array([np.ravel(xx), np.ravel(yy)]).T
fine_angles = np.linspace(angles[0], angles[-1], N)
fine_field = np.linspace(B_thresh, 41.5, N)

grid_x, grid_y = np.meshgrid(fine_angles, fine_field)

# fix angle, get lambda f(B)

fields = np.array(fields)

quad = 2 * np.array(ps) * possible_fields

exp = np.array(qs)*np.exp(possible_fields/np.array(ws))/np.array(ws)

fig, ax = MakePlot().create()

ax.plot(possible_fields, exp[0])
plt.show()
#
# deviation_inds = argrelmin(quad[0]-exp[0])[0]

# parameter = possible_fields[deviation_inds]
#


array = (np.array(ps)/np.array(qs)).T

# array[array>3] = 3

interpd = scipy.interpolate.griddata(data_grid_og, np.ravel(array), (grid_x, grid_y), method='cubic')#, rescale=True)
# interpd = scipy.ndimage.median_filter(interpd,size=25)


# interpd[interpd<0]=0



rdgn = sns.color_palette("viridis", as_cmap=True,
                         n_colors=1024)

#rdgn = plt.cm.hsv()

fig, ax = MakePlot().create()

plt.imshow(interpd, cmap=rdgn, aspect='auto', extent=[2, 85, B_thresh, 41.5], origin='lower')

publication_plot(ax, 'Angle', 'Magnetic Field')
# ax.set_yscale('log')
plt.colorbar()
# plt.title(r'$\frac{R_{B=14}}{R_{B=0}}$ ('+sample+')',fontsize=22,usetex=True)
# plt.title(r'Magentoresistive Ratio for ' + sample + r'$(\frac{R_{14T}}{R_{0T}})$ ', fontsize=22)
# plt.savefig(save_path+sample+'_interpd_clipped.png',dpi=200)

plt.show()


