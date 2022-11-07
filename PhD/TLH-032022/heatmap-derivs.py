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

def fit_powerlaw_as_function(inds_list, field, volts, plot=False):
    lst = []
    eps = 5e-7
    popt=(1,1,1)

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        model = Model(power_law2)

        params = model.make_params(b=2)  # can also set initial values here

        # optionally, put min/max bounds on parameters:
        params['b'].min = 0
        params['b'].max = 3
        #params['a'].vary = False

        result = model.fit(v, params, x=B)
        #
        # # print report with results and fitting statistics
        # print(result.fit_report())

        # print(np.mean(B))
        if plot:
            fig, axs = MakePlot(nrows=2).create()

            axs[0].plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            axs[0].plot(B, result.best_fit, lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')

            axs[1].plot(B, v-result.best_fit, lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')

            publication_plot(axs[0],'Field', 'Torque')
            publication_plot(axs[1], 'Field', 'Subtracted Torque')

            plt.show()
        # remove the 1,1,1 and put with actual parameters once I know where they are
        lst.append([np.mean(B), result.best_values['a'], result.best_values['b'], result.best_values['c']])

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


files = ['/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_60deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_64deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_67.5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_72deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_75.5deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_79deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.4K_82.5deg_sweep1.csv',
         '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/0.3K_90deg_sweep1.csv']

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
    volts = 1e7*medfilt(np.genfromtxt(f1, delimiter=',')[5:, 1],101)
    # fig, ax = MakePlot().create()
    # #
    # ax.plot(field, volts, lw=2, c=select_discrete_cmap('venasaur')[0])
    # # ax.plot(B, power_law(B, a, b, c), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
    # # print(b)
    # plt.show()
    # # remove area where field < 2

    low_B_inds = field >= B_thresh

    field = field[low_B_inds]
    volts = savgol_filter(volts[low_B_inds], 101, 3)


    # fields.append(field)
    # lambdas.append(volts)

    n = 100

    max_deriv = 25

    deriv_volts = volts[n-1::n] - volts[:-1*n:n]


    deriv_volts[deriv_volts>max_deriv] = max_deriv
    deriv_volts[deriv_volts < -1*max_deriv] = -1*max_deriv

    # fields.append(field[n-1::n])
    # lambdas.append(deriv_volts)

    fields.append(field[:-1])
    lambdas.append(np.diff(volts))

    # second_deriv_volts = np.diff(deriv_volts)
    # corresponding_fields = field[n-1::n][:-1]
    #
    # fields.append(corresponding_fields)
    # lambdas.append(second_deriv_volts)



#pseudocode
# array of fields
# array of angles

N = 800

possible_fields = fields[0]


big_list_angles = [angles[i]*np.ones(len(arr)) for i, arr in enumerate(fields)]

xx, yy = np.meshgrid(angles, possible_fields)
data_grid_og = np.array([np.ravel(xx), np.ravel(yy)]).T
fine_angles = np.linspace(angles[0], angles[-1], N)
fine_field = np.linspace(B_thresh, 41.5, 10000)

grid_x, grid_y = np.meshgrid(fine_angles, fine_field)

# fix angle, get lambda f(B)

# array = np.abs(np.array(lambdas)).T

# array[array>3] = 3

interpd = scipy.interpolate.griddata((np.concatenate(big_list_angles), np.concatenate(fields)), np.concatenate(lambdas), (grid_x, grid_y), method='cubic', rescale=True)
#interpd = scipy.ndimage.median_filter(interpd,size=25)


# interpd[interpd<0]=0


#
# rdgn = sns.color_palette("viridis", as_cmap=True,
#                          n_colors=1024)

# rdgn = plt.cm.RdYlBu()

#rdgn = plt.cm.hsv()

fig, ax = MakePlot().create()


norm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=25)

plt.imshow(interpd, cmap='RdYlBu_r', aspect='auto', extent=[60, 90, B_thresh, 41.5], origin='lower', norm=norm)

publication_plot(ax, 'Angle', 'Magnetic Field')
# ax.set_yscale('log')
plt.colorbar()
# plt.title(r'$\frac{R_{B=14}}{R_{B=0}}$ ('+sample+')',fontsize=22,usetex=True)
# plt.title(r'Magentoresistive Ratio for ' + sample + r'$(\frac{R_{14T}}{R_{0T}})$ ', fontsize=22)
# plt.savefig(save_path+sample+'_interpd_clipped.png',dpi=200)

plt.show()
print(array.shape)

#
# lst = []
# eps=5e-7
# popt=[1.6550562103875173e-07, 2.6435454206477895, 7.325353193822921e-07]
#
#
# fig, axs = MakePlot(ncols=3).create()
#
# arr = np.array(lst)
#
# fields = arr[:,0]
# As = arr[:,1]
# lambdas = arr[:,2]
# consts = arr[:,3]
#
# axs[0].plot(fields, As,lw=2,c=select_discrete_cmap('venasaur')[4])
# axs[1].plot(fields, lambdas,lw=2,c=select_discrete_cmap('venasaur')[4])
# axs[2].plot(fields, consts,lw=2,c=select_discrete_cmap('venasaur')[4])
# # ax.plot(B, power_law(B,a,b, c), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
#
# publication_plot(axs[0],'','A')
# publication_plot(axs[1],'',r'$\lambda$')
# publication_plot(axs[2],'','C')
#
# fig.text(0.5, -0.0001, 'Magnetic Field', ha='center',fontname='arial',fontsize=26)
#
# plt.tight_layout(pad=1)
# plt.show()
#
#
#
#
# # if this works, take median/mean and use the critical exponent for that field range.
# # then, interpolate over this range for all angles as complete heatmap
#
#
#
