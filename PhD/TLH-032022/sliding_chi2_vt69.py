import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap

def g(x, c, p, q, w):
    return c + p * x ** 2 + q * np.exp(x / w)

def extract_moving_indices2(field, window_size = 1.5, n=.1):

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

def extract_moving_indices(field, n=.1):

    min_field = np.min(field)
    max_field = np.max(field)

    inds_list = []

    last_field = max_field

    first_field = max_field - n

    end_field = max_field

    while first_field >= min_field:

        inds1 = field < last_field
        inds2 = field > first_field

        inds = inds1 & inds2

        inds_list.append(inds)

        first_field -= n

    return inds_list

def power_law(x,a,b,c):
    return a * np.power(x,b) + c

def fit_g_as_function(inds_list, field, volts, plot=False):
    lst = []
    eps = 5e-7
    popt=(1,1,1)

    c,p,q,w = 1,1,1,1

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        popt, pcov = curve_fit(g, B, v, p0=(c,p,q,w), maxfev=500000)

        c, p, q, w = popt

        # print(np.mean(B))
        if plot:
            fig, ax = MakePlot().create()

            print(np.mean(B))

            ax.plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            ax.plot(B, g(B,c, p, q, w), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
            plt.show()

        lst.append([np.mean(B), c, p, q, w])

    return np.array(lst)

def exp(x,a,b):

    return a * np.exp(x*b)

def fit_exp_as_function(inds_list, field, volts, plot=False):
    lst = []
    eps = 5e-7
    popt=(1,1,1)

    a,b = 1,1

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        popt, pcov = curve_fit(exp, B, v, p0=(a,b), maxfev=500000)

        a,b = popt

        # print(np.mean(B))
        if plot:
            fig, ax = MakePlot().create()

            print(np.mean(B))

            ax.plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            ax.plot(B, exp(B,a,b), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
            plt.show()

        lst.append([np.min(B), a,b, np.sum(ind_set)])

    return np.array(lst)

def get_test_stat_b(full_field, full_volts, param_array):

    test_stat = []

    for j in range(len(param_array[:, 1])):
        residuals = full_volts - exp(full_field, param_array[j,1], param_array[j,2])

        sum_squares = np.sum(np.square(residuals))

        dof = param_array[j,3] - 2

        test_stat.append(sum_squares/dof)

    return test_stat

def get_test_stat(full_field, full_volts, param_array):

    residuals = full_volts - exp(full_field, param_array[:, 1], param_array[:, 2])

    return np.sum(np.square(residuals))






files = ["/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_60deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_64deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_67.5deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_72deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_79deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_82.5deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.3K_90deg_sweep1.csv"]

cmap = select_discrete_cmap('venasaur')

angles = ['60$^\mathregular{o}$', '64$^\mathregular{o}$', '67.5$^\mathregular{o}$', '72$^\mathregular{o}$', '79$^\mathregular{o}$', '82$^\mathregular{o}$','89$^\mathregular{o}$'
]

angles_number = [60, 64, 67.5, 72, 79, 82, 89]

fig, a = MakePlot(gs=True,figsize=(12,10)).create()
gs = fig.add_gridspec(3,3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[2, 1])
ax7 = fig.add_subplot(gs[0, 2])
ax8 = fig.add_subplot(gs[1:3, 2])

axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

for i, file in enumerate(files):

    B = np.genfromtxt(file, delimiter=',',skip_header=4)[:,0]
    B_copy = B
    B_inds1 = B > 5.5
    B_inds2 = B < 30
    B_inds = B_inds1 & B_inds2
    B = B[B_inds]
    x=B

    tau = 1e3*np.genfromtxt(file, delimiter=',', skip_header=4)[:,1]
    tau_copy = tau
    tau = tau[B_inds]

    field = B
    volts = tau

    volts -= volts[0]

    # if volts[0] > volts[10]:
    #
    #     volts *= -1


    fitting_poly_inds = extract_moving_indices(field,1)


    param_array = fit_g_as_function(fitting_poly_inds,field,volts,plot=False)

    test_stat = get_test_stat_b(field, volts, param_array)



    for j in range(len(param_array[:, 1])-1):
        axs[i].scatter(param_array[j, 0], test_stat[j],
                   c=plt.cm.plasma(i / len(angles)))

    publication_plot(axs[i],'','',title=str(angles[i]))



    ax8.plot(field, volts, linewidth=2,c=plt.cm.plasma(i / len(angles)), label=angles[i])


publication_plot(ax1, '', r'$\chi^2$')
publication_plot(ax2, '', r'$\chi^2$')
publication_plot(ax3, 'Magnetic Field (T)', r'$\chi^2$')
publication_plot(ax4, '', '')
publication_plot(ax5, '', '')
publication_plot(ax6, 'Magnetic Field (T)', '')
publication_plot(ax7, '', '')
publication_plot(ax8, 'Magnetic Field (T)', 'Capactance (arb.)')


legend = ax8.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 16, 'family': 'arial'}, handlelength=0,columnspacing = 1)

for line,text in zip(np.flip(legend.get_lines()), np.flip(legend.get_texts())):
    text.set_color(line.get_color())

plt.tight_layout(pad=1)

# plt.savefig('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/Popiel-Compendium-Heatmap/vt16-alpha.png', dpi=200)

plt.show()



