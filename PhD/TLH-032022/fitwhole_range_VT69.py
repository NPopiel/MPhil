import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap

def g(x, c, p, q, w):
    return c + p * x ** 2 + q * np.exp(x / w)

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
    c, p, q, w = 1,1,1,1

    for ind_set in inds_list:

        v = volts[ind_set]
        B = field[ind_set]

        popt, pcov = curve_fit(g, B, v, p0=(c,p,q,w),maxfev=150000)

        c, p, q, w = popt

        # print(np.mean(B))
        if plot:
            fig, ax = MakePlot().create()

            ax.plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            ax.plot(B, g(B,c, p, q, w), lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
            plt.show()

        lst.append([np.mean(B), c, p, q, w])

    return np.array(lst)


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



for i, file in enumerate(files):

    B = np.genfromtxt(file, delimiter=',')[:,0]
    B_copy = B
    B_inds = B < 50
    B = B[B_inds]
    x=B

    tau = 1e3*np.genfromtxt(file, delimiter=',')[:,1]
    tau_copy = tau
    tau = tau[B_inds]

    field = B[B_inds]
    volts = tau[B_inds]

    volts -= volts[0]

    if volts[0] > volts[10]:

        volts *= -1
    fitting_poly_inds1 = field <= 32
    fitting_poly_inds2 = field >= 0

    field_fit = field[fitting_poly_inds1 & fitting_poly_inds2]
    volts_fit = volts[fitting_poly_inds1 & fitting_poly_inds2]


    popt, pcov = curve_fit(g, field_fit, volts_fit, maxfev=15000)

    c = popt[0]
    p = popt[1]
    q = popt[2]
    w = popt[3]

    fig, axs = MakePlot(nrows=2).create()
    axs[0].plot(field, volts)
    axs[0].plot(field_fit, g(field_fit, c, p, q, w), c='r', linestyle='dashed')
    # axs[0].plot(field, f(field), c='r', linestyle='dashed')
    axs[1].plot(field_fit, (q * np.exp(field_fit / w)) + p * field_fit ** 2, c='midnightblue', label='Sum')
    axs[1].plot(field_fit, p * field_fit ** 2, c='purple', linestyle='dashed', alpha=0.6, label='Quadratic')
    axs[1].plot(field_fit, (q * np.exp(field_fit / w)), c='forestgreen', linestyle='dashed', alpha=0.6,
                label='Exponential')

    publication_plot(axs[0], '', 'Torque (arb,)')
    publication_plot(axs[1], 'Magnetic FIeld (T)', 'Fit Torque (arb,)')

    legend = axs[1].legend(framealpha=0, ncol=1, loc='best',
                        prop={'size': 16, 'family': 'arial'}, handlelength=0, columnspacing=1)

    for line, text in zip(np.flip(legend.get_lines()), np.flip(legend.get_texts())):
        text.set_color(line.get_color())

    plt.tight_layout(pad=1)

    plt.show()
