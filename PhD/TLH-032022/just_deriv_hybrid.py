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


files = ['/Users/npopiel/Desktop/Hybrid/VT15-hybrid_0.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_7.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_10p5.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_12.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_14.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_15p5.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_16p5.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_17.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_21.dat']

angles_vt15 = ['0$^\mathregular{o}$', '7$^\mathregular{o}$', '10.5$^\mathregular{o}$', '12$^\mathregular{o}$', '14$^\mathregular{o}$',
               '15.5$^\mathregular{o}$', '16.5$^\mathregular{o}$', '17$^\mathregular{o}$', '21$^\mathregular{o}$']

angles_number_vt15 = [0, 7, 10.5, 12, 14, 15.5, 16.5, 17, 21]


med_num, savgol_num = 51, 501
N=500
for i, file in enumerate(files):

    B = np.genfromtxt(file, delimiter='\t', skip_header=4)[:,0]
    B_copy = B
    B_inds = B >13
    B = B[B_inds]
    x=B

    tau = median_filter(1e3*np.genfromtxt(file, delimiter='\t', skip_header=4)[:,1],med_num)
    tau_copy = tau
    tau = tau[B_inds]

    field = B
    volts = tau

    volts -= volts[0]

    if volts[0] > volts[10]:

        volts *= -1


    # deriv = np.diff(volts)

    deriv = savgol_filter(volts, savgol_num,2,deriv=1)
    second_deriv = savgol_filter(deriv, savgol_num, 2, deriv=1)
    # second_deriv = np.diff(deriv)

    mean_2nd_deriv_og = np.mean(second_deriv[N:2*N])
    std_2nd_deriv_og = np.std(second_deriv[:N])

    dev_locs = np.abs(second_deriv) > 3*np.abs(mean_2nd_deriv_og) #+ 25 * std_2nd_deriv_og


    dev_loc = np.argmax(dev_locs > 0)



    fig, axs = MakePlot(nrows=2).create()
    axs[0].plot(field, volts, linewidth=2,c='IndianRed',alpha=.7,label='Raw Data')
    axs[0].axvline(field[dev_loc], linewidth=2, linestyle='dashed', c='midnightblue', alpha=0.7)
    axs[1].plot(field, second_deriv, c='purple', alpha=0.5)
    axs[1].axhline(mean_2nd_deriv_og, linewidth=2, linestyle='dashed', c='darkorange', alpha=0.5)
    axs[1].axvline(field[dev_loc], linewidth=2, linestyle='dashed', c='midnightblue', alpha=0.7)

    publication_plot(axs[0], '', 'Raw Torque (arb,)',title=angles_vt15[i])
    publication_plot(axs[1], 'Magnetic FIeld (T)', r'$\frac{\partial^2\tau}{\partial B^2}$')

    # legend = axs[1].legend(framealpha=0, ncol=1, loc='best',
    #                     prop={'size': 16, 'family': 'arial'}, handlelength=0, columnspacing=1)
    #
    # for line, text in zip(np.flip(legend.get_lines()), np.flip(legend.get_texts())):
    #     text.set_color(line.get_color())

    plt.tight_layout(pad=1)

    plt.show()
