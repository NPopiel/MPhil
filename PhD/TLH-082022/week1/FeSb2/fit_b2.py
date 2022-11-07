import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap

def get_bsqr_deviation_analytic(field, volts, N, plot=False,threshold=3.,std=False):

    func = lambda x, alpha, beta, gamma, delta : alpha * x ** 2 + beta * np.exp(x * gamma) + delta

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts)

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]
    delta = popt[3]

    err_in_fit = np.sqrt(np.diag(pcov))

    err_in_deriv = 2*err_in_fit[0] + np.sqrt((err_in_fit[1]/beta)**2 + 2*(err_in_fit[2]/gamma)**2)

    second_deriv = 2 * alpha + beta * gamma ** 2 * np.exp(gamma * field)

    mean_2nd_deriv_og = np.mean(second_deriv[:N])
    std_2nd_deriv_og = np.std(second_deriv[:N])

    locs_of_mean = np.arange(N)
    locs = np.setdiff1d(np.arange(len(field)), locs_of_mean)



    if not std:
        dev_locs = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og)  # + 25 * std_2nd_deriv_og
    else:
        dev_locs = np.abs(second_deriv)[locs] > threshold * std_2nd_deriv_og

    min_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
    max_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

    dev_loc2nd = np.argmax(dev_locs > 0) + N

    min_err_loc = np.argmax(min_err > 0) + N
    max_err_loc = np.argmax(max_err > 0) + N

    if plot:
        fig, axs = plt.subplots(ncols=2,figsize=(16,9))
        axs[0].plot(field, volts,linewidth=2,c='indianred',label='Data')
        axs[0].plot(field, func(field,alpha, beta, gamma, delta),linewidth=2,c='midnightblue',label='Fit',linestyle='dashed')

        axs[1].plot(field, second_deriv,linewidth=2,c='darkgray')
        axs[1].axvline(field[dev_loc2nd])
        axs[0].axvline(field[dev_loc2nd])

        axs[0].legend(framealpha=0, ncol=1, loc='best',
                            prop={'size': 24, 'family': 'arial'})
        publication_plot(axs[0],'Magnetic Field (T)', 'Torque (arb.)')
        publication_plot(axs[1], 'Magnetic Field (T)', r'$\frac{\partial^2 \tau}{\partial B^2}$')
        plt.tight_layout(pad=1)
        plt.show()



    return field[dev_loc2nd], field[min_err_loc], field[max_err_loc]



