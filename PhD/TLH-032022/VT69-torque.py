from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import seaborn as sns

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

        (a,b,c) = np.polyfit(B, v, deg=2)



        # print(np.mean(B))
        if plot:
            fig, ax = MakePlot().create()

            ax.plot(B,v,lw=2, c=select_discrete_cmap('venasaur')[0])
            ax.plot(B, a*B**2 + b*B +c, lw=2, c=select_discrete_cmap('venasaur')[6], linestyle='dashed')
            print(b)
            plt.show()

        lst.append([np.mean(B), a, b, c])

    return np.array(lst)



def get_bsqr_deviation_analytic(field, volts, N, plot=False,threshold=3.,std=False):

    func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta * np.exp(x * gamma)

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts)

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]
    print('(',alpha,',',beta,',',gamma,')')
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
        axs[0].plot(field, func(field,alpha, beta, gamma),linewidth=2,c='midnightblue',label='Fit',linestyle='dashed')

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


files = ['/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/0.4K_2222deg_sweep1.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/0.4K_2222deg_sweep2.csv']

linestyles = ['solid', 'dashed']

fig, ax = MakePlot(figsize=(7,9)).create()
# #
B_thresh = 10
med_num, savgol_num = 51, 501
N=50
threshold=1.001
threshold_numeric = 3
for i, f1 in enumerate(files):
    print(f1)

    field = np.genfromtxt(f1, delimiter=',')[5:, 0]
    volts = medfilt(np.genfromtxt(f1, delimiter=',')[5:, 1],31)
    const = 1

    B_dev_num = get_bsqr_deviation_analytic(field, volts, 500, threshold=threshold_numeric, plot=True)
    print(B_dev_num)

    if i == 1:
        const = 13.92/3.99
    ax.plot(field, const*1e4*volts, lw=2, c='#007EA7', linestyle=linestyles[i])

publication_plot(ax, 'Magnetic Field (T)', 'Torque (arb.)')

ax.set_ybound(-0.1, 14.1)
ax.set_xbound(0,41.8)
ax.set_xticks([0,10,20,30,40])
plt.tight_layout(pad=2)
plt.show()