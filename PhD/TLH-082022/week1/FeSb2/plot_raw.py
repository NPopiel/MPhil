import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sounddevice as sd


def get_bsqr_deviation_analytic_no_const(field, volts, N, plot=False,threshold=3.,std=False):

    func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta * np.exp(x * gamma)

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts)

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]

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



colours = ['#001219',
           '#005F73',
           '#0A9396',
           '#94D2BD',
           '#E9D8A6',
           '#EE9B00',
           '#CA6702',
           '#BB3E03',
           '#AE2012',
           '#9B2226']

resistance_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT19/resistance/'
torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT19/torque/'

fig, a = MakePlot(figsize=(12, 12), gs=True).create()
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[0,1])
ax4 = fig.add_subplot(gs[1,1])
# # Draw image
# axin = ax1.inset_axes([0.12,.5,.27,.35])    # create new inset axes in data coordinates


sweeps = ['0.29K_-6.66deg_sweep010_up.csv',
          '0.29K_-3.33deg_sweep017_down.csv',
          '0.29K_0deg_sweep019_up.csv',
          '0.29K_6.66deg_sweep022_up.csv',
          '0.29K_10deg_sweep024_down.csv',
          '0.29K_13.33deg_sweep026_up.csv',
          '0.29K_20deg_sweep033_down.csv',
          '0.29K_26deg_sweep040_down.csv']

lstyles = ['solid', 'dashed']

angles = [-6.66, -3.33, 0, 6.66, 10, 13.33, 20, 26.66]

med_num, savgol_num = 51, 501
N=50
threshold=1.5

grads = []

for i, s_name in enumerate(sweeps):


    torque_dat = load_matrix(torque_path + s_name)
    field = torque_dat[:, 0]
    tau = torque_dat[:, 1]

    if field[0] > field[-1]:
        field = np.flip(field)
        tau = np.flip(tau)

    # if tau[-1] < 0:
    #     tau *=-1

    tau -= tau[0]

    if i > 5:
        tau *= 100 / 14.14

    fit_locs = field > 2


    # B_dev_anal, min_err_loc, max_err_loc = get_bsqr_deviation_analytic_no_const(field[fit_locs], tau[fit_locs], 50,plot=True, threshold=threshold)

    resistance_dat = load_matrix(resistance_path + s_name)
    B = resistance_dat[:, 0]
    V = resistance_dat[:, 1]  # mV

    I = 10e-6 #A

    R = np.abs(V / I) #Ohm, I tink abs val

    ax1.plot(field, tau*1e5, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')


    ax2.plot(field**2, tau * 1e5, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')

    f = np.poly1d(np.polyfit(field[:round(len(field)/1)]**2, tau[:round(len(field)/1)]*1e5, 1))

    ax2.plot(field**2, f(field**2), linewidth=2, c=colours[i], linestyle='dashed')


    slope = np.polyfit(field[:round(len(field)/4)]**2, tau[:round(len(field)/4)]*1e5, 1)[0]
    grads.append(slope)

    ax4.scatter(angles[i], slope, c=colours[i], s=275)


    if i != 0:
        ax3.plot(B, R, linewidth=2, c=colours[i])



def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 0., guess_offset])

    def sinfunc(t, A, p, c):  return A * np.sin(2*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=250000)
    A, p, c = popt
    f = 2/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(2*t + p) + c
    return {"amp": A, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

res = fit_sin(np.deg2rad(angles), grads)
print( "Amplitude=%(amp)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )

oversamples_angles = np.deg2rad(np.linspace(-10,90,250))


ax4.plot(np.rad2deg(oversamples_angles), res["fitfunc"](oversamples_angles), c='k', linewidth=2)

print(np.rad2deg(oversamples_angles)[np.argmax(res["fitfunc"](oversamples_angles))])



publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (arb.)')
publication_plot(ax2, r'$(\mu_0H)^2$ (T$^2$)', r'$\tau$ (arb.)')
publication_plot(ax3, r'$\mu_0H$ (T)', r'$R$ ($\Omega$)')
publication_plot(ax4, r'$\theta$ ($\degree$)', r'$\tau^\prime$ (arb.)')

handles, labels = ax1.get_legend_handles_labels()

# ax1.set_ybound(-10,18)

ax1.set_xbound(0,45)

ax3.set_ybound(0,7000)
# ax3.set_yticks([0, 2500, 5000])

legend = ax1.legend(handles, labels, framealpha=0, ncol=1, loc='lower right',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=1)

plt.show()




