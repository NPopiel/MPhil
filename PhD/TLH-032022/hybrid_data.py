import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap


# ALL PATHS ARE WRONG SO FIX THEM YA DONKEY


files_vt4 = [
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_0.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_7.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_14.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_21.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_28.dat']



cmap = select_discrete_cmap('venasaur')

angles_vt4 = ['0$^\mathregular{o}$', '7$^\mathregular{o}$', '14$^\mathregular{o}$', '21$^\mathregular{o}$', '28$^\mathregular{o}$']

angles_number_vt4 = [0, 7, 14, 21, 28]

files_vt15 = ['/Users/npopiel/Desktop/Hybrid/VT15-hybrid_0.dat',
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




fig, (axs) = MakePlot(figsize=(8,10), gs=True).create()


gs = fig.add_gridspec(4,2)
ax1 = fig.add_subplot(gs[0:2, 0])
ax2 = fig.add_subplot(gs[0:2, 1])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[3, 0])
ax6 = fig.add_subplot(gs[3, 1])

fig2, (other_ax1, other_ax2) = MakePlot(figsize=(8,6), ncols=2).create()


vt16_ppms_mins, vt69_mins, vt16_tlh_mins = [], [], []
vt16_ppms_inflects, vt69_inflects, vt16_tlh_inflects = [], [], []

# ax3ins = ax3.inset_axes([0.12, 0.65, 0.55, 0.3])

for i, file in enumerate(files_vt4):

    B = np.genfromtxt(file, delimiter='\t',skip_header=3)[:,0]
    B_copy = B

    if i == 2:
        B_inds = B < 27

    elif i == 1:
        B_inds = B < 27

    else:
        B_inds = B < 50

    B = B[B_inds]
    x=B

    tau = 1e3*np.genfromtxt(file, delimiter='\t',skip_header=3)[:,1]
    tau_copy = tau
    tau = tau[B_inds]


    fit_y = np.polyfit(B, tau-tau[0], 3)

    first_deriv = 3 * fit_y[0] * x**2 + 2 * fit_y[1] * x + fit_y[2]

    second_deriv = 6 * fit_y[0] * x + 2 * fit_y[1]

    inflection_point = B[np.argmax(second_deriv >= 0)]
    min_point = B[np.argmin(np.poly1d(fit_y)(B))]

    print('Angle:', angles_number_vt4[i])
    print('Inflection: ', inflection_point)
    print('Min: ', min_point)


    ax1.plot(B, tau - tau[0], linewidth=2, c=plt.cm.autumn(i/len(files_vt4)), label=str(angles_vt4[i]), alpha=.6)
    ax1.plot(B, np.poly1d(fit_y)(B), linewidth=2, c=plt.cm.autumn(i / len(files_vt4)), linestyle='dashed', alpha=.6)

    other_ax1.plot(np.linspace(0,45,200), np.poly1d(fit_y)(np.linspace(0,45,200)), linewidth=2, c=plt.cm.autumn(i / len(files_vt4)), alpha=.6, label=str(angles_vt4[i]),)

    ax3.plot(B, second_deriv, linewidth=2, c=plt.cm.autumn(i / len(files_vt4)), alpha=.6)

    ax5.scatter(angles_number_vt4[i], inflection_point, s=200, c=plt.cm.autumn(i/len(files_vt4)), alpha=.4)
    # ax1.plot(B_copy[~B_inds], tau_copy[~B_inds] - tau[0], linewidth=2, linestyle='dashed', c=plt.cm.autumn(i/len(files)), alpha=.6)

    # ax3ins.plot(B, tau - tau[0], linewidth=2, c=plt.cm.autumn(i/len(files)), label=str(angles[i]), alpha=.6)
    #
    # ax4.scatter(angles_number[i], inflection_point, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    # ax5.scatter(angles_number[i], min_point, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    #
    # vt69_mins.append(min_point)
    # vt69_inflects.append(inflection_point)


    # ax.plot(B, VT69s[i] - VT69s[i][0] , linewidth=2, c=cmap[i], label=str(angles[i]))




for i, file in enumerate(files_vt15):

    B = np.genfromtxt(file, delimiter='\t',skip_header=3)[:,0]
    B_copy = B
    B_inds = B < 50
    B = B[B_inds]
    x=B

    tau = 1e3*np.genfromtxt(file, delimiter='\t',skip_header=3)[:,1]
    tau_copy = tau
    tau = tau[B_inds]

    if x[0] > x[-1]:
        x = np.flip(x)
        B = np.flip(B)
        tau = np.flip(tau)
        tau_copy = np.flip(tau)


    fit_y = np.polyfit(B, tau-tau[0], 3)

    first_deriv = 3 * fit_y[0] * x**2 + 2 * fit_y[1] * x + fit_y[2]

    second_deriv = 6 * fit_y[0] * x + 2 * fit_y[1]

    inflection_point = B[np.argmax(second_deriv >= 0)]
    min_point = B[np.argmin(np.poly1d(fit_y)(B))]

    print('Angle:', angles_number_vt15[i])
    print('Inflection: ', inflection_point)
    print('Min: ', min_point)


    ax2.plot(B, tau - tau[0], linewidth=2, c=plt.cm.winter(i/len(files_vt15)), label=str(angles_vt15[i]), alpha=.6)
    ax2.plot(B, np.poly1d(fit_y)(B), linewidth=2, c=plt.cm.winter(i / len(files_vt15)), linestyle='dashed', alpha=.6)

    other_ax2.plot(np.linspace(0,45,200), np.poly1d(fit_y)(np.linspace(0,45,200)), linewidth=2, c=plt.cm.winter(i / len(files_vt15)), alpha=.6,label=str(angles_vt15[i]),)

    ax4.plot(B, second_deriv, linewidth=2, c=plt.cm.winter(i / len(files_vt15)),  alpha=.6)
    ax6.scatter(angles_number_vt15[i], inflection_point, s=200, c=plt.cm.winter(i/len(files_vt15)), alpha=.4)
    # ax1.plot(B_copy[~B_inds], tau_copy[~B_inds] - tau[0], linewidth=2, linestyle='dashed', c=plt.cm.autumn(i/len(files)), alpha=.6)

    # ax3ins.plot(B, tau - tau[0], linewidth=2, c=plt.cm.autumn(i/len(files)), label=str(angles[i]), alpha=.6)
    #
    # ax4.scatter(angles_number[i], inflection_point, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    # ax5.scatter(angles_number[i], min_point, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    #
    # vt69_mins.append(min_point)
    # vt69_inflects.append(inflection_point)


    # ax.plot(B, VT69s[i] - VT69s[i][0] , linewidth=2, c=cmap[i], label=str(angles[i]))



# ax1.set_xticks([0,5,10,15])
# ax3ins.set_xticks([0,5,10,15])
# # ax.set_xticks([0,5,10,15])
# # ax.set_yticks([4,5,6,7,8])
# # ax4.set_yticks([6,8,10,12,14])
# ax4.set_xticks([0, 20, 40, 60, 80])
# ax4.set_xticklabels([0, 20, 40, '', 80])
#
# ax5.set_xticks([0, 20, 40, 60, 80])
# ax5.set_xticklabels([0, 20, 40, '', 80])



publication_plot(ax1, 'Magnetic Field (T)', r'Capacitance (arb.)', title='VT4 41.5 T')
publication_plot(ax2, 'Magnetic Field (T)', '', title='VT15 Hybrid')


publication_plot(other_ax1, 'Magnetic Field (T)', r'Capacitance (arb.)', title='VT4 41.5 T')
publication_plot(other_ax2, 'Magnetic Field (T)', '', title='VT15 Hybrid')

publication_plot(ax3, 'Magnetic Field (T)', r'$\frac{d\tau}{dB^2}$')
publication_plot(ax4, 'Magnetic Field (T)', '')

publication_plot(ax5, r'$\phi$', r'$B_{inflection}$')
publication_plot(ax6, r'$\phi$', '')

# publication_plot(ax3, 'Magnetic Field (T)', '', title='VT69 TLH March')
# publication_plot(ax3ins, '', '',label_fontsize=8, tick_fontsize=12,y_ax_sci=True)
# publication_plot(ax4, r'$\theta$', '$B_{inflection}$')
# publication_plot(ax5, r'$\theta$', '$B_{minima}$')

legend = ax1.legend(framealpha=0, ncol=2, loc='upper left',
              prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing = 1)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

handles, labels = ax2.get_legend_handles_labels()


legend = ax2.legend(handles[::-1], labels[::-1], framealpha=0, ncol=2, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0,columnspacing = 1)

for line,text in zip(np.flip(legend.get_lines()), np.flip(legend.get_texts())):
    text.set_color(line.get_color())


legend = other_ax1.legend(framealpha=0, ncol=2, loc='upper left',
              prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing = 1)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

handles, labels = other_ax2.get_legend_handles_labels()


legend = other_ax2.legend(handles[::-1], labels[::-1], framealpha=0, ncol=2, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0,columnspacing = 1)

for line,text in zip(np.flip(legend.get_lines()), np.flip(legend.get_texts())):
    text.set_color(line.get_color())


fig.tight_layout(pad=1)
fig2.tight_layout(pad=1)
plt.show()
# plt.savefig(r'G:\My Drive\data_for_prop\3materials-irrr.png', dpi=300)