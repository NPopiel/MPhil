import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import median_filter
import seaborn as sns

main_path = '/Users/npopiel/Documents/MPhil/Data/step_data/'

df = pd.read_csv(main_path+'steplocs.csv')

temps_800 = np.array([1.75, 2,2.25,2.5,2.75,3])

upsweep800_b1s = [2.94153,3.80096,4.6564,5.79871,6.9417,8.65533]
upsweep800_b2s = [4.66094,5.51495,6.37178,7.23241,8.6558,10.08841]
dG_800 = [0.79499329,0.721376343,0.616883994,0.478612315,0.442405919,0.362756024]
B_at_max800 = [3.52791,4.67113,5.52707,6.66972,7.52316,9.52847]

down_sweep_bmax800 = [4.18747,4.47739,5.61582,6.75864,7.9038,9.6158]
downsweep_dg800 = [0.785087838,0.729703891,0.619737555,0.556096253,0.471782937,0.400373507]

dG800_pos = np.average(np.array([dG_800, downsweep_dg800]),axis=0)
yerr_pos_800 = np.std(np.array([dG_800, downsweep_dg800]),axis=0)
B800_pos = np.average(np.array([B_at_max800, down_sweep_bmax800]),axis=0)
xerr_pos_800 = np.std(np.array([B_at_max800, down_sweep_bmax800]),axis=0)

dG_800_neg_up = [0.82668116,0.741417642,0.708840738,0.487328884,0.481050777,0.422136315]
B_800_neg_up = [-3.61613,-4.47252,-5.32927,-6.76334,-8.18947,-9.90225]

dG_800_neg_down = [0.767675234,0.766226871,0.650277663,0.563551589,0.494985583,0.395527905]
B_800_neg_down = [-4.09888,-4.38417,-5.52681,-6.67018,-8.09849,-9.81255]

dG800_neg = np.average(np.array([dG_800_neg_up, dG_800_neg_down]),axis=0)
yerr_neg_800 = np.std(np.array([dG_800_neg_up, dG_800_neg_down]),axis=0)
B800_neg = np.average(np.array([B_800_neg_up, B_800_neg_down]),axis=0)
xerr_neg_800 = np.std(np.array([B_800_neg_up, B_800_neg_down]),axis=0)



temps_900 = np.array([1.75,2,2.25])

upsweep900_b1s = [8.08616,8.94385,10.08628]
upsweep900_b2s = [9.80149,10.65506,11.51821]
dG_900 = [0.447247355,0.405323512,0.323885376]
B_at_max900 = [8.95573,9.52829,10.66995]

dG_900_pos_down = [0.389427354,0.34935102,0.341579192]
B_900_pos_down = [9.04489,9.90247,11.04585]

dG900_pos = np.average(np.array([dG_900, dG_900_pos_down]),axis=0)
yerr_pos_900 = np.std(np.array([dG_900, dG_900_pos_down]),axis=0)
B900_pos = np.average(np.array([B_at_max900, B_900_pos_down]),axis=0)
xerr_pos_900 = np.std(np.array([B_at_max900, B_900_pos_down]),axis=0)

dG_900_neg_up = [0.427731652,0.396842746,0.331772202]
B_900_neg_up = [-9.33093,-10.18745,-11.3301]

dG_900_neg_down = [0.485933516,0.37910861,0.359077151]
B_900_neg_down = [-8.95647,-10.38452,-11.24178]

dG900_neg = np.average(np.array([dG_900_neg_up, dG_900_neg_down]),axis=0)
yerr_neg_900 = np.std(np.array([dG_900_neg_up, dG_900_neg_down]),axis=0)
B900_neg = np.average(np.array([B_900_neg_up, B_900_neg_down]),axis=0)
xerr_neg_900 = np.std(np.array([B_900_neg_up, B_900_neg_down]),axis=0)

line_pos800 = np.poly1d(np.polyfit(B800_pos,dG800_pos,1))
line_neg800 = np.poly1d(np.polyfit(B800_neg,dG800_neg,1))

line_pos900 = np.poly1d(np.polyfit(B900_pos,dG900_pos,1))
line_neg900 = np.poly1d(np.polyfit(B900_neg,dG900_neg,1))

linspace_pos = np.linspace(0,14,10)
linspace_neg = np.linspace(-14,0,10)

# Need to put in xy error bars on the points. STD of the above averages.


matplotlib.use('TkAgg')
fig = plt.figure(figsize=(16,9))
plt.interactive(False)



# Plot of dG v T

ax = plt.subplot(221)
ax.scatter(temps_800, dG800_pos, c='r', label=r'800 $\mu$A')
ax.scatter(temps_800, dG800_neg, c='r', marker='*')

ax.errorbar(temps_800, dG800_pos, yerr=yerr_pos_800, fmt='none', c='r', alpha=0.5)
ax.errorbar(temps_800, dG800_neg, yerr=yerr_neg_800, fmt='none', c='r', alpha=0.5)
#
# ax.plot(linspace_pos, line_pos800(linspace_pos),c='r',linestyle='dashed')
# ax.plot(linspace_neg, line_neg800(linspace_neg),c='r',linestyle='dashed')

ax.scatter(temps_900, dG900_pos, c='darkgreen',label=r'900 $\mu$A')
ax.scatter(temps_900, dG900_neg, c='darkgreen', marker='*')

ax.errorbar(temps_900, dG900_pos, yerr=yerr_pos_900, fmt='none', c='darkgreen', alpha=0.5)
ax.errorbar(temps_900, dG900_neg, yerr=yerr_neg_900, fmt='none', c='darkgreen', alpha=0.5)
#
# ax.plot(linspace_pos, line_pos900(linspace_pos),c='darkgreen',linestyle='dashed')
# ax.plot(linspace_neg, line_neg900(linspace_neg),c='darkgreen',linestyle='dashed')

#ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

ax.minorticks_on()

ax.tick_params('both', which='major', direction='in', length=6, width=2,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=4, width=1.5,
               bottom=True, top=True, left=True, right=True)


plt.setp(ax.get_xticklabels(), fontsize=22, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=22, fontname='arial')


ax.set_xlabel('Temperature (K)',fontname='arial',fontsize=24)
ax.set_ylabel(r'$\langle |\Delta G| \rangle$ $(\frac{2e^2}{h})$',fontname='arial',fontsize=24)





# plot of TvB

ax = plt.subplot(223)

ax.scatter(B800_pos,temps_800,  c='r', label=r'800 $\mu$A')
ax.scatter(B800_neg,temps_800,  c='r')

ax.errorbar(B800_pos, temps_800, xerr=xerr_pos_800, fmt='none', c='r', alpha=0.5)
ax.errorbar(B800_neg, temps_800, xerr=xerr_neg_800, fmt='none', c='r', alpha=0.5)
#
# ax.plot(linspace_pos, line_pos800(linspace_pos),c='r',linestyle='dashed')
# ax.plot(linspace_neg, line_neg800(linspace_neg),c='r',linestyle='dashed')

ax.scatter(B900_pos, temps_900, c='darkgreen',label=r'900 $\mu$A')
ax.scatter(B900_neg, temps_900, c='darkgreen')

ax.errorbar(B900_pos, temps_900, xerr=xerr_pos_900, fmt='none', c='darkgreen', alpha=0.5)
ax.errorbar(B900_neg, temps_900,  xerr=xerr_neg_900, fmt='none', c='darkgreen', alpha=0.5)
#
# ax.plot(linspace_pos, line_pos900(linspace_pos),c='darkgreen',linestyle='dashed')
# ax.plot(linspace_neg, line_neg900(linspace_neg),c='darkgreen',linestyle='dashed')

#ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
ax.set_ylim(0,3.2)

ax.minorticks_on()

ax.tick_params('both', which='major', direction='in', length=6, width=2,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=4, width=1.5,
               bottom=True, top=True, left=True, right=True)

#legend = plt.legend(title=r'Current', loc='best',frameon=False, fancybox=False, framealpha=0, borderpad=1, prop={"size": 18})




plt.setp(ax.get_xticklabels(), fontsize=22, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=22, fontname='arial')

plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1,  top=0.9, wspace=0.2, hspace=0.4)

# handles, labels = ax.get_legend_handles_labels()
# legend = fig.legend(handles, labels, framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
#                        title='Current (mA)', prop={"size": 18})
#plt.setp(legend.get_title(), fontsize=20, fontname='arial')


ax.set_ylabel('Temperature (K)',fontname='arial',fontsize=24)
ax.set_xlabel(r'$\mu_o H ^{\star}$ (T)',fontname='arial',fontsize=24)


# Plot with Linear trend
ax=plt.subplot(122)

ax.scatter(B800_pos, dG800_pos, c='r', label=r'800 $\mu$A')
ax.scatter(B800_neg, dG800_neg, c='r')

ax.errorbar(B800_pos, dG800_pos, xerr=xerr_pos_800, yerr=yerr_pos_800, fmt='none', c='r', alpha=0.5)
ax.errorbar(B800_neg, dG800_neg, xerr=xerr_neg_800, yerr=yerr_neg_800, fmt='none', c='r', alpha=0.5)

ax.plot(linspace_pos, line_pos800(linspace_pos),c='r',linestyle='dashed')
ax.plot(linspace_neg, line_neg800(linspace_neg),c='r',linestyle='dashed')

ax.scatter(B900_pos, dG900_pos, c='darkgreen',label=r'900 $\mu$A')
ax.scatter(B900_neg, dG900_neg, c='darkgreen')

ax.errorbar(B900_pos, dG900_pos, xerr=xerr_pos_900, yerr=yerr_pos_900, fmt='none', c='darkgreen', alpha=0.5)
ax.errorbar(B900_neg, dG900_neg, xerr=xerr_neg_900, yerr=yerr_neg_900, fmt='none', c='darkgreen', alpha=0.5)

ax.plot(linspace_pos, line_pos900(linspace_pos),c='darkgreen',linestyle='dashed')
ax.plot(linspace_neg, line_neg900(linspace_neg),c='darkgreen',linestyle='dashed')

#ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

ax.minorticks_on()

ax.tick_params('both', which='major', direction='in', length=6, width=2,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=4, width=1.5,
               bottom=True, top=True, left=True, right=True)

legend = plt.legend(title=r'Current', loc='best',frameon=False, fancybox=False, framealpha=0, borderpad=1, prop={"size": 18})




plt.setp(ax.get_xticklabels(), fontsize=22, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=22, fontname='arial')

plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1,  top=0.9, wspace=0.2, hspace=0.4)

# handles, labels = ax.get_legend_handles_labels()
# legend = fig.legend(handles, labels, framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
#                        title='Current (mA)', prop={"size": 18})
plt.setp(legend.get_title(), fontsize=20, fontname='arial')


ax.set_xlabel('Magnetic Field (T)',fontname='arial',fontsize=24)
ax.set_ylabel(r'$\langle |\Delta G| \rangle$ $(\frac{2e^2}{h})$',fontname='arial',fontsize=24)


plt.savefig('/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/all_step_together.png', dpi=300)
#plt.show()

