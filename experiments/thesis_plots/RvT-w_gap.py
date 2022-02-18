import pandas as pd
from tools.utils import *
from matplotlib.cm import get_cmap
from tools.DataFile import DataFile
from tools.MakePlot import *
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

main_path = '/Volumes/GoogleDrive/My Drive/Heatmap Cooldowns/'

save_path = '/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/'


possible_currents1 = np.array([10,20,50,100,200,500,1000,1500])

fig, ax = MakePlot(figsize=(32,18)).create()
# scan is at 1 mA
df = load_matrix(main_path+'RvsT_FeSb2-VT1&26-cooldown.dat')
df = df[['Temperature (K)', 'Bridge 1 Resistance (Ohms)']]

resistance = scipy.ndimage.median_filter(np.array(df['Bridge 1 Resistance (Ohms)']),size=3)
temperature = np.array(df['Temperature (K)'])

ax.plot(temperature, resistance, linewidth=5, c='purple', label='VT1')

df = load_matrix(main_path+'RvsT_FeSb2-VT1&26-cooldown.dat')
df = df[['Temperature (K)', 'Bridge 2 Resistance (Ohms)']]

resistance = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
temperature = np.array(df['Temperature (K)'])
ax.plot(temperature, resistance, linewidth=5, c='orange', label='VT26')

df = load_matrix('/Volumes/GoogleDrive/My Drive/VT64/Transport/BRT_cooldown_VT73_VT64_VLS2.dat')
resistance = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
temperature = np.array(df['Temperature (K)'])

ax.plot(temperature, resistance, linewidth=5, c='green', label='VT64')


# Here I use the gaps from before, namely:

# VT1
#Gap 1  13.966120247439143
#Gap 2  4.623796164104342
#Gap 3  0.13387629964095393

#VT26
#Gap 1  14.414517986995492
#Gap 2  5.044833474977517
#Gap 3  0.10220368516803635

#VT64
#Gap 1  14.800319380720282
#Gap 2  3.624122415315234
#Gap 3  0.16870960997800727

gap1_avg = np.average([13.966120247439143,14.414517986995492,14.800319380720282]).round(2)
gap1_std = np.std([13.966120247439143,14.414517986995492,14.800319380720282]).round(2)

gap2_avg = np.average([4.623796164104342,5.044833474977517,3.624122415315234]).round(2)
gap2_std = np.std([4.623796164104342,5.044833474977517,3.624122415315234]).round(2)

gap3_avg = np.average([0.13387629964095393, 0.10220368516803635, 0.16870960997800727]).round(2)
gap3_std = np.std([0.13387629964095393, 0.10220368516803635, 0.16870960997800727]).round(2)


# Region to shade for gap3 approx T in [1.8, 4]
mid_range3 = np.geomspace(1.8,3.5,1000)[500]
# Region to shade for gap2 approx T in [10,30]
mid_range2 = np.geomspace(10,30,1000)[500]
# region to shade for gap1 approx T in [100,300]
mid_range1 = np.geomspace(100,300,1000)[500]

ax.axvspan(1.8, 3.5, alpha=0.3, color='mediumslateblue')
str3 = r'$\Delta_{3} = $' + str(gap3_avg) + '\n' + r' $\pm$ ' + str(gap3_std) + ' meV'
ax.annotate(str3, xy=(mid_range3,100), fontname='arial',fontsize='36',ha='center',va='center')

ax.axvspan(10, 30, alpha=0.3, color='mediumvioletred')
str2 = r'$\Delta_{2} = $' + str(gap2_avg) + '\n' + r' $\pm$ ' + str(gap2_std) + '0 meV'
ax.annotate(str2, xy=(mid_range2,100), fontname='arial',fontsize='36',ha='center',va='center')

ax.axvspan(100, 300, alpha=0.3, color='mediumturquoise')
str1 = r'$\Delta_{1} = $' + str(gap1_avg) + '\n' + r' $\pm$'  + str(gap1_std) + ' meV'
ax.annotate(str1, xy=(mid_range1,100), fontname='arial',fontsize='36',ha='center',va='center')


#ax.set_title(str(current) + r' $\mu \mathrm{A}$', fontname='arial', fontsize=44)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
ax.set_yscale('log')
ax.set_xscale('log')

# ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.minorticks_on()
ax.set_xlabel('Temperature (K)', fontsize=58, fontname='arial')
ax.set_ylabel(r'Resistance $(\Omega)$', fontsize=58, fontname='arial')

ax.tick_params('both', which='major', direction='in', length=12, width=4,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=8, width=3,
               bottom=True, top=True, left=True, right=True)
ax.yaxis.offsetText.set_fontsize(27)
# plt.legend(title=r'Temperature $(K)$', loc='best',frameon=False, fancybox=False, framealpha=0, borderpad=1)
# ax.set_title('Magnetization in Positive Field Region',fontsize=title_size,fontname='arial', pad=30)
#
# ax.annotate(r'FeSb$_2$',xy=(4,3.2e-2),fontname='arial',fontsize=24,va='center',ha='center')
# ax.annotate(r'VT66',xy=(4,2.9e-2),fontname='arial',fontsize=24,va='center',ha='center')
# ax.annotate(r'$ \vec H \parallel c $',xy=(4,2.6e-2), fontname='arial',fontsize=24,va='center',ha='center')


plt.setp(ax.get_xticklabels(), fontsize=36, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=36, fontname='arial')

#ax.locator_params(axis='y', nbins=2)


legend = ax.legend(framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
                       title='Sample', prop={"size": 32})
#plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)

handles, labels = ax.get_legend_handles_labels()

plt.setp(legend.get_title(), fontsize=48, fontname='arial')

# cbar_ax = fig.add_axes([0.91, 0.09, 0.02, 0.84])
#
# cbar_tick_locs = np.arange(len(possible_currents1)) / len(possible_currents1)
#
# sm = plt.cm.ScalarMappable(cmap='viridis')
# cbar = fig.colorbar(sm, ticks=cbar_tick_locs, cax=cbar_ax, pad=0.04, aspect=30)
# cbar.ax.set_yticklabels(possible_currents1)
# cbar.ax.set_title(r'Current $(\mu \mathrm{A})$', fontname='arial', fontsize=36, pad=10.0)
# cbar.ax.tick_params(labelsize=28)

# fig.text(0.4, 0.03, 'Magnetic Field (T)', fontname='arial', fontsize=58)
# fig.text(0.03, 0.4, r'Resistance $(\Omega)$', fontname='arial', fontsize=58, rotation=90)
# fig.suptitle('3 Kelvin', fontname='arial', fontsize=69)
# plt.tight_layout()
#plt.show()
fig.savefig(save_path + 'RvT-w-gap_draft1.png', dpi=400)
