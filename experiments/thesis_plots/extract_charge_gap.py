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

# VT1

# df = load_matrix(main_path+'RvsT_FeSb2-VT1&26-cooldown.dat')
# df = df[['Temperature (K)', 'Bridge 1 Resistance (Ohms)']]
#
# resistance = scipy.ndimage.median_filter(np.array(df['Bridge 1 Resistance (Ohms)']),size=3)
# temperature = np.array(df['Temperature (K)'])

# VT26
# df = load_matrix(main_path+'RvsT_FeSb2-VT1&26-cooldown.dat')
# df = df[['Temperature (K)', 'Bridge 2 Resistance (Ohms)']]
#
# resistance = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
# temperature = np.array(df['Temperature (K)'])

# VT64
df = load_matrix('/Volumes/GoogleDrive/My Drive/VT64/Transport/BRT_cooldown_VT73_VT64_VLS2.dat')
resistance = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
temperature = np.array(df['Temperature (K)'])


y = np.log(resistance)
x = 1/temperature

ax.plot(x, y, linewidth=5, c='purple', label='VT1')


# Fit Delta 1, this is where 1/T is in [1/300, 0.03] for VT1 and VT26 and VT64

locs1 = np.where(np.logical_and(x>=1/300, x<=0.03))
gap1, c1 = np.polyfit(x[locs1],y[locs1], deg=1)
x1 = np.linspace(1/300,0.035,25)
y1 = gap1*x1 + c1

# Fit Delta2, 1/T in [0.11, 0.16] for VT1
# 1/T in [0.135, 0.2] for VT26
# 1/T in [0.13, 0.22]
locs2 = np.where(np.logical_and(x>=0.13, x<=0.22))
gap2, c2 = np.polyfit(x[locs2],y[locs2], deg=1)
x2 = np.linspace(0.13,0.25,25)
y2 = gap2*x2 + c2

# Fit Delta2, 1/T in [0.11, 0.16] for VT1 and VT26 and VT64
locs3 = np.where(np.logical_and(x>=0.26, x<=0.55))
gap3, c3 = np.polyfit(x[locs3],y[locs3], deg=1)
x3 = np.linspace(0.26,0.56,25)
y3 = gap3*x3 + c3

ax.plot(x1,y1,linestyle='dashed',linewidth=2.5,c='k')
ax.plot(x2,y2,linestyle='dashed',linewidth=2.5,c='k')
ax.plot(x3,y3,linestyle='dashed',linewidth=2.5,c='k')

print('Gap 1 ', kelvin_2_mev(gap1))
print('Gap 2 ', kelvin_2_mev(gap2))
print('Gap 3 ', kelvin_2_mev(gap3))





# df = load_matrix(main_path+'RvsT_FeSb2-VT1&26-cooldown.dat')
# df = df[['Temperature (K)', 'Bridge 2 Resistance (Ohms)']]
#
# resistance = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
# temperature = np.array(df['Temperature (K)'])
# ax.plot(temperature, resistance, linewidth=5, c='orange', label='VT26')
#
# df = load_matrix('/Volumes/GoogleDrive/My Drive/VT64/Transport/BRT_cooldown_VT73_VT64_VLS2.dat')
# resistance = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
# temperature = np.array(df['Temperature (K)'])

#ax.plot(temperature, resistance, linewidth=5, c='green', label='VT64')

#ax.set_title(str(current) + r' $\mu \mathrm{A}$', fontname='arial', fontsize=44)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
# ax.set_yscale('log')
# ax.set_xscale('log')

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
plt.show()
#fig.savefig(save_path + 'RvT-draft1.png', dpi=400)
