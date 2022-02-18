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

df = load_matrix(main_path+'RvsT_FeSb2-VT1&26-cooldown.dat')
df = df[['Temperature (K)', 'Bridge 1 Resistance (Ohms)']]

resistance_vt1 = scipy.ndimage.median_filter(np.array(df['Bridge 1 Resistance (Ohms)']),size=3)
temperature_vt1 = np.array(df['Temperature (K)'])

# VT26
df = load_matrix(main_path+'RvsT_FeSb2-VT1&26-cooldown.dat')
df = df[['Temperature (K)', 'Bridge 2 Resistance (Ohms)']]

resistance_vt26 = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
temperature_vt26 = np.array(df['Temperature (K)'])

# VT64
df = load_matrix('/Volumes/GoogleDrive/My Drive/VT64/Transport/BRT_cooldown_VT73_VT64_VLS2.dat')
resistance_vt64 = scipy.ndimage.median_filter(np.array(df['Bridge 2 Resistance (Ohms)']),size=3)
temperature_vt64 = np.array(df['Temperature (K)'])

rs = [resistance_vt1, resistance_vt26, resistance_vt64]
ts = [temperature_vt1, temperature_vt26, temperature_vt64]
ranges = [[(1/300,0.03), (0.11, 0.16)],
          [(1/300,0.03), (0.135, 0.2)],
           [(1/300,0.03), (0.13, 0.22)]]

colors_main = ['purple', 'orange', 'green']
fig, axs = MakePlot(nrows=1, ncols=2,figsize=(32,18)).create()

ax=axs[0]
ax.plot(temperature_vt1, resistance_vt1, linewidth=5, c='purple', label='VT1')
ax.plot(temperature_vt26, resistance_vt26, linewidth=5, c='orange', label='VT26')
ax.plot(temperature_vt64, resistance_vt64, linewidth=5, c='green', label='VT64')

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


plt.setp(ax.get_xticklabels(), fontsize=36, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=36, fontname='arial')


legend = ax.legend(framealpha=0, ncol=1,#bbox_to_anchor=(0,1),  # len(dset)//12+
                       title='Sample', prop={"size": 32})

handles, labels = ax.get_legend_handles_labels()

plt.setp(legend.get_title(), fontsize=48, fontname='arial')

gap_colours = [('plum', 'magenta'),
               ('burlywood', 'darkgoldenrod'),
               ('mediumspringgreen', 'limegreen')]

ax=axs[1]
labels = ['VT1', 'VT26', 'VT64']

for i in range(len(rs)):


    y = np.log(rs[i])
    x = 1/ts[i]

    ax.plot(x, y, linewidth=5, c=colors_main[i], label=labels[i])

    range1 = ranges[i][0]
    range2 = ranges[i][1]

    # Fit Delta 1, this is where 1/T is in [1/300, 0.03] for VT1 and VT26 and VT64

    locs1 = np.where(np.logical_and(x>=range1[0], x<=range1[1]))
    gap1, c1 = np.polyfit(x[locs1],y[locs1], deg=1)
    x1 = np.linspace(range1[0]-.01,range1[1]+.01,25)
    y1 = gap1*x1 + c1

    # Fit Delta2, 1/T in [0.11, 0.16] for VT1
    # 1/T in [0.135, 0.2] for VT26
    # 1/T in [0.13, 0.22]
    locs2 = np.where(np.logical_and(x>=range2[0], x<=range2[1]))
    gap2, c2 = np.polyfit(x[locs2],y[locs2], deg=1)
    x2 = np.linspace(range2[0]-.02,range2[1]+.02,25)
    y2 = gap2*x2 + c2



    lab1 = r'$\Delta_{1} = $' + str(kelvin_2_mev(gap1).round(2)) + ' meV'
    lab2 = r'$\Delta_{2} = $' + str(kelvin_2_mev(gap2).round(2)) + ' meV'

    ax.plot(x1,y1,linestyle='dashed',linewidth=4.5,c=gap_colours[i][0], label=lab1)
    ax.plot(x2,y2,linestyle='dashed',linewidth=4.5,c=gap_colours[i][1], label=lab2)



ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

ax.minorticks_on()
ax.set_xlabel(r'$\frac{1}{T}$' + r' $(\mathrm{K}^{-1})$', fontsize=58, fontname='arial')
ax.set_ylabel(r'$\ln(R)$', fontsize=58, fontname='arial')

ax.tick_params('both', which='major', direction='in', length=12, width=4,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=8, width=3,
               bottom=True, top=True, left=True, right=True)
ax.yaxis.offsetText.set_fontsize(27)



plt.setp(ax.get_xticklabels(), fontsize=36, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=36, fontname='arial')

#ax.locator_params(axis='y', nbins=2)


legend = ax.legend(framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
                       title='Sample and Gap', prop={"size": 32})
#plt.subplots_adjust(left=0.09, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)

handles, labels = ax.get_legend_handles_labels()

plt.setp(legend.get_title(), fontsize=48, fontname='arial')

# plt.tight_layout()
#plt.show()
fig.savefig(save_path + 'RvT-two_panel-draft2.pdf', bbox_extra_artists=(legend,), bbox_inches='tight',dpi=400)

