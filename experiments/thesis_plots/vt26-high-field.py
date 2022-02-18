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

main_path = '/Users/npopiel/Documents/MPhil/Data/'

save_path = '/Volumes/GoogleDrive/My Drive/Thesis Figures/Transport/'

data = np.loadtxt('/Volumes/GoogleDrive/My Drive/Thesis Figures/NJM_Nov19_sweep.130.dat',
                  delimiter='\t',
                  skiprows=10)

field = data[:,0]
resistance = data[:,1]*1e5

fig, ax = MakePlot(figsize=(16, 9), nrows=1, ncols=1).create()


ax.plot(field, resistance, linewidth=1.8, c='k')

ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
ax.minorticks_on()

ax.tick_params('both', which='major', direction='in', length=6, width=2,
               bottom=True, top=True, left=True, right=True)

ax.tick_params('both', which='minor', direction='in', length=4, width=1.5,
               bottom=True, top=True, left=True, right=True)
ax.yaxis.offsetText.set_fontsize(18)
# plt.legend(title=r'Temperature $(K)$', loc='best',frameon=False, fancybox=False, framealpha=0, borderpad=1)
# ax.set_title('Magnetization in Positive Field Region',fontsize=title_size,fontname='arial', pad=30)
#
# ax.annotate(r'FeSb$_2$',xy=(4,3.2e-2),fontname='arial',fontsize=24,va='center',ha='center')
# ax.annotate(r'VT66',xy=(4,2.9e-2),fontname='arial',fontsize=24,va='center',ha='center')
# ax.annotate(r'$ \vec H \parallel c $',xy=(4,2.6e-2), fontname='arial',fontsize=24,va='center',ha='center')


plt.setp(ax.get_xticklabels(), fontsize=28, fontname='arial')
plt.setp(ax.get_yticklabels(), fontsize=28, fontname='arial')

# handles, labels = ax.get_legend_handles_labels()
# legend = fig.legend(handles, labels, framealpha=0, ncol=1,bbox_to_anchor=(1,1),  # len(dset)//12+
#                        title='Current (mA)', prop={"size": 18})
# plt.setp(legend.get_title(), fontsize=20, fontname='arial')


fig.text(0.4, 0.03, 'Magnetic Field (T)', fontname='arial', fontsize=32)
fig.text(0.03, 0.4, r'Resistance $(\Omega)$', fontname='arial', fontsize=32, rotation=90)

# plt.tight_layout()
#plt.show()
fig.savefig(save_path + 'raw_VT26-alexE-NJM.pdf', dpi=400)
