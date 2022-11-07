import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap

main_path = r'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/'



B = np.genfromtxt('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2021_07_TLH/0.36/0.35K_30deg_sweep1.csv', delimiter=',', skip_header=(4), dtype=np.float64)[:,0]
tau = np.genfromtxt('/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2021_07_TLH/0.36/0.35K_30deg_sweep1.csv', delimiter=',', skip_header=(4), dtype=np.float64)[:,1]

inds = B > 6.5

B = B[inds]
tau = tau[inds]

f = np.poly1d(np.polyfit(B[B<11]**2, 1e4*tau[B<11], deg=1))

fig, ax = MakePlot(figsize=(8,8)).create()

ax.plot(B**2, tau*1e4, c='midnightblue', label=)
ax.plot(B**2, f(B**2), c='darkslategray', linestyle='dashed')

ax.annotate('Fit')
topax = ax.twiny()

topax.set_xlim([6,41.5])
topax.set_xlabel(r'$\mu_0 H$ (T)',fontsize=24 , fontname='arial', labelpad=10)
plt.setp(topax.get_xticklabels(),fontsize=24 , fontname='arial')

# publication_plot(axtop,r'$\mu_0 H$ (T)', r'')
# publication_plot(ax,r'$(\mu_0 H)^2$ (T$^2$)', r'$\tau$ ($\times 10^{-3}$ $\mu_B$T per f.u.)',)

ax.set_xlabel(r'$(\mu_0 H)^2$ (T$^2$)', fontname='arial',fontsize=24)
ax.set_ylabel(r'$\tau$ ($\times 10^{-3}$ $\mu_B$T per f.u.)', fontname='arial',fontsize=24)

plt.setp(ax.get_xticklabels(),fontsize=22, fontname='arial')
plt.setp(ax.get_yticklabels(),fontsize=22, fontname='arial' )

ax.minorticks_on()
ax.tick_params('both', which='major', direction='in',
      bottom=True, top=False, left=True, right=True,length=6, width=2)
ax.tick_params('both', which='minor', direction='in',
      bottom=True, top=False, left=True, right=True,length=4, width=1.6)
topax.tick_params(which='major', length=6, width=2,
                  direction='in',top=True, labeltop=True, left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)

topax.tick_params(which='minor', length=4, width=1.6,
                  direction='in',top=True, labeltop=True, left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)


from matplotlib.ticker import AutoMinorLocator

ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
topax.xaxis.set_minor_locator(AutoMinorLocator(2))



plt.tight_layout(pad=1)

plt.show()