import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
def gradient_fill(x, y, ymin_val=None, ymax_val=None, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    ymin = y.min() if ymin_val is None else ymin_val
    ymax = y.max() if ymax_val is None else ymax_val

    xmin, xmax= x.min(), x.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


# f_c2a = lambda x, a, c, e : a * x ** 4 + c * x ** 2 + e
# f_c2a = lambda x, a, e : a * x ** 4  + e


f_c2a = lambda x, a, b, c, d, e : a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

files_c2a = ['/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/B2_dev_c-aVT69.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/B2_dev_c-aVT16PPMS.csv',
'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/B2_dev_c-aVT16TLH.csv']


dat_c2aVT69 = np.genfromtxt(r'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/B2_dev_c-aVT69.csv',delimiter=',',skip_header=1)

angles_c2aVT69 = dat_c2aVT69[:,0]
B_c2aVT69 = dat_c2aVT69[:,1]
err_c2aVT69 = dat_c2aVT69[:,2]

dat_c2aVT16PPMS = np.genfromtxt(r'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/B2_dev_c-aVT16PPMS.csv',delimiter=',',skip_header=1)

angles_c2aVT16PPMS = dat_c2aVT16PPMS[:,0]
B_c2aVT16PPMS = dat_c2aVT16PPMS[:,1]
err_c2aVT16PPMS = dat_c2aVT16PPMS[:,2]

dat_c2aVT16TLH = np.genfromtxt(r'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/B2_dev_c-aVT16TLH.csv',delimiter=',',skip_header=1)

angles_c2aVT16TLH = dat_c2aVT16TLH[:,0]
B_c2aVT16TLH = dat_c2aVT16TLH[:,1]
err_c2aVT16TLH = dat_c2aVT16TLH[:,2]

angles_c2a = np.hstack([angles_c2aVT69, angles_c2aVT16PPMS, angles_c2aVT16TLH])

sorted_inds = np.argsort(angles_c2a)
angles_c2a = angles_c2a[sorted_inds]
B_c2a = np.hstack([B_c2aVT69, B_c2aVT16PPMS, B_c2aVT16TLH])[sorted_inds]
err_c2a = np.hstack([err_c2aVT69, err_c2aVT16PPMS, err_c2aVT16TLH])[sorted_inds]

x_c2a = np.linspace(angles_c2a[0]-1,angles_c2a[-1]+1)

popt, pcov = curve_fit(f_c2a,angles_c2a,B_c2a)

y_c2a = f_c2a(x_c2a,*popt)


dat_c2abVT15 = np.genfromtxt('/Users/npopiel/Desktop/Hybrid/B2_dev_c-abVT15.csv',delimiter=',',skip_header=1)

angles_c2abVT15 = dat_c2abVT15[:,0]
B_c2abVT15 = dat_c2abVT15[:,1]
err_c2abVT15 = dat_c2abVT15[:,2]

dat_c2abVT4 = np.genfromtxt('/Users/npopiel/Desktop/Hybrid/B2_dev-c-abVT4b.csv',delimiter=',',skip_header=1)

angles_c2abVT4 = dat_c2abVT4[:,0]
B_c2abVT4 = dat_c2abVT4[:,1]
err_c2abVT4 = dat_c2abVT4[:,2]

angles_c2ab = np.hstack([angles_c2abVT15, angles_c2abVT4])

sorted_inds = np.argsort(angles_c2ab)
angles_c2ab = angles_c2ab[sorted_inds]
B_c2ab = np.hstack([B_c2abVT15, B_c2abVT4])[sorted_inds]
err_c2ab = np.hstack([err_c2abVT15, err_c2abVT4])[sorted_inds]

x_c2ab = np.linspace(angles_c2ab[0]-1,angles_c2ab[-1]+1)

popt, pcov = curve_fit(f_c2a,angles_c2ab,B_c2ab)

y_c2ab = f_c2a(x_c2ab,*popt)

# regime 3 data

# /Users/npopiel/Desktop/Hybrid/VT16-riii.dat
# /Users/npopiel/Desktop/Hybrid/VT69-riii.dat

dat_r3_vt69 = np.genfromtxt(r'/Users/npopiel/Desktop/Hybrid/VT69-riii.dat',delimiter='\t',skip_header=4)
dat_r3_vt16 = np.genfromtxt(r'/Users/npopiel/Desktop/Hybrid/VT16-riii.dat',delimiter='\t',skip_header=1)

angles_r3_VT69 = dat_r3_vt69[:,0]
angles_r3_VT16 = dat_r3_vt16[:,0]

B_r3_VT69 = dat_r3_vt69[:,1]
B_r3_VT16 = dat_r3_vt16[:,1]

angles_r3 = np.hstack([angles_r3_VT16, angles_r3_VT69])

sorted_inds = np.argsort(angles_r3)
angles_r3 = angles_r3[sorted_inds]
B_r3 = np.hstack([B_r3_VT16, B_r3_VT69])[sorted_inds]

popt, pcov = curve_fit(f_c2a,angles_r3,B_r3)

y_r3 = f_c2a(x_c2a,*popt)

fig, a = MakePlot(figsize=(8,6),gs=True).create()

gs = fig.add_gridspec(1,3)
ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[:, 2])


gradient_fill(x_c2a,45*np.ones(len(x_c2a)),ax=ax1,color='#87f6ff', ymax_val=45, ymin_val=0)
gradient_fill(x_c2a,y_r3,ax=ax1,color='#d65108', ymax_val=45, ymin_val=0)
gradient_fill(x_c2a,y_c2a,ax=ax1,color='#efa00b', ymin_val=0)

ax1.scatter(angles_c2aVT69, B_c2aVT69, marker='v', s=60, c='#591f0a',alpha=0.8)
ax1.errorbar(angles_c2aVT69, B_c2aVT69, yerr=err_c2aVT69/2, fmt='none', c='#591f0a', linewidth=2.1)

ax1.scatter(angles_c2aVT16PPMS, B_c2aVT16PPMS, marker='o', s=60, c='#591f0a',alpha=0.8)
ax1.errorbar(angles_c2aVT16PPMS, B_c2aVT16PPMS, yerr=err_c2aVT16PPMS/2, fmt='none', c='#591f0a', linewidth=2.1)

ax1.scatter(angles_c2aVT16TLH, B_c2aVT16TLH, marker='s', s=60, c='#591f0a',alpha=0.8)
ax1.errorbar(angles_c2aVT16TLH, B_c2aVT16TLH, yerr=err_c2aVT16TLH/2, fmt='none', c='#591f0a', linewidth=2.1)

ax1.scatter(angles_r3_VT69, B_r3_VT69, marker='v', s=60, c='#0e34a0',alpha=0.8)#0e34a0

ax1.scatter(angles_r3_VT16, B_r3_VT16, marker='s', s=60, c='#0e34a0',alpha=0.8)



ax1.plot(x_c2a, y_c2a, linewidth=2, c='darkslategray')
ax1.plot(x_c2a, y_r3, linewidth=2, c='darkslategray')


gradient_fill(x_c2ab,45*np.ones(len(x_c2ab)),ax=ax2,color='#d65108', ymax_val=45, ymin_val=0)
gradient_fill(x_c2ab,y_c2ab,ax=ax2,color='#efa00b', ymin_val=0)


ax2.scatter(angles_c2abVT4, B_c2abVT4, marker='D', s=60, c='#591f0a',alpha=0.8)
ax2.errorbar(angles_c2abVT4, B_c2abVT4, yerr=err_c2abVT4/2, fmt='none', c='#591f0a', linewidth=2.1)

ax2.scatter(angles_c2abVT15, B_c2abVT15, marker='h', s=60, c='#591f0a',alpha=0.8)
ax2.errorbar(angles_c2abVT15, B_c2abVT15, yerr=err_c2abVT15/2, fmt='none', c='#591f0a', linewidth=2.1)


ax2.plot(x_c2ab, y_c2ab, linewidth=2, c='darkslategray')

# ax2.fill_between(x_c2ab, y_c2ab, color = '#efa00b', hatch="/")
# ax2.fill_between(x_c2ab,y_c2ab, 45, color='#d65108',hatch="x")

ax1.set_ylim(0,45)
ax2.set_ylim(0,45)
ax2.set_xlim(0,28)

ax2.set_yticklabels([])

publication_plot(ax1, r'$\theta$', 'Magnetic Field (T)')
publication_plot(ax2, r'$\phi$', '')


plt.tight_layout(pad=1)
plt.show()


