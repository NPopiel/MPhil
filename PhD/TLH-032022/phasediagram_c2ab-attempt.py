import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def colorFader(c1,c2,mix=0.): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    import matplotlib as mpl
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_rgb((1-mix)*c1 + mix*c2)

def gradient_fill(x, y, ymin_val=None, ymax_val=None, fill_color=None, alpha=None, ax=None, **kwargs):
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
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb

    for z_i in range(len(z)):

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


def gradient_fill2(x, y, c1='#FEEE00', c2='#D95F02',ymin_val=None, ymax_val=None, fill_color=None, alpha=None, ax=None, **kwargs):
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
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,-1] = 1

    for z_i in range(len(z)):

        z[z_i,:,:3] = colorFader(c1,c2,mix=z_i/len(z))

    ymin = y.min() if ymin_val is None else ymin_val
    ymax = y.max() if ymax_val is None else ymax_val

    xmin, xmax= x.min(), x.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower')#, zorder=zorder)

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


dat_c2ab = np.genfromtxt(r'/Users/npopiel/Desktop/Hybrid/B2_dev-c-ab-3.csv',delimiter=',',skip_header=1)

angles_c2ab = dat_c2ab[:,0]
B_c2ab = dat_c2ab[:,1]
err_c2ab = dat_c2ab[:,2]

x_c2ab = np.linspace(angles_c2ab[0]-1,95)

popt, pcov = curve_fit(f_c2a,angles_c2ab,B_c2ab)

y_c2ab = f_c2a(x_c2ab,*popt)


fig, ax2 = MakePlot(figsize=(16/3,6)).create()

def scale_lightness(rgb, scale_l):
    import colorsys
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

lighter_c1 =mcolors.rgb2hex(scale_lightness(mcolors.hex2color('#DC3A31'), 1))
lighter_c2 =mcolors.rgb2hex(scale_lightness(mcolors.hex2color('#DC3A31'), 1))

lighter_c3 =mcolors.rgb2hex(scale_lightness(mcolors.hex2color('#E05831'), 1))
lighter_c4 =mcolors.rgb2hex(scale_lightness(mcolors.hex2color('#E53D24'), 1))




gradient_fill2(x_c2ab,45*np.ones(len(x_c2ab)),ax=ax2,c1=lighter_c1,c2=lighter_c2, ymax_val=45, ymin_val=0, zorder=-1)

gradient_fill2(x_c2ab,y_c2ab,ax=ax2, ymin_val=0, c1=lighter_c3,c2=lighter_c4, zorder=-1)

# ax2.plot(x_c2ab, y_c2ab, linewidth=2, c='darkslategray')
ax2.scatter(angles_c2ab, B_c2ab, marker='v', s=60, c='#F0BA5E', zorder=3)
ax2.errorbar(angles_c2ab, B_c2ab, yerr=err_c2ab/4, fmt='none', c='#F0BA5E', linewidth=2.1,zorder=2,elinewidth=0.6, capsize=3)


# ax2.fill_between(x_c2ab, y_c2ab, color = '#efa00b', hatch="/")
# ax2.fill_between(x_c2ab,y_c2ab, 45, color='#d65108',hatch="x")

ax2.set_ylim(0,45)
ax2.set_xlim(-0.01,95)
ax2.set_xbound(-0.3, 95)

ax2.set_yticks([0,10,20,30,40])
ax2.set_xticks([0,7,14,21,28])

# ax2.set_yticklabels([])

import matplotlib as mpl

# read image file
with mpl.cbook.get_sample_data(r"/Users/npopiel/Desktop/Hybrid/c-ab_eatonedit-removebg-preview.png") as file:
    arr_image = plt.imread(file, format='png')

# Draw image
axin = ax2.inset_axes([3,1,12,12],transform=ax2.transData)    # create new inset axes in data coordinates
axin.imshow(arr_image)
axin.axis('off')

publication_plot(ax2, r'$\phi$ ($\degree$)', r'$\mu_0H$ (T)')


bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(width=0.8, headlength=10, headwidth=8,color='k')
trans = ax2.get_xaxis_transform() # x in data untis, y in axes fraction

ax2.annotate(r'$[001]$', xycoords=trans,
                 xy=(0, -0.1),
                 ha="center", va="center", fontname='arial', fontsize=18)

ax2.annotate(r'$\perp(110)$', xycoords=trans,
                 xy=(27.7, -0.15),xytext=(22, -0.15),
                 ha="center", va="center", fontname='arial', arrowprops=arrow_args, fontsize=18)


ax2.annotate(r'$[001]$', xycoords='data',
                 xy=(7.74, 13),
                 ha="center", va="center", fontname='arial', fontsize=15)

ax2.annotate(r'$\perp(110)$', xycoords='data',
                 xy=(17, 6.76),
                 ha="center", va="center", fontname='arial', fontsize=15)

arrow_args = dict(width=6, headlength=0.001, headwidth=0.001,color='#F08484')

ax2.annotate(r'$(110)$', xycoords='data',
                 xytext=(13.2,11.9), xy = (9.5, 10.55), arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
                 ha="center", va="center", fontname='arial', fontsize=15)


axin.annotate(r'$\phi$',
                 xy=(328, 122),xycoords='data',
                 ha="center", va="center", fontname='arial', arrowprops=arrow_args, fontsize=11)


ax2.annotate(r'Regime I', xycoords='axes fraction',
                 xy=(0.6,0.3),
                 ha="left", va="center", fontname='arial', fontsize=24, fontweight='bold')

ax2.annotate(r'Regime II', xycoords='axes fraction',
                 xy=(0.1,0.85),
                 ha="left", va="center", fontname='arial', fontsize=24, fontweight='bold')

plt.tight_layout(pad=1)
plt.show()
# with plt.rc_context({'image.composite_image': False}):
#     plt.savefig('/Users/npopiel/Desktop/Hybrid/phase_diagram_c2ab.pdf', dpi=300, transparent=True)
