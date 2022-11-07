import numpy as np
import matplotlib.pyplot as plt

from tools.utils import *

# -5 is NaN bc no 2T

hall_R_1T = np.array([0.035946, np.NaN, 0.0255, 0.017, 0.0085, -0.00517])
hall_R_2T = np.array([0.069456, np.NaN, 0.0487, 0.0327, 0.0154, -0.0118, ])
approx_angles = np.array([-7.5, np.NaN, -2.5, 0, 2.5])

fig, ax = MakePlot().create()

ax.scatter(approx_angles, hall_R_1T)

plt.show()