import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# NEed dictionary with temperature key, each amplitude as a list



def lifshitz_kosevich(temps, e_mass, amp, field=20.0):  # 31.876923076923077

    kb = 1.380649e-23
    me = 9.1093837015e-31
    hbar = 1.054571817e-34
    qe = 1.602176634e-19

    chi = 2 * np.pi * np.pi * kb * temps * me * e_mass / (hbar * qe * field)

    r_lk = amp * chi / np.sinh(chi)

    return r_lk




def lk_field_val(min_field, max_field):
    denom = 1 / min_field + 1 / max_field
    return 2 / denom

temps = np.linspace(0.000001, 50, 100)

e_mass = 0.8

lk_field1 = lk_field_val(30, 33.5)
lk_field2 = lk_field_val(38.3, 41.4)

f1 = 1e3
f2 = 2e3

from tools.utils import *

fig, ax = MakePlot().create()

amps1 = lifshitz_kosevich(temps, e_mass, 1, field=lk_field1)
amps2 = lifshitz_kosevich(temps, 4, 1, field=lk_field2)
amps3 = lifshitz_kosevich(temps, 1.3, 1, field=lk_field2)
amps4 = lifshitz_kosevich(temps, 2.7, 1, field=lk_field2)

ax.plot(temps, amps1, label='1kT')
ax.plot(temps, amps2, label='7kT')
ax.plot(temps, amps3, label='2kT')
ax.plot(temps, amps4, label='5.5kT')
ax.axvline(0.3)
ax.axvline(1)
ax.axvline(2.3)
ax.axvline(4)
ax.axvline(7)

ax.set_xlim(0,10)


ax.grid()
ax.legend()

plt.show()






