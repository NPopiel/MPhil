import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
import seaborn as sns
from tools.MakePlot import *


def extract_n_remove_hall(dat):
    field = np.array(dat.field)
    resistance = 1000000 * np.array(dat.voltage)
    pos_locs, neg_locs = [], []
    for ind, val in enumerate(field):
        if val >= 0:
            pos_locs.append(ind)
        else:
            neg_locs.append(ind)
    if len(pos_locs)>=len(neg_locs):
        pos_locs = pos_locs[:len(neg_locs)]
    else:
        neg_locs = neg_locs[:len(pos_locs)]

    pos_field_res = resistance[pos_locs]
    neg_field_res = resistance[neg_locs]

    no_hall = (pos_field_res + neg_field_res) / 2
    ya_hall = (pos_field_res - neg_field_res) / 2

    return no_hall, ya_hall, field[pos_locs]

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/VT49_1p8K_lockin_rotator_1uA/'

filenames = ['VT49_lockin_rotator_deg_0.0.csv',
             'VT49_lockin_rotator_deg_30.0.csv',
             'VT49_lockin_rotator_deg_45.0.csv',
             'VT49_lockin_rotator_deg_60.0.csv',
             'VT49_lockin_rotator_deg_90.0.csv',
             'VT49_lockin_rotator_deg_120.0.csv',
             'VT49_lockin_rotator_deg_135.0.csv',
             'VT49_lockin_rotator_deg_150.0.csv',
             'VT49_lockin_rotator_deg_180.0.csv']

dat_lst = []
name_lst = []
for filename in filenames:
    dat = pd.read_csv(main_path+filename)

    r_no_hall, r_hall, field = extract_n_remove_hall(dat)
    small_df = pd.DataFrame({r"Magnetic Field (T)":field,
                             r"Transverse Resistance $(\Omega)$":r_hall,
                             r"Longitudinal Resistance $(\Omega)$": r_no_hall,
                             r"Rotation Angle (degrees)":[filename.split('.')[-3].split('_')[-1]]*len(field)})
    dat_lst.append(small_df)

df = pd.concat(dat_lst)


fig, ax = MakePlot(nrows=1,ncols=2).create()

sns.lineplot(x="Magnetic Field (T)", y=r"Longitudinal Resistance $(\Omega)$", hue='Rotation Angle (degrees)', data=df,ax=ax[0],legend=False)
sns.lineplot(x="Magnetic Field (T)", y=r"Transverse Resistance $(\Omega)$", hue='Rotation Angle (degrees)', data=df,ax=ax[1])
plt.suptitle('VT49 Transverse and Longitudinal Resistance by Field and Angle (1.8 K)')
plt.savefig(main_path+'resistance_all_same_ax.png',dpi=600)
#plt.close()
plt.show()



