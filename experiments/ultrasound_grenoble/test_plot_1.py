import pandas as pd
from tools.utils import *
from tools.MakePlot import *
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

import seaborn as sns

def remove_copies(x,y):
    if_copy_y = y[1:] != y[:-1]
    if_copy_x = x[1:] != x[:-1]
    if_copy = if_copy_y & if_copy_x
    if_copy = if_copy[1:] & if_copy[:-1]
    if_copy = np.concatenate([[True], if_copy, [True]])
    return x[if_copy] , y[if_copy]

'''
files to use for first run:

.001 sees transition for first time
#.008 for other sample -- no transition
.037 has transition
.048 some high field artifact

Column 1 -- Time I think?
Column 2 -- Temperature
Column 3 -- 
Column 4 -- B Field
column 5 -- Frequency
Column 6 -- Echo 1?
Column 7 -- Echo 2?



'''

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Grenoble2021/FeSb2_Fev21/FeSb2_L111_Fev21/'


data_1 = pd.read_csv(main_path+'FeSb2_L111_Fev21.002',sep='\t')
data_1.columns = ['time', 'Temperature', 'Kmin', 'Magnetic Field','Frequency','Velocity', 'Amplitude', '']
y = np.array(data_1['Velocity'])
x = np.array(data_1['Magnetic Field'])

x, y = remove_copies(x, y)

x = median_filter(x,5)
y = median_filter(y,5)


fig, ax = MakePlot().create()

plt.plot(x,y)
plt.show()

y = np.array(data_1['Kmin'])
x = np.array(data_1['Magnetic Field'])

x, y = remove_copies(x, y)

x = median_filter(x,5)
y = median_filter(y,5)


fig, ax = MakePlot().create()

plt.plot(x,y)
plt.show()

y = np.array(data_1['Amplitude'])
x = np.array(data_1['Magnetic Field'])

x, y = remove_copies(x, y)

x = median_filter(x,5)
y = median_filter(y,5)


fig, ax = MakePlot().create()

plt.plot(x,y)
plt.show()