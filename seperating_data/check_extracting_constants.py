import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
from tools.MakePlot import MakePlot
from matplotlib.offsetbox import AnchoredText

# This is code to verify the methods used for extracting regions with constant magnetic fields, and temperatures.
#
# As the magnetic field is changing, we find the regions where the absolute value of the
# derivative is below a certain threshold (i.e. 0.001). We then based on the sign of the derivative deterine if it is
# increasing or decreasing.
#
# To determine the regions of constant temperature, a similar method is employed. The temperature is increased in a
# step-wise manner, so we find regions where the derivative is on the order of 1, and use consecutive regions to
# classify constant temperature regions.
#
# This code has plots to verify the method, and should be consulted when new data is present.


main_path = '/Users/npopiel/Documents/MPhil/Data/data_csvs_cleaned/'

file_names = ['FeSb2_data1.csv',
              'FeSb2_data2.csv']

df = pd.read_csv(main_path+file_names[0])

df, locs=extract_stepwise_peaks(df,'temp','temp_flag','const_temp_')

# This part of the code ensure the threshold I set for classifying the derivative is correct.
# First uncomment the plotting of the temperature. Ensure it is stepwise.
# Then uncomment the derivative of the temperature to see the small peaks line up where the steps are
# then uncomment the vertical lines to make sure they line up with the derivatives and the steps.
# run each one seperately

# If it doesn't line up, alter the parameter threshold in extract_stepwise_peaks()
#
# fig, axs = MakePlot().create()
# plt.plot(df.temp)
# #plt.plot(np.diff(df.temp))
# for peak in locs: plt.axvline(peak)
# plt.title('Verification of Extracting Constant Temperature Regions')
# plt.show()

# Now time to check the magnetic field!
# Since it is very large and is constantly changing, we need to index a specific region for visual purposes -- say 10000
# points? You can change the points with the variable num_pts_b_vis
# Comment out the derivative plot to ensure everything works!

num_pts_b_vis = 10000

df = extract_changing_field(df, col_name='b_field', new_col_name='b_flag',root_flag_marker='b')

sigma=13.95

def select_values_near_n(array,n=13.95,cond='max'):
    lst = []
    c=0
    for ind,el in enumerate(array):
        if cond=='max':
            if el>n:
                lst.append(c)
        else:
            if el<n:
                lst.append(c)
        c+=1

    return lst

locs_of_maxima = select_values_near_n(df['b_field'])

ordered_bs = np.argsort(df.b_field)

fig2, axs2 = MakePlot().create()
plt.plot(df.b_field)
for loc in locs_of_maxima:
    plt.axvline(loc,c='red')
#plt.plot(np.diff(df.b_field[:num_pts_b_vis]))
plt.title('Verification of Extracting Constant Temperature Regions')
plt.show()
print()


