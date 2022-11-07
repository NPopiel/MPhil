import numpy as np
import matplotlib.pyplot as plt
from tools.utils import *
from tools.MakePlot import *

'''
FIles

/Users/npopiel/Desktop/41T/JP_UMD_March2022.033.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.032.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.031.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.030.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.029.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.028.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.027.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.026.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.025.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.024.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.023.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.022.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.021.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.020.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.019.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.018.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.017.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.016.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.015.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.014.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.013.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.012.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.011.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.010.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.009.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.008.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.007.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.006.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.005.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.004.txt
/Users/npopiel/Desktop/41T/Instruments 2022-03-14 CELL06.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.003.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.002.txt
/Users/npopiel/Desktop/41T/JP_UMD_March2022.001

'''

files = ['/Users/npopiel/Desktop/41T/JP_UMD_March2022.033.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.032.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.031.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.030.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.029.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.028.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.027.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.026.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.025.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.024.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.023.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.022.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.021.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.020.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.019.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.018.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.017.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.016.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.015.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.014.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.013.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.012.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.011.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.010.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.009.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.008.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.007.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.006.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.005.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.004.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.003.txt',
'/Users/npopiel/Desktop/41T/JP_UMD_March2022.002.txt']


files = np.array(files)[::-1]

ind = 2
rel_columns = ['Field_007',	'ProbeT_007', 'FeSi_R_007', 'FeSb2_cant_X_007',	'FeSb2_cant_Y_007',	'FeSi_cant_X_007',	'FeSi_cant_Y_007',	'di_HY_007',	'dv_HY_007']

line_to_open = 9

f1, a1 = MakePlot().create()
f2, a2 = MakePlot().create()
f3, a3 = MakePlot().create()
f4, a4 = MakePlot().create()
f5, a5 = MakePlot().create()
f6, a6 = MakePlot().create()
f7, a7 = MakePlot().create()
f8, a8 = MakePlot().create()

for i, file in enumerate(files):
    dat = pd.read_csv(file, delimiter='\t', skiprows=6)



    if ind < 10:
        str_ind = '0'+str(ind)
    else:
        str_ind = str(ind)

    B = np.array(dat['Field_0'+str_ind])
    T = np.array(dat['ProbeT_0'+str_ind])
    R_u = np.array(dat['UTe2_Mont_R_0'+str_ind])
    R_fesi = np.array(dat['FeSi_R_0'+str_ind])
    R_fesb2 = np.array(dat['FeSb2_R_0'+str_ind])
    Cant_X_fesb2 = np.array(dat['FeSb2_cant_X_0'+str_ind])
    Cant_Y_fesb2 = np.array(dat['FeSb2_cant_Y_0'+str_ind])
    Cant_X_fesi = np.array(dat['FeSi_cant_X_0'+str_ind])
    Cant_Y_fesi = np.array(dat['FeSi_cant_Y_0'+str_ind])
    di_hy = np.array(dat['di_HY_0'+str_ind])
    dv_hy = np.array(dat['dv_HY_0'+str_ind])

    ind+=1

    a1.plot(B,R_fesi,c=plt.cm.jet(i/len(files)), linewidth=2)
    a2.plot(B,R_fesb2,c=plt.cm.jet(i/len(files)), linewidth=2)
    a3.plot(B,Cant_X_fesb2,c=plt.cm.jet(i/len(files)), linewidth=2)
    a4.plot(B,Cant_Y_fesb2,c=plt.cm.jet(i/len(files)), linewidth=2)
    a5.plot(B,Cant_X_fesi,c=plt.cm.jet(i/len(files)), linewidth=2)
    a6.plot(B,Cant_Y_fesi,c=plt.cm.jet(i/len(files)), linewidth=2)
    a7.plot(dv_hy, di_hy,c=plt.cm.jet(i/len(files)), linewidth=2)
    if i !=0:
        a8.plot(B, R_u,c=plt.cm.jet(i/len(files)), linewidth=2)

publication_plot(a1, 'Magnetic Field (T)', 'R FeSi')
publication_plot(a2, 'Magnetic Field (T)', 'R FeSb2')
publication_plot(a3, 'Magnetic Field (T)', 'Cant X FeSb2')
publication_plot(a4, 'Magnetic Field (T)', 'Cant Y FeSb2')
publication_plot(a5, 'Magnetic Field (T)', 'Cant X FeSi')
publication_plot(a6, 'Magnetic Field (T)', 'Cant Y FeSi')
publication_plot(a7, 'dV', 'dI')
publication_plot(a8, 'B', 'R_u')

f1.show()
f2.show()
f3.show()
f4.show()
f5.show()
f6.show()
f7.show()

f8.show()


print('he')