import pandas as pd
from tools.utils import *
from tools.MakePlot import *
import matplotlib.pyplot as plt

import seaborn as sns

main_path = '/Users/npopiel/Documents/MPhil/Data/step_data/'

df = pd.read_csv(main_path+'steplocs.csv')

pos_up_800 = np.array([[3.52791, 0.739499329, 1.75, 2.94153, 4.66094],
                       [4.67113, 0.721376343, 2, 3.80096, 5.51495],
                       [5.52707, 0.616883994, 2.25, 4.6564, 6.37178],
                       [6.66972, 0.478612315, 2.5, 5.79871, 7.23241],
                       [7.52316, 0.442405919, 2.75, 6.9417, 8.6558],
                       [9.52847, 0.362756024, 3, 8.65533, 10.08841]])

pos_down_800 = np.array([[4.18747, 0.655087838, 1.75, 5.05902, 3.33168],
                [4.47739, 0.729703891, 2, 5.62787, 3.91325],
                [5.61582, 0.619737555, 2.25, 6.48823, 4.77349],
                [6.75864, 0.556096253, 2.5, 7.62832, 5.91386],
                [7.9038, 0.471782937, 2.75, 8.77214, 7.34588],
                [9.6158, 0.400373507, 3, 10.48877, 8.77085]])

neg_up_800 = np.array([[-3.61613, 0.82668116, 1.75, -5.05601, -3.05567],
                       [-4.47252, 0.741417642, 2, -5.6287, - 3.91484],
                       [-5.32927, 0.708840738, 2.25, -6.77039, -4.771],
                       [-6.76334, 0.487328884, 2.5, -7.91421, -6.19328],
                       [-8.18947, 0.481050777, 2.75, -9.05713, -7.62703],
                       [-9.90225, 0.422136315, 3, -10.77181, -9.3418]])
neg_down_800 = np.array([[-4.09888, 0.767675234, 1.75, -2.65838, -4.65787],
                         [-4.38417, 0.766226871, 2, -3.51289, -5.51554],
                         [-5.52681, 0.650277663, 2.25, -4.65551, -6.37218],
                         [-6.67018, 0.563551589, 2.5, -5.80117, -7.51558],
                         [-8.09849, 0.494985583, 2.75, -7.23006, -8.94343],
                         [-9.81255, 0.395527905, 3, -8.94037, -10.65845]])

pos_up_900 = np.array([[8.95573, 0.447247355, 1.75, 8.08616, 9.80149],
                       [9.52829, 0.405323512,    2, 8.94385, 10.65506],
                       [10.66995, 0.323885376, 2.25, 10.08628, 11.51821]])

pos_down_900 = np.array([[9.04489, 0.389427354, 1.75, 9.91378, 8.48221],
                        [9.90247, 0.34935102, 2, 10.77168, 9.33577],
                        [11.04585, 0.341579192, 2.25, 11.91432, 10.48632]])

neg_up_900 = np.array([[-9.33093, 0.427731652, 1.75, -10.20046, -8.76451],
                       [-10.18745, 0.396842746, 2, -11.34541, -9.62865],
                       [-11.3301, 0.331772202, 2.25, -12.48784, -10.77173]])

neg_down_900 = np.array([[-8.95647, 0.485933516, 1.75, -8.36886, -10.0867],
                         [-10.38452, 0.37910861, 2, -9.22583, -10.94365],
                         [-11.24178, 0.359077151, 2.25, -10.37269, -12.08658]])

temps_800 = pos_down_800[:,2]
dG_pos_800 = np.array([pos_up_800[:,1], pos_down_800[:,1]])
field_pos_800 = np.array([pos_up_800[:,0], pos_down_800[:,0]])
B1_pos_800 = np.array([pos_up_800[:,3], pos_down_800[:,3]])
B2_pos_800 = np.array([pos_up_800[:,4], pos_down_800[:,4]])

dG_neg_800 = np.array([neg_up_800[:,1], neg_down_800[:,1]])
field_neg_800 = np.array([neg_up_800[:,0], neg_down_800[:,0]])
B1_neg_800 = np.array([neg_up_800[:,3], neg_down_800[:,3]])
B2_neg_800 = np.array([neg_up_800[:,4], neg_down_800[:,4]])

dG_800 = np.mean(np.array([pos_up_800[:,1], pos_down_800[:,1],neg_up_800[:,1], neg_down_800[:,1]]),axis=0)
std_dG_800 = np.std(np.array([pos_up_800[:,1], pos_down_800[:,1],neg_up_800[:,1], neg_down_800[:,1]]),axis=0)

field_800 = np.mean(np.array([pos_up_800[:,0], pos_down_800[:,0],neg_up_800[:,0], neg_down_800[:,0]]),axis=0)
std_field_800 = np.std(np.array([pos_up_800[:,0], pos_down_800[:,0],neg_up_800[:,0], neg_down_800[:,0]]),axis=0)

B1_800 = np.mean(np.array([pos_up_800[:,3], pos_down_800[:,3],neg_up_800[:,3], neg_down_800[:,3]]),axis=0)
std_B2_800 = np.std(np.array([pos_up_800[:,3], pos_down_800[:,3],neg_up_800[:,3], neg_down_800[:,3]]),axis=0)

B2_800 = np.mean(np.array([pos_up_800[:,4], pos_down_800[:,4],neg_up_800[:,4], neg_down_800[:,4]]),axis=0)
std_B2_800 = np.std(np.array([pos_up_800[:,4], pos_down_800[:,4],neg_up_800[:,4], neg_down_800[:,4]]),axis=0)

temps_900 = pos_down_900[:,2]
dG_pos_900 = np.array([pos_up_900[:,1], pos_down_900[:,1]])
field_pos_900 = np.array([pos_up_900[:,0], pos_down_900[:,0]])
B1_pos_900 = np.array([pos_up_900[:,3], pos_down_900[:,3]])
B2_pos_900 = np.array([pos_up_900[:,4], pos_down_900[:,4]])

dG_neg_900 = np.array([neg_up_900[:,1], neg_down_900[:,1]])
field_neg_900 = np.array([neg_up_900[:,0], neg_down_900[:,0]])
B1_neg_900 = np.array([neg_up_900[:,3], neg_down_900[:,3]])
B2_neg_900 = np.array([neg_up_900[:,4], neg_down_900[:,4]])

dG_900 = np.mean(np.array([pos_up_900[:,1], pos_down_900[:,1],neg_up_900[:,1], neg_down_900[:,1]]),axis=0)
std_dG_900 = np.std(np.array([pos_up_900[:,1], pos_down_900[:,1],neg_up_900[:,1], neg_down_900[:,1]]),axis=0)

field_900 = np.mean(np.array([pos_up_900[:,0], pos_down_900[:,0],neg_up_900[:,0], neg_down_900[:,0]]),axis=0)
std_field_900 = np.std(np.array([pos_up_900[:,0], pos_down_900[:,0],neg_up_900[:,0], neg_down_900[:,0]]),axis=0)

B1_900 = np.mean(np.array([pos_up_900[:,3], pos_down_900[:,3],neg_up_900[:,3], neg_down_900[:,3]]),axis=0)
std_B2_900 = np.std(np.array([pos_up_900[:,3], pos_down_900[:,3],neg_up_900[:,3], neg_down_900[:,3]]),axis=0)

B2_900 = np.mean(np.array([pos_up_900[:,4], pos_down_900[:,4],neg_up_900[:,4], neg_down_900[:,4]]),axis=0)
std_B2_900 = np.std(np.array([pos_up_900[:,4], pos_down_900[:,4],neg_up_900[:,4], neg_down_900[:,4]]),axis=0)


y_dG_800 = np.poly1d(np.polyfit(temps_800,dG_800,1))
y_dG_900 = np.poly1d(np.polyfit(temps_900,dG_900,1))
x = np.linspace(0,4,100)

plt.style.use('seaborn-paper')
sns.set_context('paper')


fig, ax = MakePlot().create()

sns.set_palette('husl', n_colors=4)

sns.scatterplot(x=temps_800,y=np.mean(dG_pos_800,axis=0), s=100, label=r'Positive Field $800 \mu A$')
ax.errorbar(temps_800,np.mean(dG_pos_800,axis=0), np.std(dG_pos_800,axis=0),fmt='o')
sns.scatterplot(x=temps_800,y=np.mean(dG_neg_800,axis=0),s=100,label=r'Negative Field $800 \mu A$')
ax.errorbar(temps_800,np.mean(dG_neg_800,axis=0),np.std(dG_neg_800,axis=0),fmt='o')
plt.plot(x,y_dG_800(x),'-k',label=r'$800 \mu A$ Fitted Line')

sns.scatterplot(x=temps_900,y=np.mean(dG_pos_900,axis=0), s=100,label=r'Positive Field $900 \mu A$')
ax.errorbar(temps_900,np.mean(dG_pos_900,axis=0),np.std(dG_pos_900,axis=0),fmt='o')
sns.scatterplot(x=temps_900,y=np.mean(dG_neg_900,axis=0), s=100,label=r'Negative Field $900 \mu A$')
ax.errorbar(temps_900,np.mean(dG_neg_900,axis=0),np.std(dG_neg_900,axis=0),fmt='o')
plt.plot(x,y_dG_900(x),'--k',label=r'$900 \mu A$ Fitted Line')

plt.title(r'Maxima in Conductance by Temperature', fontsize=18)
ax.set_xlabel(r'Temperature $(K)$', fontsize=14)
ax.set_ylabel(r'$\delta G $ $(\frac{2 e^2}{h})$', fontsize=14)
ax.set_xlim()
ax.set_ylim()
ax.minorticks_on()
#ax.axis('equal')
ax.tick_params('both', which='both', direction='in',
               bottom=True, top=True, left=True, right=True)

#ax.ticklabel_format(style='sci', axis='y')  # , scilimits=(0, 0)

# handles, labels = plt.gca().get_legend_handles_labels()
#
# labels, ids = np.unique(labels, return_index=True)
# handles = [handles[i] for i in ids]
plt.legend(title='Field and Current', loc='best',frameon=True, fancybox=False, edgecolor='k', framealpha=1, borderpad=1)


plt.show()

y_dG_800_pos = np.poly1d(np.polyfit(np.mean(field_pos_800,axis=0),np.mean(dG_pos_800, axis=0),1))
y_dG_800_neg = np.poly1d(np.polyfit(np.mean(field_neg_800,axis=0),np.mean(dG_neg_800, axis=0),1))
y_dG_900_pos = np.poly1d(np.polyfit(np.mean(field_pos_900,axis=0),np.mean(dG_pos_900, axis=0),1))
y_dG_900_neg = np.poly1d(np.polyfit(np.mean(field_neg_900,axis=0),np.mean(dG_neg_900, axis=0),1))
x1 = np.linspace(0,14,100)
x2 = np.linspace(-14,0,100)
plt.style.use('seaborn-paper')
sns.set_context('paper')


fig, ax = MakePlot().create()

sns.set_palette('husl', n_colors=4)

sns.scatterplot(x=np.mean(field_pos_800,axis=0),y=np.mean(dG_pos_800,axis=0), s=100, label=r'Positive Field $800 \mu A$')
ax.errorbar(np.mean(field_pos_800,axis=0),np.mean(dG_pos_800,axis=0), np.std(dG_pos_800,axis=0),fmt='o')
sns.scatterplot(x=np.mean(field_neg_800,axis=0),y=np.mean(dG_neg_800,axis=0),s=100,label=r'Negative Field $800 \mu A$')
ax.errorbar(np.mean(field_neg_800,axis=0),np.mean(dG_neg_800,axis=0),np.std(dG_neg_800,axis=0),fmt='o')
plt.plot(x1,y_dG_800_pos(x1),'-k',label=r'$800 \mu A$ Positive Field')
plt.plot(x2,y_dG_800_neg(x2),'--k',label=r'$800 \mu A$ Negative Field')

sns.scatterplot(x=np.mean(field_pos_900,axis=0),y=np.mean(dG_pos_900,axis=0), s=100,label=r'Positive Field $900 \mu A$')
ax.errorbar(np.mean(field_pos_900,axis=0),np.mean(dG_pos_900,axis=0),np.std(dG_pos_900,axis=0),fmt='o')
sns.scatterplot(x=np.mean(field_neg_900,axis=0),y=np.mean(dG_neg_900,axis=0), s=100,label=r'Negative Field $900 \mu A$')
ax.errorbar(np.mean(field_neg_900,axis=0),np.mean(dG_neg_900,axis=0),np.std(dG_neg_900,axis=0),fmt='o')
plt.plot(x1,y_dG_900_pos(x1),'-r',label=r'$900 \mu A$ Positive Field')
plt.plot(x2,y_dG_900_neg(x2),'--r',label=r'$900 \mu A$ Negative Field')

plt.title(r'Maxima in Conductance by Field', fontsize=18)
ax.set_xlabel(r'Magnetic Field $(T)$', fontsize=14)
ax.set_ylabel(r'$\delta G $ $(\frac{2 e^2}{h})$', fontsize=14)
ax.set_xlim()
ax.set_ylim()
ax.minorticks_on()
#ax.axis('equal')
ax.tick_params('both', which='both', direction='in',
               bottom=True, top=True, left=True, right=True)

#ax.ticklabel_format(style='sci', axis='y')  # , scilimits=(0, 0)

# handles, labels = plt.gca().get_legend_handles_labels()
#
# labels, ids = np.unique(labels, return_index=True)
# handles = [handles[i] for i in ids]
plt.legend(title='Field and Current', loc='best',frameon=True, fancybox=False, edgecolor='k', framealpha=1, borderpad=1)


plt.show()

y_dG_800 = np.poly1d(np.polyfit(temps_800,dG_800,1))
y_dG_900 = np.poly1d(np.polyfit(temps_900,dG_900,1))
x = np.linspace(0,4,100)

plt.style.use('seaborn-paper')
sns.set_context('paper')


fig, ax = MakePlot().create()

sns.set_palette('husl', n_colors=4)

sns.scatterplot(y=temps_800,x=np.mean(field_pos_800,axis=0), s=100, label=r'Positive Field $800 \mu A$')

#ax.errorbar(np.mean(field_pos_800,axis=0),temps_800, np.std(field_pos_800,axis=0),fmt='o')
sns.scatterplot(y=temps_800,x=np.mean(field_neg_800,axis=0),s=100,label=r'Negative Field $800 \mu A$')
#ax.errorbar(np.mean(field_neg_800,axis=0),temps_800,np.std(field_neg_800,axis=0),fmt='o')
#plt.plot(x,y_dG_800(x),'-k',label=r'$800 \mu A$ Fitted Line')

sns.scatterplot(y=temps_900,x=np.mean(field_pos_900,axis=0), s=100,label=r'Positive Field $900 \mu A$')
#ax.errorbar(np.mean(field_pos_900,axis=0),temps_900,np.std(field_pos_900,axis=0),fmt='o')
sns.scatterplot(y=temps_900,x=np.mean(field_neg_900,axis=0), s=100,label=r'Negative Field $900 \mu A$')
#ax.errorbar(np.mean(field_neg_900,axis=0),temps_900,np.std(field_neg_900,axis=0),fmt='o')
#plt.plot(x,y_dG_900(x),'--k',label=r'$900 \mu A$ Fitted Line')

plt.title(r'Temperature by Step Field ', fontsize=18)
ax.set_ylabel(r'Temperature $(K)$', fontsize=14)
ax.set_xlabel(r'$\mu _0 H^* $ $(T)$', fontsize=14)
ax.set_xlim()
ax.set_ylim()
ax.minorticks_on()
#ax.axis('equal')
ax.tick_params('both', which='both', direction='in',
               bottom=True, top=True, left=True, right=True)

plt.show()






