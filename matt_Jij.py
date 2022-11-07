import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

def to_normalize(J, netx=False):
    if netx:
        J = nx.to_numpy_array(J)
    max_J = np.max(J)
    min_J = np.min(J)

    if max_J >= 0 and max_J <= 1 and min_J >= 0 and min_J <= 1:
        if netx:
            return nx.to_networkx_graph(J)
        else:
            return J
    else:
        if netx:
            return nx.to_networkx_graph(J / max_J)
        else:
            return J / max_J


def load_matrix(file):
    import numpy as np
    import scipy.io

    extension = file.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(file, delimiter=',')
    elif str(extension) == 'npy':
        return np.load(file)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(file)
    elif str(extension) == 'npz':
        return np.load(file)


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def file_exists(filename):
    import os
    exists = os.path.isfile(filename)
    if exists:
        return True
    else:
        return False
networks = ["Aud","CO","CP","DMN","Dorsal","FP","RS","SMHand","SMMouth","Ventral","Visual"]

main_path = '/Users/npopiel/Downloads/HCP_Average/'

outputPath = main_path + 'figs/'

makedir(outputPath)

fig, axs = plt.subplots(3,4)

count = 0

for ax in axs.flat:

    if count >= len(networks):
        break

    network = networks[count]
    input_path = main_path + network + "/Jij_avg.csv"

    J = to_normalize(load_matrix(input_path))

    index = networks.index(network)

    heatmap = ax.imshow(J,vmin = 0,vmax = 1, cmap='plasma')
    ax.set_title(network)

    count += 1

fig.delaxes(axs[2,3])
plt.tight_layout()

plt.savefig(outputPath+'fig.pdf', dpi = 1000)
plt.show()
