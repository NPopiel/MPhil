import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib



class MakePlot():

    def __init__(self, nrows=1,ncols=1,figsize=(9,12)):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize

    def create(self):
        matplotlib.use('TkAgg')
        plt.rc('font', family='serif')
        sns.set_context('paper')
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols)
        plt.interactive(False)
        return fig, axs