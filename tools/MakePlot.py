import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib



class MakePlot():

    def __init__(self, nrows=1,ncols=1,figsize=(16,9)):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize

    def create(self):
        matplotlib.use('TkAgg')
        plt.rc('font', family='arial', size=14)
        #sns.set_context('paper')
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols,figsize=self.figsize)
        plt.interactive(False)
        return fig, axs