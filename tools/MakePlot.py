import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib



class MakePlot():

    def __init__(self, nrows=1,ncols=1,figsize=(16,9),gs=False):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.gs = gs

    def create(self):
        matplotlib.use('TkAgg')
        plt.rc('font', family='arial', size=14)
        #sns.set_context('paper')
        if not self.gs:
            fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols,figsize=self.figsize)
        else:
            fig = plt.figure(figsize=self.figsize)
            axs = None
        plt.interactive(False)
        return fig, axs