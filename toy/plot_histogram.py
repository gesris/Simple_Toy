#import numpy as np
#np.random.seed(1234)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import global_variables
import pickle

def makeplot(bins_for_plots_middle, s, b, bins_for_plots, n, signal_significance, background_significance, border):
    plt.figure(figsize=(7, 6))
    plt.hist(bins_for_plots_middle, weights= [s[1], s[0]], bins= bins_for_plots, histtype="step", lw=2, label="Signal")
    plt.hist(bins_for_plots_middle, weights= [b[1], b[0]], bins= bins_for_plots, histtype="step", lw=2, label="Backgorund")
    plt.legend(loc= "lower center")
    #plt.text(0.56, 7000, "Signal side of decision boundary", fontsize= 8)
    #plt.text(0.03, 7000, "Background side of decision boundary", fontsize= 8)
    plt.title("Background Significance: {:.2f},   Signal Significance: {:.2f}".format(background_significance, signal_significance))
    plt.xlabel("Projection with maximum likelihood decision boundary at {}".format(border))
    plt.ylabel("# Events")
    plt.axvline(x = border, ymin= 0, ymax= max(n[0], n[1]), color="r", linestyle= "dashed", lw=2)
    #plt.text(1.1, 0, "\t \t Signal \t Background \n \n Mean: \t {} \t     {} \n n: \t    {} \t        {} \n Cov: \t  {}    {} \n \n Decision boundary: \t {}".format(mean[0], mean[1], n[0], n[1], cov[0], cov[1], border).expandtabs(), fontsize= 12, bbox=dict(facecolor='grey', alpha=0.3))
    #plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/significance_hist_max_like_{}.png".format(picture_index), bbox_inches = "tight")
    plt.show()


# plot histogram with maximum likelihood desicion boundary
_, _, _, _, s, b = pickle.load(open("./toy/initial_data.pickle", "rb"))
signal_significance = s[0] / np.sqrt(s[0] + b[0])      # index 0 = signal side
background_significance = b[1] / np.sqrt(s[1] + b[1])  # index 1 = background side

makeplot(global_variables.bins_for_plots_middle, s, b, global_variables.bins_for_plots, global_variables.n, signal_significance, background_significance, global_variables.border)


# plot histogram after training
s, b = pickle.load(open("./toy/training_data.pickle", "rb"))
opt_sig_significance = s[1] / np.sqrt(s[1] + b[1])
opt_bkg_significance = b[0] / np.sqrt(s[0] + b[0])
opt_total_significance = opt_sig_significance + opt_bkg_significance

makeplot(global_variables.bins_for_plots_middle, s, b, global_variables.bins_for_plots, global_variables.n, opt_sig_significance, opt_bkg_significance, global_variables.border)

