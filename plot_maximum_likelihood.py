import matplotlib.pyplot as plt
import matplotlib

def makeplot(hist_signal, hist_background, lin_func, graph_limit):
    
    limit = [-3, 3]
    plt.figure(figsize=(6, 6))
    cmap_sig = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C0"] * 3)
    cmap_bkg = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C1"] * 3)
    # extent setzt den Mittelpunkt des Signals/Backgrounds auf dessen Ursprung
    plt.contour(hist_signal[0], extent= [hist_signal[1][0], hist_signal[1][-1], hist_signal[2][0] , hist_signal[2][-1]], cmap= cmap_sig)
    plt.contour(hist_background[0], extent= [hist_background[1][0], hist_background[1][-1], hist_background[2][0], hist_background[2][-1]], cmap= cmap_bkg)
    plt.plot([-999], [-999], color="C0", label="Signal")
    plt.plot([-999], [-999], color="C1", label="Background")
    plt.plot([lin_func[0], lin_func[-1]], [graph_limit[0], graph_limit[1]], color="r", label="Maximum likelihood decision boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[0], limit[1])
    #plt.text(3.5, -3, "\t \t Signal \t Background \n \n Mean: \t {} \t     {} \n n: \t    {} \t        {} \n Cov: \t  {}    {} \n \n Decision boundary: \t {}".format(mean[0], mean[1], n[0], n[1], cov[0], cov[1], border).expandtabs(), fontsize= 12, bbox=dict(facecolor='grey', alpha=0.3))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
    #plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/signal_background_plot_{}.png".format(picture_index), bbox_inches = "tight")
    plt.show()