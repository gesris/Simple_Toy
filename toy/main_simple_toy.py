import numpy as np
np.random.seed(1234)

import matplotlib.pyplot as plt

import global_variables
import create_data
import plot_histogram
import plot_maximum_likelihood
import train_data

hist_signal, hist_background, lin_func, graph_limit, s, b = create_data.create_initial_data(global_variables.mean, global_variables.cov, global_variables.n)

# approximation of significance
signal_significance = s[0] / np.sqrt(s[0] + b[0])      # index 0 = signal side
background_significance = b[1] / np.sqrt(s[1] + b[1])  # index 1 = background side

# plot ML
plot_maximum_likelihood.makeplot(hist_signal, hist_background, lin_func, graph_limit)

# plot histogram
plot_histogram.makeplot(global_variables.bins_for_plots_middle, s, b, global_variables.bins_for_plots, global_variables.n, signal_significance, background_significance, global_variables.border)

# data after training
s, b, opt_sig_significance, opt_bkg_significance, opt_total_significance, f_sig, x_sig = train_data.training()

# plot histogram after training
plot_histogram.makeplot(global_variables.bins_for_plots_middle, s, b, global_variables.bins_for_plots, global_variables.n, signal_significance, background_significance, global_variables.border)

