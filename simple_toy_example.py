import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.set_random_seed(1234)
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
import matplotlib
import myutils

####
#### Global Variables
####

def normal_distribution(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)

#mean = [[- 0.5, 0.5], [0.5, -0.5]]           # [Signal, Background]
sx = float(input("Signal Mean X: "))
sy = float(input("Signal Mean Y: "))
bx = float(input("Background Mean X: "))
by = float(input("Background Mean Y: "))
n_s = int(input("Number of Signal Events: "))
n_b = int(input("Number of Background Events: "))

mean = [[sx, sy], [bx, by]]
cov = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]  # [Signal, Background]
n = [n_s, n_b]                          # [Signal, Background]

bins_for_plots = [0.0, 0.5, 1.0]
bins_for_plots_middle = []
# in this case the middle of bins is [0.25, 0.75]
for i in range(0, len(bins_for_plots) - 1):
    bins_for_plots_middle.append(bins_for_plots[i] + (bins_for_plots[i + 1] - bins_for_plots[i]) / 2)

border = bins_for_plots[1]

picture_index = "differentmean3"


####
#### Create 2D Dataset
####

# multivariate_normal gives 2 separate lists with events-coordinates following the normal distribution
# here the two lists represent x- and y-direction
signal_raw = normal_distribution(mean[0], cov[0], n[0])
background_raw = normal_distribution(mean[1], cov[1], n[1])

signal = signal_raw.T
background = background_raw.T

number_of_bins = 10
scale = 3   # bin width = scale / number of bins
bins = np.linspace(-scale, scale, number_of_bins)

signal_data_x = signal[1]
signal_data_y = signal[0]
background_data_x = background[1]
background_data_y = background[0]

hist_signal = np.histogram2d(signal_data_x, signal_data_y, bins= [bins,bins])
hist_background = np.histogram2d(background_data_x, background_data_y, bins= [bins, bins])

# draw max likelihood decision boundry for two identical signals dnamically

vec = []
for i in range(0, len(mean[0])):
    vec.append(mean[1][i] - mean[0][i])
perpendicular_vec = [-1 * vec[1], vec[0]]
magnitude = perpendicular_vec[0] / perpendicular_vec[1]
bias = [mean[0][0] + (vec[0] / 2) * (n[0] / n[1]), mean[0][1] + (vec[1] / 2) * (n[0] / n[1])]

def linear_function(x, magnitude, bias):
    return magnitude * (x - bias[1]) + bias[0]

graph_limit = [-10, 10]
lin_func = linear_function(np.linspace(graph_limit[0],graph_limit[1],200), magnitude, bias)

#plt.figure(figsize=(6,6))
#plt.plot([lin_func[0], lin_func[-1]], [graph_limit[0], graph_limit[1]])
#plt.plot([mean[0][0], mean[1][0]], [mean[0][1], mean[1][1]])
#plt.plot(mean[0][0],mean[0][1],'ro')
#plt.plot(mean[1][0], mean[1][1], 'ro')
#plt.xlim(-3, 3)
#plt.ylim(-3, 3)
#plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/magnitude_bias_plot_{}.png".format(picture_index), bbox_inches = "tight")
#plt.show()


####
#### Plot 2D Histograms
####

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
#plt.show()

####
#### Plot significance from maximum likelihood via Histogram
####


# Betrachte alle Punkte, die auf der Seite des Signals liegen, indem alle Events gezählt werden,
# bei denen der Y-Wert > X-Wert ist -> aus Maximum Likelihood
#magnitude * (x - bias[1]) + bias[0]
def num_sig(x):
    #return [np.sum(x[1] > magnitude * (x[0] - bias[1]) + bias[0]), np.sum(x[1] < magnitude * (x[0] - bias[1]) + bias[0])]
    #return [np.sum(x[1] > x[0]), np.sum(x[1] < x[0])]
    if magnitude == 0:
        return [np.sum((x[0] - bias[0]) < 0), np.sum((x[0] - bias[0]) > 0)]
    else:
        return [np.sum(x[1] > (1 / magnitude) * (x[0] - bias[0]) + bias[1]), np.sum(x[1] < (1 / magnitude) * (x[0] - bias[0]) + bias[1])]
    
s = num_sig(signal)         # Alle Signalevents
b = num_sig(background)     # Alle Bkgevents

print("Signal: {}, Background: {}".format(s, b))

# Einfache Schätzung der Signifikanz für hohe Anzahl von Events 
signal_significance = s[0] / np.sqrt(s[0] + b[0])      # index 0 = signal side
background_significance = b[1] / np.sqrt(s[1] + b[1])  # index 1 = background side

# plotting signal versus background on expected vs unexpected
# everything on signal side of maximum likelihood ist here plotted left to 0
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
#plt.show()

####
#### Input Variables for NN model in terms of signal / background events
####

# variables have to be defined before start of tf.session()!
# running multiple sessions also possible
x_sig = tf.placeholder(tf.float32, shape=[None, 2]) # x-y-coordinates of signal events
x_bkg = tf.placeholder(tf.float32, shape=[None, 2]) # x-y-coordinates of background events
f_sig = tf.sigmoid(myutils.model(x_sig, 2)) # NN output of signal
f_bkg = tf.sigmoid(myutils.model(x_bkg, 2, reuse=True)) # NN output of background

hist_sig = myutils.hist(f_sig, bins_for_plots) # Signal events represented by histogram 
hist_bkg = myutils.hist(f_bkg, bins_for_plots) # Background events represented by histogram 

# initial run of model
sess = tf.Session()
# initiate random values to global variables such as weights and biases
sess.run(tf.global_variables_initializer())

s = [hist_sig[0], hist_sig[1]]
b = [hist_bkg[0], hist_bkg[1]]
# flip significance graph with negative sign to search for minimum instead of maximum
loss = -1.0 * (s[1] / tf.sqrt(s[1] + b[1]) + b[0] / tf.sqrt(s[0] + b[0]))
minimize = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.global_variables_initializer())

# specify training
min_loss = 0
max_patience = 10
patience = max_patience

for epoch in range(300):
    loss_, _, s_ = sess.run([loss, minimize, s], feed_dict={x_sig: normal_distribution(mean[0], cov[0], n[0]), x_bkg: normal_distribution(mean[1], cov[1], n[1])})
    if loss_ > min_loss:
        patience -= 1
    else:
        min_loss = loss_
        patience = max_patience
    
    if epoch % 10 == 0 or patience == 0:
        print("Epoch {} : Loss {}".format(epoch, loss_))
        
    if patience == 0:
        print("Trigger early stopping in epoch {}.".format(epoch))
        break

####
#### Plotting results in hist after optimization of significance
####

opt_hist_sig, opt_hist_bkg = sess.run([hist_sig, hist_bkg], feed_dict={x_sig: signal_raw, x_bkg: background_raw})

print("Hist Signal {}, Hist Background {}".format(opt_hist_sig, opt_hist_bkg))

# check signal significance on signal side of decision boundary
s = [opt_hist_sig[0], opt_hist_sig[1]]
b = [opt_hist_bkg[0], opt_hist_bkg[1]]
opt_sig_significance = s[1] / np.sqrt(s[1] + b[1])

# check background significance on background side of decision boundary
opt_bkg_significance = b[0] / np.sqrt(s[0] + b[0])

opt_total_significance = opt_sig_significance + opt_bkg_significance

plt.figure(figsize=(7, 6))
plt.hist(bins_for_plots_middle, weights= [s[0], s[1]], bins= bins_for_plots, histtype="step", label="Signal", lw=2)
plt.hist(bins_for_plots_middle, weights= [b[0], b[1]], bins= bins_for_plots, histtype="step", label="Backgorund", lw=2)
plt.legend(loc= "lower center")
#plt.text(0.56, 7000, "Signal side of decision boundary", fontsize= 8)
#plt.text(0.02, 7000, "Background side of decision boundary", fontsize= 8)
plt.title("Background Significance: {:.2f},   Signal Significance: {:.2f}".format(opt_bkg_significance, opt_sig_significance))
plt.xlabel("Projection with decision boundary from NN at {}".format(border))
plt.ylabel("# Events")
plt.axvline(x = border, ymin= 0, ymax= max(n[0], n[1]), color="r", linestyle= "dashed", lw=2)
#plt.text(1.1, 0, "\t \t Signal \t Background \n \n Mean: \t {} \t     {} \n n: \t    {} \t        {} \n Cov: \t  {}    {} \n \n Decision boundary: \t {}".format(mean[0], mean[1], n[0], n[1], cov[0], cov[1], border).expandtabs(), fontsize= 12, bbox=dict(facecolor='grey', alpha=0.3))
#plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/significance_hist_NN_{}.png".format(picture_index), bbox_inches = "tight")
#plt.show()


####
#### Plot NN Function
####

length = 2000
sensibility = 0.005
b = np.linspace(-3, 3, length)
xx, yy = np.meshgrid(b, b, sparse=False)
x = np.vstack([xx.reshape((length * length)), yy.reshape((length * length))]).T
c = sess.run(f_sig, feed_dict={x_sig: x})
c = c.reshape((length, length))
# visualize decision boundary
boundary = sess.run(f_sig, feed_dict={x_sig: x})
boundary = boundary.reshape((length, length))

for y in range(0, length):
    for x in range(0, length):
        if c[y][x] > border-sensibility and c[y][x] < border+sensibility:
            boundary[y][x] = 1
        else:
            boundary[y][x] = 0

plt.figure(figsize=(7, 6))
cbar = plt.contourf(xx, yy, c + boundary, levels=np.linspace(c.min(), c.max(), 21))
plt.colorbar(cbar)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Neural network function")
#plt.text(4.8, -3, "\t \t Signal \t Background \n \n Mean: \t {} \t     {} \n n: \t    {} \t        {} \n Cov: \t  {}    {} \n \n Decision boundary: \t {}".format(mean[0], mean[1], n[0], n[1], cov[0], cov[1], border).expandtabs(), fontsize= 12, bbox=dict(facecolor='grey', alpha=0.3))
plt.tight_layout()
plt.xlim(limit[0], limit[1])
plt.ylim(limit[0], limit[1])
plt.plot([lin_func[0], lin_func[-1]], [graph_limit[0], graph_limit[1]], color="r", label="Maximum likelihood decision boundary")
#plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/NN_function_{}.png".format(picture_index), bbox_inches = "tight")
plt.show()
