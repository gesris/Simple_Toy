import tensorflow as tf
tf.random.set_seed(1234)
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
np.random.seed(1234)
import pickle


mean = [[- 1, -1], [1, 1], [1, 2], [1, 0]]          # [Signal, Background]
cov = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]  # [Signal, Background]
n = [100000, 100000]                          # [Signal, Background]
bins_for_plots = [0.0, 0.5, 1.0]            # 
bins_for_plots_middle = []                  # Central Point of Bin 
for i in range(0, len(bins_for_plots) - 1):
    bins_for_plots_middle.append(bins_for_plots[i] + (bins_for_plots[i + 1] - bins_for_plots[i]) / 2)
border = bins_for_plots[1]                  # Decision Boundary

def normal_distribution(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)

def hist(x, bins):
    counts = []
    # splits histogram in bins regarding their left and right edges
    # zip function puts left and right edge together in one iterable array
    for right_edge, left_edge in zip(bins[1:], bins[:-1]):
        # sums up all 1 entries of each bin 
        counts.append(tf.reduce_sum(binfunction(x, right_edge, left_edge)))
    return tf.squeeze(tf.stack(counts))

@tf.custom_gradient
# binfunction selects all outputs from NN and distributes the accompanying events into the certain bin
# left and right edge represent the bin borders
def binfunction(x, right_edge, left_edge):
    # tf.cast casts a tensor to a new type -> here float32, just in case
    # tf.squeeze removes dimensions of size 1 from the shape of a tensor
    y = tf.squeeze(
        tf.cast(
            tf.cast(x > left_edge, tf.float32) * tf.cast(x <= right_edge, tf.float32), tf.float32
            )
        )
    # for the derivative, the binfunction is approximated by a normal distribution
    def grad(dy):
        width = right_edge - left_edge
        mid = left_edge + 0.5 * width
        #sigma = 0.5 * width # careful! necessary?
        sigma = width
        gauss = tf.exp(-1.0 * (x - mid)**2 / 2.0 / sigma**2)    # careful!
        g = -1.0 * gauss * (x - mid) / sigma**2
        g = tf.squeeze(g) * tf.squeeze(dy)
        return (g, None, None)
    return y, grad



####
#### Classify Dataset
####

column_names = ['x_coordinate', 'y_coordinate', 'origin']
feature_names = column_names[:-1]
label_name = column_names[-1]


####
#### Create TF Dataset
####

x_sig = tf.Variable(normal_distribution(mean[0], cov[0], n[0]), dtype=tf.float32)
x_bkg = tf.Variable(normal_distribution(mean[1], cov[1], n[1]), dtype=tf.float32)
x_bkg_up = tf.Variable(normal_distribution(mean[2], cov[1], n[1]), dtype=tf.float32)
x_bkg_down = tf.Variable(normal_distribution(mean[3], cov[1], n[1]), dtype=tf.float32)

number_of_bins = 20
scale = 4
bins = np.linspace(-scale, scale, number_of_bins)

hist_x_train_signal = np.histogram2d(x_sig[:, 1], x_sig[:, 0], bins= [bins,bins])
hist_x_train_noshift_background = np.histogram2d(x_bkg[:, 1], x_bkg[:, 0], bins= [bins,bins])
hist_x_train_upshift_background = np.histogram2d(x_bkg_up[:, 1], x_bkg_up[:, 0], bins= [bins,bins])
hist_x_train_downshift_background = np.histogram2d(x_bkg_down[:, 1], x_bkg_down[:, 0], bins= [bins,bins])

def makeplot(histograms):
    limit = [-4, 4]
    plt.figure(figsize=(6, 6))
    cmap_sig = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C0"] * 3)
    cmap_bkg = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C1"] * 3)
    cmap = [cmap_sig, cmap_bkg, cmap_bkg, cmap_bkg]
    color=["C0", "C1", "C1",  "C1"]
    label=["Signal", "Background", "Background upshift", "Background downshift"]
    alpha = [0.8, 0.8, 0.4, 0.4]
    for i in range(0, len(histograms)):
        plt.contour(histograms[i][0], extent= [histograms[i][1][0], histograms[i][1][-1], histograms[i][2][0] , histograms[i][2][-1]], cmap=cmap[i], alpha=alpha[i])
        plt.plot([-999], [-999], color=color[i], label=label[i])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[0], limit[1])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
    #plt.savefig("/home/risto/Masterarbeit/test.png", bbox_inches = "tight")
    plt.show()

#makeplot([hist_x_train_signal, hist_x_train_noshift_background, hist_x_train_upshift_background, hist_x_train_downshift_background])


####
#### Create Model
####

# NN with 2 Input Nodes (x-/y-Coordinate), 100 hidden nodes and 1 Output Node (Probability of being Signal/background)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(2,)),  # input shape required
    tf.keras.layers.Dense(1)
])


####
#### Define Loss and Optimization to train Model
####

def loss(model, x_sig, x_bkg, x_bkg_up, x_bkg_down):
    f_sig = tf.squeeze(tf.sigmoid(model(x_sig)))
    f_bkg = tf.squeeze(tf.sigmoid(model(x_bkg)))
    f_bkg_up = tf.squeeze(tf.sigmoid(model(x_bkg_up)))
    f_bkg_down = tf.squeeze(tf.sigmoid(model(x_bkg_down)))

    hist_sig = hist(f_sig, bins_for_plots) # Signal events represented by histogram 
    hist_bkg = hist(f_bkg, bins_for_plots) # Background events represented by histogram
    hist_bkg_up = hist(f_bkg, bins_for_plots) # Background events represented by histogram
    hist_bkg_down = hist(f_bkg, bins_for_plots) # Background events represented by histogram 

    s = [hist_sig[0], hist_sig[1]]
    b = [hist_bkg[0] + hist_bkg_up[0] + hist_bkg_down[0], hist_bkg[1] + hist_bkg_up[1] + hist_bkg_down[1]]

    loss_value = -1.0 * (s[1] / tf.sqrt(s[1] + b[1]) + b[0] / tf.sqrt(s[0] + b[0]))

    return loss_value

def grad(model, x_sig, x_bkg, x_bkg_up, x_bkg_down):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x_sig, x_bkg, x_bkg_up, x_bkg_down)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.keras.optimizers.Adam()


####
#### Training Loop
####

min_loss = 0
max_patience = 10
patience = max_patience
print(grad(model, x_sig, x_bkg, x_bkg_up, x_bkg_down))
loss_value, grads = grad(model, x_sig, x_bkg, x_bkg_up, x_bkg_down)
print("Step: {:02d}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
for epoch in range(1, 300):
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    current_loss = loss(model, x_sig, x_bkg, x_bkg_up, x_bkg_down).numpy()
    if current_loss > min_loss:
        patience -= 1
    else:
        min_loss = current_loss
        patience = max_patience
    
    if epoch % 10 == 0 or patience == 0:
        print("Step: {:02d},         Loss: {:.2f}".format(optimizer.iterations.numpy(), loss(model, x_sig, x_bkg, x_bkg_up, x_bkg_down).numpy()))
        
    if patience == 0:
        print("Trigger early stopping in epoch {}.".format(epoch))
        break


####
#### Plot Results
####

f_sig = tf.squeeze(tf.sigmoid(model(x_sig)))
f_bkg = tf.squeeze(tf.sigmoid(model(x_bkg)))

hist_sig = hist(f_sig, bins_for_plots) # Signal events represented by histogram 
hist_bkg = hist(f_bkg, bins_for_plots) # Background events represented by histogram 

s = [hist_sig[0], hist_sig[1]]
b = [hist_bkg[0], hist_bkg[1]]

pickle.dump([f_sig, f_bkg], open("training_data.pickle", "wb"))

opt_sig_significance = s[1] / np.sqrt(s[1] + b[1])
opt_bkg_significance = b[0] / np.sqrt(s[0] + b[0])

plt.figure(figsize=(7, 6))
plt.hist(bins_for_plots_middle, weights= [s[0], s[1]], bins= bins_for_plots, histtype="step", label="Signal", lw=2)
plt.hist(bins_for_plots_middle, weights= [b[0], b[1]], bins= bins_for_plots, histtype="step", label="Backgorund", lw=2)
plt.legend(loc= "lower center")
plt.title("Background Significance: {:.2f},   Signal Significance: {:.2f}".format(opt_bkg_significance, opt_sig_significance))
plt.xlabel("Projection with decision boundary from NN at {}".format(border))
plt.ylabel("# Events")
plt.axvline(x = border, ymin= 0, ymax= max(n[0], n[1]), color="r", linestyle= "dashed", lw=2)
#plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/significance_hist_NN_{}.png".format(picture_index), bbox_inches = "tight")
#plt.show()


length = 2000
sensibility = 0.005
b = np.linspace(-3, 3, length)
xx, yy = np.meshgrid(b, b, sparse=False)
x = np.vstack([xx.reshape((length * length)), yy.reshape((length * length))]).T
c = tf.squeeze(tf.sigmoid(model(x))).numpy()
c = c.reshape((length, length))
# visualize decision boundary
boundary = tf.squeeze(tf.sigmoid(model(x))).numpy()
boundary = boundary.reshape((length, length))

for y in range(0, length):
    for x in range(0, length):
        if c[y][x] > border-sensibility and c[y][x] < border+sensibility:
            boundary[y][x] = 1
        else:
            boundary[y][x] = 0

limit = [-3, 3]
plt.figure(figsize=(7, 6))
cbar = plt.contourf(xx, yy, c + boundary, levels=np.linspace(c.min(), c.max(), 21))
plt.colorbar(cbar)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Neural network function")
plt.tight_layout()
plt.xlim(limit[0], limit[1])
plt.ylim(limit[0], limit[1])
#plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/NN_function_{}.png".format(picture_index), bbox_inches = "tight")
plt.show()