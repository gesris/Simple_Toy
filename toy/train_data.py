import tensorflow as tf
import numpy as np
import global_variables
import create_data
import pickle

# here x are event coordinates
def model(x, num_inputs, reuse=False, scope="model"):
    # tf.variable_scope is a context manager which validates that the (optional) values are from the
    # same graph, ensures that graph is the default graph, and pushes a name scope and a variable scope
    with tf.variable_scope(scope, reuse=reuse) as scope:
        hidden_nodes = 100 # in hidden layer
        # from input to hidden layer:
        w1 = tf.get_variable('w1', shape=(num_inputs, hidden_nodes), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1)) #random_uniform!
        b1 = tf.get_variable('b1', shape=(hidden_nodes), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
        # from hidden to output layer:
        w2 = tf.get_variable('w2', shape=(hidden_nodes, 1), dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', shape=(1), dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))

    # tf.nn.relu is Rectified Linear Unit activation function - here for first neuron computation
    hidden_layer = tf.nn.relu(tf.add(b1, tf.matmul(x, w1))) # equivalent to sum(w1*x1 + b1) for each hidden node
    # no activation function for output needed, since its defined later in code (sigmoid)
    output_layer = tf.add(b2, tf.matmul(hidden_layer, w2)) # equivalent to sum(w2*x2 + b2) for signle output node
    f = tf.squeeze(output_layer)
    return f

# custom gradient allows better numerical precision for functions with diverging or not defined derivative
@tf.custom_gradient
# binfunction selects all outputs from NN and distributes the accompanying events into the certain bin
# left and right edge represent the bin borders
def binfunction(x, right_edge, left_edge):
    # tf.cast casts a tensor to a new type -> here float32, just in case
    # tf.squeeze removes dimensions of size 1 from the shape of a tensor
    y = tf.squeeze(tf.cast(tf.cast(x > left_edge, tf.float32) * tf.cast(x <= right_edge, tf.float32), tf.float32))
    # for the derivative, the binfunction is approximated by a normal distribution
    def grad(dy):
        width = right_edge - left_edge
        mid = left_edge + 0.5 * width
        sigma = 0.5 * width
        gauss = tf.exp(-1.0 * (x - mid)**2 / 2.0 / sigma**2)
        g = -1.0 * gauss * (x - mid) / sigma**2
        g = tf.squeeze(g) * tf.squeeze(dy)
        return (g, None, None)
    return y, grad


# function that counts all classified events in each bin
def hist(x, bins):
    counts = []
    # splits histogram in bins regarding their left and right edges
    # zip function puts left and right edge together in one iterable array
    for right_edge, left_edge in zip(bins[1:], bins[:-1]):
        # sums up all 1 entries of each bin
        counts.append(tf.reduce_sum(binfunction(x, right_edge, left_edge)))
    return tf.squeeze(tf.stack(counts))


def training():
    # variables have to be defined before start of tf.session()!
    # running multiple sessions also possible
    x_sig = tf.placeholder(tf.float32, shape=[None, 2]) # x-y-coordinates of signal events
    x_bkg = tf.placeholder(tf.float32, shape=[None, 2]) # x-y-coordinates of background events
    f_sig = tf.sigmoid(model(x_sig, 2)) # NN output of signal
    f_bkg = tf.sigmoid(model(x_bkg, 2, reuse=True)) # NN output of background

    hist_sig = hist(f_sig, global_variables.bins_for_plots) # Signal events represented by histogram 
    hist_bkg = hist(f_bkg, global_variables.bins_for_plots) # Background events represented by histogram 

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
        loss_, _ = sess.run([loss, minimize], feed_dict={x_sig: create_data.normal_distribution(global_variables.mean[0], global_variables.cov[0], global_variables.n[0]), x_bkg: create_data.normal_distribution(global_variables.mean[1], global_variables.cov[1], global_variables.n[1])})
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
    
    opt_hist_sig, opt_hist_bkg = sess.run([hist_sig, hist_bkg], feed_dict={x_sig: create_data.normal_distribution(global_variables.mean[0], global_variables.cov[0], global_variables.n[0]), x_bkg: create_data.normal_distribution(global_variables.mean[1], global_variables.cov[1], global_variables.n[1])})

    # Signal and background counts to create hist
    s = [opt_hist_sig[0], opt_hist_sig[1]]
    b = [opt_hist_bkg[0], opt_hist_bkg[1]]

    # NN Function to create plot
    length = 2000
    grid_size = np.linspace(-3, 3, length)
    xx, yy = np.meshgrid(grid_size, grid_size, sparse=False)
    grid = np.vstack([xx.reshape((length * length)), yy.reshape((length * length))]).T
    nn_function = sess.run(f_sig, feed_dict={x_sig: grid})
    nn_function = nn_function.reshape((length, length))
    nn_decision_boundary = sess.run(f_sig, feed_dict={x_sig: grid})
    nn_decision_boundary = nn_decision_boundary.reshape((length, length))

    return s, b, nn_function, nn_decision_boundary

s, b, nn_function, nn_decision_boundary = training()

pickle.dump([s, b, nn_function, nn_decision_boundary],
            open("training_data.pickle", "wb"))
