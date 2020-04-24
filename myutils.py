import tensorflow as tf
tf.set_random_seed(1234)
import numpy as np
np.random.seed(1234)

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

