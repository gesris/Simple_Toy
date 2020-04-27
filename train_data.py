import tensorflow as tf
import myutils
import global_variables
import create_data

def training():
    # variables have to be defined before start of tf.session()!
    # running multiple sessions also possible
    x_sig = tf.placeholder(tf.float32, shape=[None, 2]) # x-y-coordinates of signal events
    x_bkg = tf.placeholder(tf.float32, shape=[None, 2]) # x-y-coordinates of background events
    f_sig = tf.sigmoid(myutils.model(x_sig, 2)) # NN output of signal
    f_bkg = tf.sigmoid(myutils.model(x_bkg, 2, reuse=True)) # NN output of background

    hist_sig = myutils.hist(f_sig, global_variables.bins_for_plots) # Signal events represented by histogram 
    hist_bkg = myutils.hist(f_bkg, global_variables.bins_for_plots) # Background events represented by histogram 

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

    opt_sig_significance = s[1] / tf.sqrt(s[1] + b[1])
    opt_bkg_significance = b[0] / tf.sqrt(s[0] + b[0])
    opt_total_significance = opt_sig_significance + opt_bkg_significance

    return [opt_hist_sig[0], opt_hist_sig[1]], [opt_hist_bkg[0], opt_hist_bkg[1]], opt_sig_significance, opt_bkg_significance, opt_total_significance
