import numpy as np
np.random.seed(1234)

def normal_distribution(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)

def create_initial_data(mean, cov, n):
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
    
    def num_sig(x):
        if magnitude == 0:
            return [np.sum((x[0] - bias[0]) < 0), np.sum((x[0] - bias[0]) > 0)]
        else:
            return [np.sum(x[1] > (1 / magnitude) * (x[0] - bias[0]) + bias[1]), np.sum(x[1] < (1 / magnitude) * (x[0] - bias[0]) + bias[1])]

    return hist_signal, hist_background, lin_func, graph_limit, num_sig(signal), num_sig(background)


