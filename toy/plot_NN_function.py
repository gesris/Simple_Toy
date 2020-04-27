import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import global_variables

length = 2000
sensibility = 0.005
grid_size = np.linspace(-3, 3, length)
xx, yy = np.meshgrid(grid_size, grid_size, sparse=False)
x = np.vstack([xx.reshape((length * length)), yy.reshape((length * length))]).T

# load data
_, _, nn_function, nn_decision_boundary = pickle.load(open("training_data.pickle", "rb"))
_, _, lin_func, graph_limit, _, _ = pickle.load(open("initial_data.pickle", "rb"))

for y in range(0, length):
    for x in range(0, length):
        if nn_function[y][x] > global_variables.border-sensibility and nn_function[y][x] < global_variables.border+sensibility:
            nn_decision_boundary[y][x] = 1
        else:
            nn_decision_boundary[y][x] = 0

plt.figure(figsize=(7, 6))
cbar = plt.contourf(xx, yy, nn_function + nn_decision_boundary, levels=np.linspace(nn_function.min(), nn_function.max(), 21))
plt.colorbar(cbar)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Neural network function")
#plt.text(4.8, -3, "\t \t Signal \t Background \n \n Mean: \t {} \t     {} \n n: \t    {} \t        {} \n Cov: \t  {}    {} \n \n Decision boundary: \t {}".format(mean[0], mean[1], n[0], n[1], cov[0], cov[1], border).expandtabs(), fontsize= 12, bbox=dict(facecolor='grey', alpha=0.3))
plt.tight_layout()
plt.xlim(global_variables.limit[0], global_variables.limit[1])
plt.ylim(global_variables.limit[0], global_variables.limit[1])
plt.plot([lin_func[0], lin_func[-1]], [graph_limit[0], graph_limit[1]], color="r", label="Maximum likelihood decision boundary")
#plt.savefig("/home/risto/Masterarbeit/Python/significance_plots/NN_function_{}.png".format(picture_index), bbox_inches = "tight")
plt.show()
