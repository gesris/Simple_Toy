# list of global variables as arrays 

mean = [[- 0.5, 0.5], [0.5, -0.5]]           
#sx = float(input("Signal Mean X: "))
#sy = float(input("Signal Mean Y: "))
#bx = float(input("Background Mean X: "))
#by = float(input("Background Mean Y: "))
s_cov = [[1, 0], [0, 1]]
b_cov = [[1, 0], [0, 1]]
#n_s = int(input("Number of Signal Events: "))
#n_b = int(input("Number of Background Events: "))                        
picture_index = "differentmean3"

bins_for_plots = [0.0, 0.5, 1.0]
bins_for_plots_middle = []
# in this case the middle of bins is [0.25, 0.75]
for i in range(0, len(bins_for_plots) - 1):
    bins_for_plots_middle.append(bins_for_plots[i] + (bins_for_plots[i + 1] - bins_for_plots[i]) / 2)

# Final global variables
#mean = [[sx, sy], [bx, by]]
cov = [s_cov, b_cov]  
#n = [n_s, n_b]  
n = [10000, 10000]
border = bins_for_plots[1]

# Limit of Axes in plots
limit = [-3, 3]