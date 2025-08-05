import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the health data
health_data = pd.read_csv('health_data.csv', header=0, sep=',')

correlation_matrix = health_data.corr()

axis_corr = sns.heatmap(
    correlation_matrix,
    vmin = -1,
    vmax = 1,
    center = 0,
    cmap = sns.diverging_palette(50, 50),
    annot = True,
)

plt.show()