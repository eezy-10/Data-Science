import pandas as pd
import numpy as np

health_data = pd.read_csv('health_data.csv', header=0, sep=',')
# Calculate the correlation matrix
correlation_matrix = health_data.corr()
print(correlation_matrix)