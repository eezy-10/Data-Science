import pandas as pd
import matplotlib.pyplot as plt

health_data = pd.read_csv('health_data.csv', header=0, sep=',')

# health_data.plot(x='Average_Pulse', y='Calorie_Burnage', kind='scatter')
# plt.title('Average Pulse vs Calorie Burnage')
health_data.plot(x = 'Duration', y = 'Max_Pulse', kind = 'scatter')
plt.show()