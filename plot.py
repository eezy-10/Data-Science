import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

health_data = pd.read_csv('health_data.csv', header=0, sep=',')
health_data.dropna(axis=0, inplace=True)
health_data["Average_Pulse"] = health_data["Average_Pulse"].astype(float)
health_data['Max_Pulse'] = health_data['Max_Pulse'].astype(float)
print(health_data)
# x = health_data['Average_Pulse']
# y = health_data['Calorie_Burnage']
# slope_intercept = np.polyfit(x, y, 1)
# print(slope_intercept)

# def func(x):
#     return 2*x + 80

# print(func(80))
# def slope(x1, y1, x2, y2):
#     return (y2 - y1) / (x2 - x1)
# print('The Slope of Line is:', slope(80, 240, 90, 260))

# health_data.plot(x ='Average_Pulse', y='Calorie_Burnage', kind='line')

# plt.xlim(xmin=0, xmax=150)
# plt.ylim(ymin=0, ymax=400)

# plt.show()