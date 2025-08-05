import pandas as pd
import numpy as np

health_data = pd.read_csv('health_data.csv', header=0, sep=',')
# health_data.dropna(axis=0, inplace=True)
# health_data["Average_Pulse"] = health_data["Average_Pulse"].astype(float)
# health_data['Max_Pulse'] = health_data['Max_Pulse'].astype(float)

# slope_intercept = np.polyfit(health_data['Average_Pulse'], health_data['Calorie_Burnage'], 1)
# print("Slope and Intercept:", slope_intercept)

# print(health_data.describe())

# Calculate the 10th percentile of the Average_Pulse column
# percentile10 = np.percentile(health_data['Average_Pulse'], 10)
# print("10th Percentile is: ",percentile10)
print(np.var(health_data, axis=0))
print(np.mean(health_data["Duration"]))
# Output only the Duration
# cv = np.std(health_data) / np.mean(health_data)
# print(cv)