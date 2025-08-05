import pandas as pd

health_data = pd.read_csv('health_data.csv', header=0, sep=',')
# health_data.dropna(axis=0, inplace=True)
print(health_data.info())
print("\nThe data is loaded successfully.\n")
health_data.dropna(axis=0, inplace=True)
print(health_data)