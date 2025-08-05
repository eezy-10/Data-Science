import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

health_data = pd.read_csv('health_data.csv', header = 0, sep = ',')

x = health_data['Average_Pulse']
y = health_data['Calorie_Burnage']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.ylim(ymin=0, ymax=350)
plt.xlim(xmin=0, xmax=130)
plt.xlabel("Average_Pulse")
plt.ylabel ("Calorie_Burnage")
plt.show()