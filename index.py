# my first Python script
# This script creates a DataFrame and prints it along with the number of rows and columns
import pandas as pd

d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}

df = pd.DataFrame(data=d)
count_column = df.shape[1]
count_row = df.shape[0]

print(df)
print("Number of Column", count_column)
print("Number of Row", count_row)