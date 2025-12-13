import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('housing.csv')
df = df.drop(columns=['Unnamed: 0'])
# print("Head result: ")
# print(df.head())
# print("Info result: ")
# df.info()
# print("Describe result: ")
# print(df.describe())

# Count missing values
# print(df.isnull().sum())


# Example: if 'baths' or 'beds' are missing, impute or drop
# df['baths'] = df['baths'].fillna(df['baths'].median())
# df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())

# Basic EDA: distribution of price
# sns.histplot(df['price'])
# plt.ticklabel_format(style='plain', axis='x')
# plt.show()

# Price vs Area
# sns.scatterplot(data=df, x='Total_Area', y='price')
# plt.show()

# Correlation matrix (numeric features)
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.show()

# print(df['property_type'].value_counts())
# print(df['city'].value_counts())

# Outliers Methods
# sns.boxplot(x=df['price'])
# plt.show()
# print(df.sort_values(by='price', ascending=False).head(10))

# sns.boxplot(x=df['Total_Area'])
# plt.show()
# sns.boxplot(data=df, x='property_type', y='price')
# plt.show()

# print(df.groupby('property_type')['price'].mean())
# print(df['property_type'].value_counts())

# need = df['baths'].isna().sum()
# if need > 0 :
#     df['baths'] = df['baths'].fillna(df['baths'].median())
# else:
#     print("No change needed in '/baths/' column")
# need = df['bedrooms'].isna().sum()
# if need > 0 : 
#     df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
# else: 
#     print("No Chnage in '/bedrooms/' column aswell")