import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import sys
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')
df = df.drop(columns=['Unnamed: 0', 'property_id'])
df = df.fillna(df.median(numeric_only=True))

# as there are two types of houses
# Sale Model
df_sale = df[df['purpose'] == 'For Sale'].copy().drop(columns=['purpose'])

# # checking outliers
# # sns.boxplot(df_sale['price'])
# # plt.show()
# # sys.exit()

X_sale = df_sale[['Total_Area', 'location_id', 'latitude', 'longitude', 'bedrooms', 'baths']]
y_sale = df_sale['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sale, y_sale, test_size=0.2, random_state=42
)

sale_model = LinearRegression()
sale_model.fit(X_train_s, y_train_s)

y_sale_pred = sale_model.predict(X_test_s)
mae_sale = mean_absolute_error(y_test_s, y_sale_pred)
mse_sale = mean_squared_error(y_test_s, y_sale_pred)
rmse_sale = np.sqrt(mse_sale)
r2_sale = r2_score(y_test_s, y_sale_pred)

print("SALE MODEL:")
print("MAE:", mae_sale)
print("MSE:", mse_sale)
print("RMSE:", rmse_sale)
print("R² Score:", r2_sale)

sample = X_test_s.iloc[5]
predicted_price = sale_model.predict([sample])[0]

print("Predicted Price:", predicted_price)
print("Actual Price:", y_test_s.iloc[5])

#“Rent data contained only 12 samples with extreme variance. Due to insufficient data and unstable distribution, rent price modeling was excluded to maintain statistical validity.”
# Rent Model
# df_rent = df[df['purpose'] == 'For Rent'].copy().drop(columns=['purpose'])

# X_rent = df_rent[['Total_Area', 'location_id', 'latitude', 'longitude', 'bedrooms', 'baths']]
# y_rent = df_rent['price']

# X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
#     X_rent, y_rent, test_size=0.2, random_state=42
# )

# rent_model = LinearRegression()
# rent_model.fit(X_train_r, y_train_r)

# y_rent_pred = rent_model.predict(X_test_r)

# mae_rent = mean_absolute_error(y_test_r, y_rent_pred)
# mse_rent = mean_squared_error(y_test_r, y_rent_pred)
# rmse_rent = np.sqrt(mse_rent)
# r2_rent = r2_score(y_test_r, y_rent_pred)

# print("\nRENT MODEL:")
# print("MAE:", mae_rent)
# print("MSE:", mse_rent)
# print("RMSE:", rmse_rent)
# print("R² Score:", r2_rent)

# sample = X_test_r.iloc[1]
# predicted_price = rent_model.predict([sample])[0]

# print("Predicted Price:", predicted_price)
# print("Actual Price:", y_test_r.iloc[1])