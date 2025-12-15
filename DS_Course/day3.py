import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("housing.csv")

df = df.drop(columns=['Unnamed: 0', 'property_id'])
df = df[df['purpose'] == 'For Sale'].drop(columns=['purpose'])

df = df.fillna(df.median(numeric_only=True))

# One-Hot Encode Location
df = pd.get_dummies(df, columns=['location_id'], drop_first=True)

# X = df.drop(columns=['price', 'page_url', 'property_type', 'location', 'city', 'province_name', 'date_added', 'agent', 'agency'])
X = df.select_dtypes(include=['int64', 'float64', 'bool']).drop(columns=['price'])
# print(X.dtypes)

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("LINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

sample = X_test.iloc[5]
predicted_price = lr.predict([sample])[0]
print("Predicted Price:", predicted_price)
print("Actual Price:", y_test.iloc[[5]])

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

print("\nRIDGE REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("R²:", r2_score(y_test, y_pred_ridge))

predicted_price = ridge.predict([sample])[0]
print("Predicted Price:", predicted_price)
print("Actual Price:", y_test.iloc[5])

coeffs = pd.Series(ridge.coef_, index=X.columns)
print(coeffs.sort_values(ascending=False).head(10))