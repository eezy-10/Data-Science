import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("housing.csv")

df = df.drop(columns=['Unnamed: 0', 'property_id'])
df = df[df['purpose'] == 'For Sale'].drop(columns=['purpose'])

df = df.fillna(df.median(numeric_only=True))

df['price_per_sqft'] = df['price'] / df['Total_Area']
df['bath_bed_ratio'] = df['baths'] / (df['bedrooms'] + 1)

q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1

df = df[
    (df['price'] >= q1 - 1.5 * iqr) &
    (df['price'] <= q3 + 1.5 * iqr)
]

# One-Hot Encode
df = pd.get_dummies(df, columns=['location_id'], drop_first=True)

X = df.select_dtypes(include=['int64', 'float64', 'bool']).drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(name)
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))
    print("-" * 30)

evaluate("Linear", y_test, y_pred_lr)
evaluate("Ridge", y_test, y_pred_ridge)
evaluate("Random Forest", y_test, y_pred_rf)

sample = X_test.iloc[[5]]
predicted_price = rf.predict(sample)[0]

print("Predicted Price:", predicted_price)
print("Actual Price:", y_test.iloc[5])