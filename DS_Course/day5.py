import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
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

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(name)
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))
    print("-" * 30)

evaluate("Random Forest Tuned", y_test, y_pred_rf)

sample = X_test.iloc[[5]]
predicted_price = rf.predict(sample)[0]

print("Predicted Price:", predicted_price)
print("Actual Price:", y_test.iloc[5], "\n")

importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# print(importances)
important_features = importances[importances > 0.002].index
# print(important_features)

X_imp = X[important_features]

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imp, y, test_size=0.2, random_state=42
)

rf_small = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_small.fit(X_train_i, y_train_i)
y_pred_small = rf_small.predict(X_test_i)

evaluate("RF Reduced Features", y_test_i, y_pred_small)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1
)

grid.fit(X_train_i, y_train_i)

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test_i)

evaluate("Best RF (Tuned)", y_test_i, y_pred_best)