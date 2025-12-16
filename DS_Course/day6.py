import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

df = pd.read_csv("housing.csv")

df = df.drop(columns=['Unnamed: 0', 'property_id'])
df = df[df['purpose'] == 'For Sale'].drop(columns=['purpose'])

q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1

df_sale = df[
    (df['price'] >= q1 - 1.5 * iqr) &
    (df['price'] <= q3 + 1.5 * iqr)
]

features = [
    'Total_Area',
    'bedrooms',
    'baths',
    'latitude',
    'longitude',
    'location_id'
]

X = df_sale[features]
y = df_sale['price']

num_features = ['Total_Area', 'bedrooms', 'baths', 'latitude', 'longitude']
cat_features = ['location_id']

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
cat_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

model_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

joblib.dump(model_pipeline, 'pakistan_housing_price_model.pkl')

loaded_model = joblib.load('pakistan_housing_price_model.pkl')

new_property = pd.DataFrame([{
    'Total_Area': 1200,
    'bedrooms': 3,
    'baths': 2,
    'latitude': 31.5204,
    'longitude': 74.3587,
    'location_id': 3852
}])

prediction = loaded_model.predict(new_property)
print("Predicted Price:", prediction[0])

test_cases = pd.DataFrame([
    {'Total_Area': 500, 'bedrooms': 1, 'baths': 1, 'latitude': 31.5, 'longitude': 74.3, 'location_id': 9999},
    {'Total_Area': 3000, 'bedrooms': 6, 'baths': 5, 'latitude': 24.86, 'longitude': 67.01, 'location_id': 3852}
])

loaded_model.predict(test_cases)