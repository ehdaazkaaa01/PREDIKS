
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("Toyota (1).csv")

# Preprocessing
df = df.dropna()
df['transmission'] = LabelEncoder().fit_transform(df['transmission'])
df['fuelType'] = LabelEncoder().fit_transform(df['fuelType'])

# Fitur dan target
X = df[['year', 'mileage', 'transmission', 'fuelType', 'tax', 'mpg']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Simpan model
joblib.dump(model_rf, "model_rf.pkl")
print("Model Random Forest berhasil disimpan sebagai model_rf.pkl")
