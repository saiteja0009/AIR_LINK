import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Load cleaned dataset
df = pd.read_csv("data/enhanced_flight_data.csv")

print("Dataset Loaded:", df.shape)

# -------------------------
# Separate Features & Target
# -------------------------

X = df.drop("Price", axis=1)
y = df["Price"]

# -------------------------
# Identify Categorical Columns
# -------------------------

categorical_cols = ["Airline", "Source", "Destination"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# -------------------------
# Column Transformer
# -------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder="passthrough"
)

X_processed = preprocessor.fit_transform(X)

# -------------------------
# Train Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# -------------------------
# Train XGBoost Model
# -------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Evaluate Model
# -------------------------

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("Model saved successfully!")