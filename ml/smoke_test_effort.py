import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Synthetic practice data (NOT real: only to test the pipeline)
rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "size_kloc": rng.uniform(5, 100, n),
    "team_size": rng.integers(2, 20, n),
    "complexity": rng.integers(1, 6, n),
    "experience": rng.integers(1, 6, n),
})
df["effort_pm"] = (
    2.9 * df["size_kloc"] ** 1.05
    * (0.8 + 0.1 * df["complexity"])
    * (1.3 - 0.06 * df["experience"])
    + rng.normal(0, 15, n)
)

# 2. Split into train and test
X = df.drop(columns="effort_pm")
y = df["effort_pm"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train and compare models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

best_name, best_model, best_mae = None, None, float("inf")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name:20s} MAE={mae:7.2f}  RMSE={rmse:7.2f}  R2={r2:5.3f}")
    if mae < best_mae:
        best_name, best_model, best_mae = name, model, mae

# 4. Save the best model
joblib.dump(best_model, "models/effort_smoke.joblib")
print(f"\nBest model: {best_name} (saved to models/effort_smoke.joblib)")