import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dữ liệu đã clean
df = pd.read_csv("traffic_cleaned.csv")

# 2. Chọn feature
features = ['hour', 'day', 'month', 'dayofweek', 'is_weekend', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']

# Encode weather (one-hot)
df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)

# Thêm các cột weather vào feature
weather_cols = [col for col in df.columns if 'weather_main_' in col]
features = features + weather_cols

X = df[features]
y = df['traffic_volume']

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# ==============================
# 4. Linear Regression
# ==============================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("\n=== Linear Regression ===")
print("MAE:", mae_lr)
print("RMSE:", rmse_lr)


# ==============================
# 5. Random Forest
# ==============================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n=== Random Forest ===")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)

plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()