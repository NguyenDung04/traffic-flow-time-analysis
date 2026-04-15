import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1. ĐỌC DỮ LIỆU
# ==============================
df = pd.read_csv("traffic_cleaned.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

print("=== 5 dòng đầu ===")
print(df.head())

# ==============================
# 2. CHUẨN BỊ DỮ LIỆU TRAIN
# ==============================
# One-hot encode weather_main
df_model = pd.get_dummies(df, columns=["weather_main"], drop_first=True)

# Feature cơ bản
base_features = [
    "hour",
    "day",
    "month",
    "dayofweek",
    "is_weekend",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all"
]

# Các cột weather sau khi encode
weather_features = [col for col in df_model.columns if col.startswith("weather_main_")]

features = base_features + weather_features

X = df_model[features]
y = df_model["traffic_volume"]

# ==============================
# 3. CHIA TRAIN / TEST
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# ==============================
# 4. TRAIN MODEL
# ==============================
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ==============================
# 5. ĐÁNH GIÁ MODEL
# ==============================
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== ĐÁNH GIÁ MODEL RANDOM FOREST ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# ==============================
# 6. TẠO DỮ LIỆU DỰ BÁO 1 TUẦN TỚI
# ==============================
last_time = df["datetime"].max()

future_times = pd.date_range(
    start=last_time + pd.Timedelta(hours=1),
    periods=24 * 7,
    freq="h"
)

future_df = pd.DataFrame({
    "datetime": future_times
})

# Tạo feature thời gian
future_df["hour"] = future_df["datetime"].dt.hour
future_df["day"] = future_df["datetime"].dt.day
future_df["month"] = future_df["datetime"].dt.month
future_df["dayofweek"] = future_df["datetime"].dt.dayofweek
future_df["is_weekend"] = future_df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

# ==============================
# 7. GÁN THỜI TIẾT CHO TƯƠNG LAI
# ==============================
# Demo: dùng giá trị trung bình
future_df["temp"] = df["temp"].mean()
future_df["rain_1h"] = df["rain_1h"].mean()
future_df["snow_1h"] = df["snow_1h"].mean()
future_df["clouds_all"] = df["clouds_all"].mean()

# Giả định thời tiết phổ biến
most_common_weather = df["weather_main"].mode()[0]
future_df["weather_main"] = most_common_weather

print("\nThời tiết giả định cho dự báo:", most_common_weather)

# ==============================
# 8. XỬ LÝ ONE-HOT CHO DỮ LIỆU TƯƠNG LAI
# ==============================
future_model = pd.get_dummies(future_df, columns=["weather_main"], drop_first=True)

# Bổ sung cột còn thiếu để khớp với train
for col in features:
    if col not in future_model.columns:
        future_model[col] = 0

# Đúng thứ tự cột
future_model = future_model[features]

# ==============================
# 9. DỰ BÁO 1 TUẦN TỚI
# ==============================
future_df["predicted_traffic_volume"] = rf.predict(future_model)

print("\n=== DỰ BÁO 1 TUẦN TỚI ===")
print(future_df[["datetime", "predicted_traffic_volume"]].head(20))

# ==============================
# 10. LƯU FILE KẾT QUẢ
# ==============================
future_df.to_csv("forecast_next_7_days.csv", index=False, encoding="utf-8-sig")
print("\nĐã lưu file forecast_next_7_days.csv")

# ==============================
# 11. THỐNG KÊ NHANH
# ==============================
peak_idx = future_df["predicted_traffic_volume"].idxmax()
low_idx = future_df["predicted_traffic_volume"].idxmin()

print("\n=== THỐNG KÊ DỰ BÁO ===")
print("Cao nhất:")
print(future_df.loc[peak_idx, ["datetime", "predicted_traffic_volume"]])

print("\nThấp nhất:")
print(future_df.loc[low_idx, ["datetime", "predicted_traffic_volume"]])

# Trung bình theo ngày trong tuần của tuần dự báo
avg_by_day = future_df.groupby("dayofweek")["predicted_traffic_volume"].mean()
print("\nTrung bình theo ngày trong tuần:")
print(avg_by_day)

# ==============================
# 12. VẼ BIỂU ĐỒ DỰ BÁO
# ==============================
plt.figure(figsize=(14, 5))
plt.plot(
    future_df["datetime"],
    future_df["predicted_traffic_volume"],
    marker="o",
    markersize=3
)
plt.title("Forecast Traffic Volume for Next 7 Days")
plt.xlabel("Datetime")
plt.ylabel("Predicted Traffic Volume")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()