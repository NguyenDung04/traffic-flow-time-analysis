import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1. LOAD DỮ LIỆU
# ==============================
df = pd.read_csv("traffic_cleaned.csv")

# Đảm bảo datetime đúng kiểu
df["datetime"] = pd.to_datetime(df["datetime"])

# ==============================
# 2. CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN
# ==============================
# One-hot encode weather_main
df_model = pd.get_dummies(df, columns=["weather_main"], drop_first=True)

# Chọn feature
base_features = [
    "hour", "day", "month", "dayofweek", "is_weekend",
    "temp", "rain_1h", "snow_1h", "clouds_all"
]

weather_features = [col for col in df_model.columns if col.startswith("weather_main_")]
features = base_features + weather_features

X = df_model[features]
y = df_model["traffic_volume"]

# Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. HUẤN LUYỆN MODEL
# ==============================
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Đánh giá nhanh
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== ĐÁNH GIÁ MODEL RANDOM FOREST ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# ==============================
# 4. TẠO DỮ LIỆU TƯƠNG LAI
# ==============================
# Ví dụ: dự đoán 24 giờ tiếp theo kể từ mốc cuối cùng trong dữ liệu
last_time = df["datetime"].max()
future_times = pd.date_range(
    start=last_time + pd.Timedelta(hours=1),
    periods=24,
    freq="h"   # 🔥 sửa ở đây
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
# 5. GÁN GIÁ TRỊ THỜI TIẾT CHO TƯƠNG LAI
# ==============================
# Cách đơn giản để demo:
# dùng giá trị trung bình từ dữ liệu gốc
future_df["temp"] = df["temp"].mean()
future_df["rain_1h"] = df["rain_1h"].mean()
future_df["snow_1h"] = df["snow_1h"].mean()
future_df["clouds_all"] = df["clouds_all"].mean()

# Chọn một loại thời tiết giả định
# Ví dụ: Clouds
future_df["weather_main"] = "Clouds"

# ==============================
# 6. ONE-HOT ENCODE CHO DỮ LIỆU TƯƠNG LAI
# ==============================
future_model = pd.get_dummies(future_df, columns=["weather_main"], drop_first=True)

# Đảm bảo future có đủ cột giống train
for col in features:
    if col not in future_model.columns:
        future_model[col] = 0

# Sắp đúng thứ tự cột
future_model = future_model[features]

# ==============================
# 7. DỰ ĐOÁN TƯƠNG LAI
# ==============================
future_df["predicted_traffic_volume"] = rf.predict(future_model)

print("\n=== DỰ ĐOÁN 24 GIỜ TIẾP THEO ===")
print(future_df[["datetime", "predicted_traffic_volume"]])

# ==============================
# 8. LƯU FILE KẾT QUẢ
# ==============================
future_df.to_csv("future_traffic_prediction.csv", index=False)
print("\nĐã lưu file: future_traffic_prediction.csv")

# ==============================
# 9. VẼ BIỂU ĐỒ DỰ ĐOÁN
# ==============================
plt.figure(figsize=(12, 5))
plt.plot(future_df["datetime"], future_df["predicted_traffic_volume"], marker="o")
plt.title("Predicted Traffic Volume for Next 24 Hours")
plt.xlabel("Datetime")
plt.ylabel("Predicted Traffic Volume")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()