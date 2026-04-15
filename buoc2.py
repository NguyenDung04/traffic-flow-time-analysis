import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# 1. Load dữ liệu
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

print("=== 5 dòng đầu ===")
print(df.head())

print("\n=== Thông tin dữ liệu ===")
print(df.info())


# 2. Convert datetime
df['datetime'] = pd.to_datetime(df['date_time'])


# 3. Tạo feature thời gian
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.dayofweek


# 4. Kiểm tra missing
print("\n=== Missing values ===")
print(df.isnull().sum())


# 5. Xử lý missing (CÁCH ĐÚNG)
# Không drop toàn bộ vì holiday thiếu rất nhiều
df['holiday'] = df['holiday'].fillna('None')


# 6. Tạo thêm feature hữu ích
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)


# 7. Kiểm tra lại
print("\n=== Dữ liệu sau khi xử lý ===")
print(df[['datetime', 'hour', 'day', 'month', 'traffic_volume']].head())

print("\n=== Kích thước dữ liệu ===")
print(df.shape)


# 8. Thống kê traffic
print("\n=== Thống kê traffic ===")
print(df['traffic_volume'].describe())


# 9. (Khuyến nghị) Lưu file clean
df.to_csv("traffic_cleaned.csv", index=False)
print("\nĐã lưu file traffic_cleaned.csv")