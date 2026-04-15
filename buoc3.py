import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dữ liệu đã clean
df = pd.read_csv("traffic_cleaned.csv")

print("=== Dữ liệu ===")
print(df.head())


# ==============================
# 1. Thống kê mô tả
# ==============================
print("\n=== Thống kê traffic_volume ===")
print(df['traffic_volume'].describe())

print("\nMean:", df['traffic_volume'].mean())
print("Max:", df['traffic_volume'].max())
print("Min:", df['traffic_volume'].min())


# ==============================
# 2. Phân tích theo giờ (GIỜ CAO ĐIỂM)
# ==============================
traffic_by_hour = df.groupby('hour')['traffic_volume'].mean()

print("\n=== Lưu lượng theo giờ ===")
print(traffic_by_hour)

peak_hour = traffic_by_hour.idxmax()
print("\n🔥 Giờ cao điểm:", peak_hour)


# Vẽ biểu đồ
plt.figure()
traffic_by_hour.plot()
plt.title("Traffic Volume by Hour")
plt.xlabel("Hour")
plt.ylabel("Traffic Volume")
plt.show()


# ==============================
# 3. Phân tích theo ngày trong tuần
# ==============================
traffic_by_dayofweek = df.groupby('dayofweek')['traffic_volume'].mean()

print("\n=== Lưu lượng theo ngày trong tuần ===")
print(traffic_by_dayofweek)

plt.figure()
traffic_by_dayofweek.plot(kind='bar')
plt.title("Traffic by Day of Week (0=Mon)")
plt.xlabel("Day of Week")
plt.ylabel("Traffic Volume")
plt.show()


# ==============================
# 4. Phân tích theo tháng
# ==============================
traffic_by_month = df.groupby('month')['traffic_volume'].mean()

print("\n=== Lưu lượng theo tháng ===")
print(traffic_by_month)

plt.figure()
traffic_by_month.plot(kind='bar')
plt.title("Traffic by Month")
plt.xlabel("Month")
plt.ylabel("Traffic Volume")
plt.show()


# ==============================
# 5. So sánh theo thời tiết
# ==============================
traffic_by_weather = df.groupby('weather_main')['traffic_volume'].mean()

print("\n=== Lưu lượng theo thời tiết ===")
print(traffic_by_weather.sort_values(ascending=False))

plt.figure()
traffic_by_weather.sort_values().plot(kind='barh')
plt.title("Traffic by Weather")
plt.xlabel("Traffic Volume")
plt.ylabel("Weather")
plt.show()


# ==============================
# 6. Heatmap (giờ + ngày trong tuần)
# ==============================
pivot_table = df.pivot_table(
    values='traffic_volume',
    index='dayofweek',
    columns='hour',
    aggfunc='mean'
)

print("\n=== Heatmap data ===")
print(pivot_table)

plt.figure()
sns.heatmap(pivot_table)
plt.title("Heatmap Traffic (Day vs Hour)")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.show()