import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Load dữ liệu
# ==============================
df = pd.read_csv("traffic_cleaned.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort + set index (QUAN TRỌNG)
df = df.sort_values('datetime')
df = df.set_index('datetime')


# ==============================
# 2. LINE CHART THEO THỜI GIAN (ĐÃ FIX)
# ==============================
# Lấy trung bình theo ngày để mượt
df_daily = df['traffic_volume'].resample('D').mean()

plt.figure()
plt.plot(df_daily)
plt.title("Traffic Volume Over Time (Daily Average)")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.show()


# ==============================
# 3. TRAFFIC THEO GIỜ
# ==============================
traffic_by_hour = df.groupby(df.index.hour)['traffic_volume'].mean()

plt.figure()
plt.plot(traffic_by_hour)
plt.title("Traffic Volume by Hour")
plt.xlabel("Hour")
plt.ylabel("Traffic Volume")
plt.show()


# ==============================
# 4. HEATMAP (GIỜ vs NGÀY)
# ==============================
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

pivot = df.pivot_table(
    values='traffic_volume',
    index='dayofweek',
    columns='hour',
    aggfunc='mean'
)

plt.figure()
sns.heatmap(pivot)
plt.title("Heatmap Traffic (Day vs Hour)")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.show()


# ==============================
# 5. PHÂN TÍCH TỰ ĐỘNG
# ==============================
print("\n=== PHÂN TÍCH ===")

peak_hour = traffic_by_hour.idxmax()
low_hour = traffic_by_hour.idxmin()

print(f"- Giờ cao điểm: {peak_hour}h")
print(f"- Giờ thấp điểm: {low_hour}h")

weekday = df[df['dayofweek'] < 5]['traffic_volume'].mean()
weekend = df[df['dayofweek'] >= 5]['traffic_volume'].mean()

print(f"- Trung bình ngày thường: {weekday:.2f}")
print(f"- Trung bình cuối tuần: {weekend:.2f}")

# Tìm tháng cao điểm
df['month'] = df.index.month
traffic_by_month = df.groupby('month')['traffic_volume'].mean()

peak_month = traffic_by_month.idxmax()
print(f"- Tháng có lưu lượng cao nhất: {peak_month}")