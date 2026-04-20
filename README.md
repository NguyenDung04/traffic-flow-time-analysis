# 🚦 Phân tích dữ liệu giao thông – Traffic Analysis

**Chuyên đề 3 – Bài tập lớn số 10**  
Phân tích lưu lượng giao thông theo thời gian sử dụng Python.

## 📌 Mục tiêu
Dự án được thực hiện nhằm khai thác và phân tích dữ liệu giao thông để nhận biết các quy luật biến động lưu lượng xe theo thời gian. Bên cạnh đó, bài toán còn hướng đến việc xây dựng mô hình dự báo lưu lượng giao thông trong tương lai, phục vụ cho việc đánh giá, hỗ trợ quản lý và ra quyết định.

Các mục tiêu chính gồm:
- Xử lý dữ liệu thời gian `datetime`
- Phân tích **giờ cao điểm**
- Phát hiện **bất thường** trong lưu lượng giao thông
- Trực quan hóa xu hướng giao thông
- Dự báo lưu lượng giao thông trong tương lai

## 📁 Nguồn dữ liệu
Bộ dữ liệu sử dụng trong bài là **Metro Interstate Traffic Volume Dataset** trên Kaggle.  
Dữ liệu mô tả lưu lượng phương tiện lưu thông trên tuyến xa lộ liên bang tại khu vực **Minneapolis – St. Paul**, bao gồm các thông tin như:
- Thời gian ghi nhận
- Nhiệt độ
- Mưa, tuyết
- Mức độ mây
- Điều kiện thời tiết chính
- Ngày lễ
- Lưu lượng giao thông

Nguồn tham khảo: [Metro Interstate Traffic Volume Dataset](https://www.kaggle.com/datasets)

## 🧰 Hệ sinh thái Python sử dụng
Dự án được xây dựng bằng Python và các thư viện phân tích dữ liệu phổ biến:

- `pandas` – xử lý và biến đổi dữ liệu bảng
- `numpy` – hỗ trợ tính toán số học
- `matplotlib` – trực quan hóa dữ liệu
- `seaborn` – hỗ trợ biểu đồ thống kê
- `scikit-learn` – xây dựng mô hình dự báo và đánh giá mô hình
- `Jupyter Notebook` – môi trường chạy và trình bày phân tích

## 🔁 Quy trình phân tích (Flow)

| Bước | Nội dung |
|------|----------|
| 1 | Thu thập dữ liệu (Collect) |
| 2 | Làm sạch & tiền xử lý (Clean / Preprocess) |
| 3 | Khám phá & phân tích (Explore / Analyze) |
| 4 | Mô hình hóa (Model – Forecasting) |
| 5 | Trực quan hóa & báo cáo (Visualize & Report) |

## 🧪 Nội dung thực hiện

### Bước 1. Thu thập dữ liệu
Dữ liệu được lấy từ bộ dữ liệu giao thông trên Kaggle. Sau khi tải về, dữ liệu được đọc vào bằng `pandas` để phục vụ cho các bước xử lý tiếp theo.

### Bước 2. Làm sạch và tiền xử lý
Ở bước này, dữ liệu được chuẩn hóa để sẵn sàng cho việc phân tích:
- Chuyển cột thời gian sang kiểu `datetime`
- Kiểm tra giá trị thiếu
- Tạo thêm các thuộc tính thời gian như:
  - `hour`
  - `day`
  - `month`
  - `dayofweek`
  - `is_weekend`
- Chuẩn hóa hoặc mã hóa các thuộc tính thời tiết khi cần thiết

### Bước 3. Khám phá và phân tích dữ liệu
Dữ liệu được phân tích để tìm ra các đặc điểm nổi bật:
- Khung giờ có lưu lượng cao nhất trong ngày
- Sự khác biệt giữa ngày thường và cuối tuần
- Ảnh hưởng của điều kiện thời tiết đến lưu lượng xe
- Phát hiện các điểm dữ liệu bất thường hoặc đột biến

### Bước 4. Xây dựng mô hình dự báo
Sử dụng các đặc trưng thời gian và thời tiết để xây dựng mô hình dự báo lưu lượng giao thông.  
Trong bài, mô hình **Random Forest Regressor** được áp dụng để:
- Huấn luyện trên dữ liệu lịch sử
- Dự đoán lưu lượng giao thông
- Dự báo cho **24 giờ tiếp theo** hoặc **7 ngày tiếp theo**

Các chỉ số đánh giá thường dùng:
- `MAE` – Mean Absolute Error
- `RMSE` – Root Mean Squared Error

### Bước 5. Trực quan hóa và báo cáo
Kết quả được trình bày thông qua các biểu đồ:
- Biểu đồ lưu lượng theo thời gian
- Biểu đồ theo giờ trong ngày
- Biểu đồ theo ngày trong tuần
- Biểu đồ so sánh giá trị thực tế và giá trị dự đoán
- Biểu đồ dự báo lưu lượng trong tương lai

## 📊 Kết quả nổi bật
Từ quá trình phân tích, có thể rút ra một số nhận xét:
- Lưu lượng giao thông thường tăng mạnh vào **giờ đi làm** và **giờ tan tầm**
- Các ngày trong tuần có lưu lượng cao hơn cuối tuần
- Điều kiện thời tiết có ảnh hưởng nhất định đến mật độ giao thông
- Mô hình Random Forest cho kết quả dự báo khá tốt trên dữ liệu thực tế

## 📂 Cấu trúc thư mục gợi ý
```bash
traffic-analysis/
│── data/
│   ├── traffic.csv
│   ├── traffic_cleaned.csv
│
│── notebooks/
│   ├── BTL_buoc1_den_buoc5.ipynb
│   ├── next24h.ipynb
│   ├── next7d.ipynb
│
│── scripts/
│   ├── next7d.py
│
│── output/
│   ├── forecast_next_7_days.csv
│   ├── images/
│
│── README.md
```

## ▶️ Cách chạy dự án

### 1. Clone project
```bash
git clone https://github.com/NguyenDung04/traffic-flow-time-analysis traffic-analysis 
```
```bash 
cd traffic-analysis
```

### 2. Tạo môi trường ảo
```bash
python -m venv .venv
```

### 3. Kích hoạt môi trường ảo
Windows:
```bash
.venv\Scripts\activate
```

macOS / Linux:
```bash
source .venv/bin/activate
```

### 4. Cài đặt thư viện
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 5. Chạy Jupyter Notebook
```bash
jupyter notebook
```

## 💡 Hướng phát triển
Trong tương lai, dự án có thể được mở rộng theo các hướng:
- Tích hợp dữ liệu thời tiết thực tế theo thời gian thực
- So sánh nhiều mô hình dự báo khác nhau
- Xây dựng giao diện dashboard trực quan
- Dự báo theo nhiều mốc thời gian khác nhau
- Ứng dụng vào bài toán hỗ trợ điều phối giao thông thông minh

## 👨‍💻 Tác giả
Học phần chuyên đề 3  
Sinh viên thực hiện: *[Nhóm 8]*

## 📚 Ghi chú
Dự án mang tính chất học tập và nghiên cứu, phục vụ cho việc thực hành:
- xử lý dữ liệu
- phân tích dữ liệu theo thời gian
- trực quan hóa dữ liệu
- xây dựng mô hình học máy cơ bản
