# Phân loại phương tiện giao thông bằng Neural Network (MLP)

Dự án tiểu luận môn học **Data Mining** (Khai phá dữ liệu) nhằm mục đích phân loại 4 nhóm phương tiện giao thông phổ biến: **Bus (Xe buýt), Car (Ô tô), Motorcycle (Xe máy), và Truck (Xe tải)** sử dụng mạng Neural Network nhiều lớp (Multi-layer Perceptron - MLP) xây dựng trên nền tảng TensorFlow/Keras.

---

## 📂 Cấu trúc thư mục dự án

```text
vehicle-classification/
│
├── Dataset/                          # Dữ liệu ảnh thô (phân chia theo thư mục lớp)
│   ├── Bus/
│   ├── Car/
│   ├── Motorcycle/
│   └── Truck/
│
├── data/                             # Dữ liệu sau tiền xử lý (Thành viên B xuất ra)
│   ├── images.npy                    # Vector ảnh dạng ma trận (N, 4096)
│   ├── labels.npy                    # Mảng nhãn tương ứng (N,)
│   └── class_names.npy               # Danh sách tên các nhãn
│
├── models/                           # Thư mục lưu trữ Model đã train (Thành viên D xuất ra)
│   ├── model_v1_relu.h5              # Model V1 (MLP ReLU đơn giản, không Dropout)
│   └── model_v2_sigmoid.h5           # Model V2 (MLP Sigmoid phức tạp, có Dropout)
│
├── outputs/                          # Kết quả đồ thị, báo cáo thực nghiệm
│   ├── loss_curve_v1.png             # Đường cong Loss/Accuracy của Model V1
│   ├── loss_curve_v2.png             # Đường cong Loss/Accuracy của Model V2
│   ├── confusion_matrix_v1.png       # Ma trận nhầm lẫn của Model V1 trên tập Test
│   └── confusion_matrix_v2.png       # Ma trận nhầm lẫn của Model V2 trên tập Test
│
├── examples/                         # Ảnh mẫu test nhanh tích hợp trên Gradio UI
│   ├── bus.jpg
│   ├── car.jpg
│   ├── motorcycle.jpg
│   ├── truck.jpg
│   └── difficult.jpg                 # Ảnh khó để thử nghiệm lỗi phân loại
│
├── tv_A_demo.py                      # Code giao diện Gradio Demo (Thành viên A)
├── tv_B_preprocessing.py             # Code đọc ảnh thô & tiền xử lý (Thành viên B)
├── tv_C_model.py                     # Code định nghĩa cấu trúc mạng MLP (Thành viên C)
├── tv_D_train.ipynb                  # Notebook chia data, Train & Đánh giá (Thành viên D)
│
├── requirements.txt                  # Các thư viện phụ thuộc của dự án
└── README.md                         # Hướng dẫn dự án này
```

---

## 🛠️ Hướng dẫn cài đặt môi trường (Bằng Python venv chuẩn)

> [!IMPORTANT]
> **Lưu ý về phiên bản Python:** 
> Dự án sử dụng thư viện **TensorFlow**. Hiện tại TensorFlow chưa hỗ trợ các phiên bản Python quá mới (như Python 3.14 trở lên trên Windows). Khuyên dùng **Python 3.11** hoặc **Python 3.12** để tránh lỗi cài đặt.

Thực hiện các bước sau để thiết lập môi trường chạy dự án sau khi clone:

1.  **Mở terminal** tại thư mục gốc của dự án.
2.  **Tạo môi trường ảo** đặt tên là `.venv`:
    ```bash
    python -m venv .venv
    ```
3.  **Kích hoạt môi trường ảo**:
    *   **Windows (PowerShell):**
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   **Windows (CMD):**
        ```cmd
        .venv\Scripts\activate.bat
        ```
    *   **macOS / Linux:**
        ```bash
        source .venv/bin/activate
        ```
    *(Sau khi kích hoạt, bạn sẽ thấy ký tự `(.venv)` xuất hiện ở đầu dòng lệnh).*
4.  **Cập nhật pip và cài đặt các thư viện** từ file [requirements.txt](file:///d:/GitHub/vehicle-classification/requirements.txt):
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## 🚀 Quy trình chạy dự án từng bước

Sau khi kích hoạt môi trường ảo ở phần trên, thực hiện theo thứ tự luồng xử lý:

### Bước 1: Tiền xử lý dữ liệu (Thành viên B)
Nạp dữ liệu ảnh gốc ban đầu nằm trong [Dataset/](file:///d:/GitHub/vehicle-classification/Dataset) đưa về dạng chuẩn hóa vector:
```bash
python tv_B_preprocessing.py
```
*Đầu ra:* Xuất hiện các file `images.npy`, `labels.npy` và `class_names.npy` trong thư mục `data/`.

### Bước 2: Kiểm tra cấu trúc mạng MLP (Thành viên C)
Xác thực kiến trúc các Layer và số lượng tham số huấn luyện của hai mô hình:
```bash
python tv_C_model.py
```
*Đầu ra:* Hiển thị bảng tóm tắt chi tiết `model.summary()` trên màn hình terminal.

### Bước 3: Huấn luyện và Đánh giá (Thành viên D)
1.  Mở Jupyter Notebook [tv_D_train.ipynb](file:///d:/GitHub/vehicle-classification/tv_D_train.ipynb) trong VS Code hoặc Jupyter Lab.
2.  Chọn Kernel chạy là môi trường ảo `.venv` bạn vừa tạo.
3.  Chạy toàn bộ các Cell để huấn luyện cả 2 model.
*Đầu ra:* 
*   Lưu model tốt nhất vào thư mục `models/` (`model_v1_relu.h5` và `model_v2_sigmoid.h5`).
*   Xuất các biểu đồ đánh giá chất lượng (Loss curves, Confusion matrices) lưu vào thư mục `outputs/`.

### Bước 4: Chạy giao diện tương tác Demo (Thành viên A)
Khởi chạy giao diện Gradio trên trình duyệt web để test ảnh xe bất kỳ:
```bash
python tv_A_demo.py
```
*Đầu ra:* Terminal xuất hiện đường dẫn `http://127.0.0.1:7860`. Hãy mở link này bằng trình duyệt của bạn để trải nghiệm tải ảnh xe lên và dự đoán loại phương tiện.