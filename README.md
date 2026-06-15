```
vehicle-classification/
│
├── Dataset/                          # TV B quản lý (dữ liệu gốc)
│   ├── Bus/
│   ├── Car/
│   ├── Motorcycle/                   # sửa lại từ "motorcycle"
│   └── Truck/
│
├── data/                             # TV B xuất ra, TV D + TV A dùng
│   ├── images.npy
│   ├── labels.npy
│   ├── class_names.npy
│   ├── anh_mau_tung_lop.png
│   └── phan_phoi_nhan.png
│
├── models/                           # TV D xuất ra, TV A dùng
│   ├── model_v1_relu.h5
│   └── model_v2_sigmoid.h5
│
├── outputs/                          # TV D xuất ra, đưa vào báo cáo
│   ├── loss_curve_v1.png
│   ├── loss_curve_v2.png
│   ├── confusion_matrix.png
│   └── classification_report.csv
│
├── examples/                         # TV A chuẩn bị, dùng khi demo
│   ├── bus_test.jpg
│   ├── car_test.jpg
│   ├── motorcycle_test.jpg
│   ├── truck_test.jpg
│   └── truck_hard.jpg                # ảnh khó để phân tích lỗi
│
├── tv_A_demo.py                      # TV A viết
├── tv_B_preprocessing.py             # TV B viết (đã có)
├── tv_C_model.py                     # TV C viết (bạn)
└── tv_D_train.py                     # TV D viết
```