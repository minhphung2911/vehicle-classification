vehicle-classification/
│
├── data/                        # TV1 tạo và quản lý
│   ├── raw/                     # ảnh thô chưa xử lý
│   │   ├── train/
│   │   │   ├── car/
│   │   │   ├── motorbike/
│   │   │   ├── bus/
│   │   │   └── truck/
│   │   └── test/
│   │       ├── car/
│   │       ├── motorbike/
│   │       ├── bus/
│   │       └── truck/
│   └── processed/               # TV1 xuất ra, TV2 dùng tiếp
│       ├── images.npy
│       ├── images_flat.npy
│       └── labels.npy
│
├── models/                      # TV2 xuất ra, TV3+TV4 dùng tiếp
│   ├── model_v1_relu.h5
│   └── model_v2_sigmoid.h5
│
├── outputs/                     # TV3 xuất ra, TV4 dùng cho báo cáo
│   ├── confusion_matrix.png
│   ├── classification_report.csv
│   ├── loss_curve_v1.png
│   └── loss_curve_v2.png
│
├── examples/                    # TV4 chuẩn bị ảnh test demo
│   ├── car_test.jpg
│   ├── motorbike_test.jpg
│   ├── bus_test.jpg
│   └── truck_hard.jpg           # ảnh khó để demo phân tích lỗi
│
├── src/                         # SOURCE CODE CHÍNH
│   ├── tv1_preprocessing.py     # TV1
│   ├── tv2_train.py             # TV2
│   ├── tv3_evaluate.py          # TV3
│   └── tv4_demo.py              # TV4
│
├── notebooks/                   # Jupyter Notebook từng thành viên
│   ├── TV1_EDA_Preprocessing.ipynb
│   ├── TV2_Train_Model.ipynb
│   ├── TV3_Evaluate.ipynb
│   └── TV4_Demo.ipynb
│
├── requirements.txt             # TV4 tạo
└── README.md                    # TV4 tạo