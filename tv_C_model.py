import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_v1(input_shape=(4096,), num_classes=4):
    """
    Xây dựng Model V1: Cấu trúc MLP đơn giản dùng hàm kích hoạt ReLU (baseline model).
    
    Tham số:
        input_shape (tuple): Kích thước của vector đặc trưng đầu vào (mặc định: 4096 từ ảnh 64x64).
        num_classes (int): Số lượng lớp phân loại (mặc định: 4).
        
    Trả về:
        keras.Model: Model V1 đã được compile.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Ẩn tầng 1
        layers.Dense(128, activation='relu', name='dense_1_relu'),
        
        # Ẩn tầng 2
        layers.Dense(64, activation='relu', name='dense_2_relu'),
        
        # Tầng đầu ra
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ], name="MLP_V1_ReLU")
    
    # Compile model theo yêu cầu phân công
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_model_v2(input_shape=(4096,), num_classes=4):
    """
    Xây dựng Model V2: Cấu trúc MLP phức tạp hơn dùng hàm kích hoạt Sigmoid (có Dropout).
    
    Tham số:
        input_shape (tuple): Kích thước của vector đặc trưng đầu vào (mặc định: 4096 từ ảnh 64x64).
        num_classes (int): Số lượng lớp phân loại (mặc định: 4).
        
    Trả về:
        keras.Model: Model V2 đã được compile.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Ẩn tầng 1 (nhiều nơ-ron hơn)
        layers.Dense(512, activation='sigmoid', name='dense_1_sigmoid'),
        layers.Dropout(0.4, name='dropout_1'),
        
        # Ẩn tầng 2
        layers.Dense(256, activation='sigmoid', name='dense_2_sigmoid'),
        layers.Dropout(0.4, name='dropout_2'),
        
        # Ẩn tầng 3
        layers.Dense(128, activation='sigmoid', name='dense_3_sigmoid'),
        layers.Dropout(0.4, name='dropout_3'),
        
        # Tầng đầu ra
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ], name="MLP_V2_Sigmoid")
    
    # Compile model theo yêu cầu phân công
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("  KHỞI TẠO VÀ HIỂN THỊ KIẾN TRÚC MODEL PHÂN LOẠI XE (TV C)")
    print("=" * 60)
    
    # Kiểm tra xem GPU có khả dụng hay không
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[THÔNG TIN] GPU khả dụng: {gpus}\n")
    else:
        print("[THÔNG TIN] Chạy trên CPU.\n")
        
    # Tạo và in thông tin Model V1
    print("--- KHỞI TẠO MODEL V1 (ĐƠN GIẢN - RELU) ---")
    model_v1 = build_model_v1()
    model_v1.summary()
    print("\n" + "=" * 60 + "\n")
    
    # Tạo và in thông tin Model V2
    print("--- KHỞI TẠO MODEL V2 (PHỨC TẠP - SIGMOID) ---")
    model_v2 = build_model_v2()
    model_v2.summary()
    print("=" * 60)
