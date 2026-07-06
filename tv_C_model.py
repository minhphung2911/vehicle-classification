import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_v1(input_shape=(4096,), num_classes=4):
    """
    Xây dựng Model V1: Cấu trúc CNN đơn giản dùng hàm kích hoạt ReLU (baseline model).
    Có tầng Reshape nội bộ để khôi phục đặc trưng ảnh từ vector phẳng 4096 chiều.
    
    Tham số:
        input_shape (tuple): Kích thước của vector đầu vào (mặc định: 4096 từ ảnh phẳng 64x64).
        num_classes (int): Số lượng lớp phân loại (mặc định: 4).
        
    Trả về:
        keras.Model: Model V1 đã được compile.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Phục hồi ảnh 2D từ vector 1 chiều
        layers.Reshape((64, 64, 1), name='reshape_input'),
        
        # Tích chập khối 1
        layers.Conv2D(16, (3, 3), activation='relu', name='conv_1'),
        layers.MaxPooling2D((2, 2), name='pool_1'),
        
        # Tích chập khối 2
        layers.Conv2D(32, (3, 3), activation='relu', name='conv_2'),
        layers.MaxPooling2D((2, 2), name='pool_2'),
        
        # Duỗi đặc trưng ra nối vào Dense
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense_1_relu'),
        
        # Tầng đầu ra
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ], name="CNN_V1_ReLU")
    
    # Compile model theo yêu cầu phân công
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_model_v2(input_shape=(4096,), num_classes=4):
    """
    Xây dựng Model V2: Cấu trúc CNN phức tạp hơn dùng hàm kích hoạt Sigmoid (có Dropout).
    Có tầng Reshape nội bộ để khôi phục đặc trưng ảnh từ vector phẳng 4096 chiều.
    
    Tham số:
        input_shape (tuple): Kích thước của vector đầu vào (mặc định: 4096 từ ảnh phẳng 64x64).
        num_classes (int): Số lượng lớp phân loại (mặc định: 4).
        
    Trả về:
        keras.Model: Model V2 đã được compile.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Phục hồi ảnh 2D từ vector 1 chiều
        layers.Reshape((64, 64, 1), name='reshape_input'),
        
        # Tích chập khối 1
        layers.Conv2D(32, (3, 3), activation='sigmoid', name='conv_1'),
        layers.MaxPooling2D((2, 2), name='pool_1'),
        
        # Tích chập khối 2
        layers.Conv2D(64, (3, 3), activation='sigmoid', name='conv_2'),
        layers.MaxPooling2D((2, 2), name='pool_2'),
        
        # Tích chập khối 3
        layers.Conv2D(128, (3, 3), activation='sigmoid', name='conv_3'),
        layers.MaxPooling2D((2, 2), name='pool_3'),
        
        # Duỗi đặc trưng ra nối vào Dense
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='sigmoid', name='dense_1_sigmoid'),
        layers.Dropout(0.4, name='dropout_1'),
        
        # Tầng đầu ra
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ], name="CNN_V2_Sigmoid")
    
    # Compile model theo yêu cầu phân công
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("  KHỞI TẠO VÀ HIỂN THỊ KIẾN TRÚC MODEL TÍCH CHẬP CNN (TV C)")
    print("=" * 60)
    
    # Kiểm tra xem GPU có khả dụng hay không
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[THÔNG TIN] GPU khả dụng: {gpus}\n")
    else:
        print("[THÔNG TIN] Chạy trên CPU.\n")
        
    # Tạo và in thông tin Model V1
    print("--- KHỞI TẠO MODEL V1 (CNN ĐƠN GIẢN - RELU) ---")
    model_v1 = build_model_v1()
    model_v1.summary()
    print("\n" + "=" * 60 + "\n")
    
    # Tạo và in thông tin Model V2
    print("--- KHỞI TẠO MODEL V2 (CNN PHỨC TẠP - SIGMOID) ---")
    model_v2 = build_model_v2()
    model_v2.summary()
    print("=" * 60)
