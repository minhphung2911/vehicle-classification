import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# IMPORT HÀM TIỀN XỬ LÝ TỪ TV B

from tv_B_preprocessing import preprocess_image

# LOAD MODEL ĐÃ TRAIN

MODEL_PATH = "models/model_v1_relu.h5"

print("=" * 60)
print("ĐANG LOAD MODEL...")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print(f"[INFO] Đã load model từ: {MODEL_PATH}")

# LOAD TÊN CÁC LỚP

try:
    class_names = np.load(
        "data/class_names.npy",
        allow_pickle=True
    )

    print("[INFO] Đã đọc class_names.npy")
    print(class_names)

except:

    print("[CẢNH BÁO] Không tìm thấy class_names.npy")

    class_names = np.array([
        "Bus",
        "Car",
        "Motorcycle",
        "Truck"
    ])

# HÀM VẼ BIỂU ĐỒ XÁC SUẤT

def create_probability_chart(probabilities):

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        class_names,
        probabilities,
        color=["#4C72B0", "#DD8452",
               "#55A868", "#C44E52"]
    )

    ax.set_ylim(0, 1)

    ax.set_title(
        "Prediction Probability",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel("Probability")

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height:.2f}",
            ha='center'
        )

    plt.tight_layout()

    return fig

# HÀM DỰ ĐOÁN

def classify_vehicle(image_path):

    if image_path is None:
        return "Vui lòng tải ảnh lên.", None

    try:

        # TIỀN XỬ LÝ ẢNH

        image_vector = preprocess_image(image_path)

        if image_vector is None:
            return "Không thể xử lý ảnh.", None

        # shape: (4096,) -> (1,4096)

        image_vector = np.expand_dims(
            image_vector,
            axis=0
        )

        # DỰ ĐOÁN

        prediction = model.predict(
            image_vector,
            verbose=0
        )

        probabilities = prediction[0]

        predicted_index = np.argmax(
            probabilities
        )

        predicted_label = class_names[
            predicted_index
        ]

        confidence = (
            probabilities[predicted_index]
            * 100
        )

        result = (
            f"Loại xe dự đoán: "
            f"{predicted_label}\n\n"
            f"Độ tin cậy: "
            f"{confidence:.2f}%"
        )

        fig = create_probability_chart(
            probabilities
        )

        return result, fig

    except Exception as e:

        return f"Lỗi:\n{str(e)}", None

# GIAO DIỆN GRADIO

demo = gr.Interface(

    fn=classify_vehicle,

    inputs=gr.Image(
        type="filepath",
        label="Tải ảnh phương tiện"
    ),

    outputs=[
        gr.Textbox(
            label="Kết quả dự đoán"
        ),

        gr.Plot(
            label="Biểu đồ xác suất"
        )
    ],

    title="PHÂN LOẠI PHƯƠNG TIỆN GIAO THÔNG BẰNG NEURAL NETWORK",

    description="""
Upload ảnh xe để hệ thống dự đoán.

Mô hình sử dụng MLP Neural Network.
""",

    examples=[
        ["examples/bus.jpg"],
        ["examples/car.jpg"],
        ["examples/motorcycle.jpg"],
        ["examples/truck.jpg"],
        ["examples/difficult.jpg"]
    ]
)

# CHẠY CHƯƠNG TRÌNH

if __name__ == "__main__":

    print("=" * 60)
    print("KHỞI ĐỘNG GIAO DIỆN DEMO")
    print("=" * 60)

    demo.launch()
