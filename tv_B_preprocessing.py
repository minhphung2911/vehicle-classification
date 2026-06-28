import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter

DATASET_DIR = "dataset"
OUTPUT_DIR  = "data"
IMG_SIZE    = 64
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preprocess_image(image_input, img_size=IMG_SIZE):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            print(f"  [CẢNH BÁO] Không đọc được ảnh: {image_input}")
            return None
    else:
        img = image_input.copy()

    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized    = cv2.resize(gray, (img_size, img_size))
    normalized = resized.astype(np.float32) / 255.0
    flattened  = normalized.flatten()

    return flattened


def load_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE):
    images      = []
    labels      = []

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: '{dataset_dir}'")

    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    print(f"Tìm thấy {len(class_names)} lớp: {class_names}\n")

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        file_list = [f for f in os.listdir(class_dir) if f.lower().endswith(IMG_EXTENSIONS)]

        print(f"  Xử lý lớp [{label_idx}] '{class_name}' — {len(file_list)} ảnh...")

        count = 0
        for filename in file_list:
            vector = preprocess_image(os.path.join(class_dir, filename), img_size)
            if vector is not None:
                images.append(vector)
                labels.append(label_idx)
                count += 1

        print(f"    ✓ {count}/{len(file_list)} ảnh thành công")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\nTổng ảnh hợp lệ : {len(images)}")
    print(f"Kích thước vector: {images.shape[1]} chiều ({img_size}×{img_size})")

    return images, labels, class_names


def save_output(images, labels, class_names, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "images.npy"),      images)
    np.save(os.path.join(output_dir, "labels.npy"),      labels)
    np.save(os.path.join(output_dir, "class_names.npy"), class_names)

    print(f"\n Đã lưu vào '{output_dir}/':")
    print(f"   images.npy      — shape {images.shape}")
    print(f"   labels.npy      — shape {labels.shape}")
    print(f"   class_names.npy — {class_names}")


def visualize_data(images, labels, class_names, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    counter = Counter(labels)
    counts  = [counter[i] for i in range(len(class_names))]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_names, counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    plt.title("Phân phối số lượng ảnh theo từng lớp xe", fontsize=14, fontweight="bold")
    plt.xlabel("Loại xe")
    plt.ylabel("Số lượng ảnh")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phan_phoi_nhan.png"), dpi=120)
    plt.show()

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        class_indices = np.where(labels == idx)[0]
        if len(class_indices) == 0:
            ax.axis("off")
            continue
        sample_img = images[class_indices[0]].reshape(IMG_SIZE, IMG_SIZE)
        ax.imshow(sample_img, cmap="gray")
        ax.set_title(class_name, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Ảnh mẫu đại diện từng loại xe", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anh_mau_tung_lop.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    images, labels, class_names = load_dataset()
    visualize_data(images, labels, class_names)
    save_output(images, labels, class_names)
