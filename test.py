# pca_svd_compare_images.py
import os, cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Cấu hình ===
DATA_DIR = "Datasets/archive"   # ví dụ: "att_faces" (ORL), hoặc "dataset"
IMG_SIZE = (100, 100)    # resize về kích thước thống nhất

# === 1) Load ảnh: mỗi thư mục con = 1 người ===
def load_images(root, size=(100,100)):
    X, y, paths = [], [], []
    root = Path(root)
    for label in sorted(p.name for p in root.iterdir() if p.is_dir()):
        folder = root/label
        for f in sorted(folder.iterdir()):
            if not f.is_file(): 
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            X.append(img.flatten().astype(np.float64))
            y.append(label)
            paths.append(str(f))
    X = np.array(X)  # (n_samples, n_pixels)
    y = np.array(y)
    return X, y, paths

X, y, paths = load_images(DATA_DIR, IMG_SIZE)
n, d = X.shape
print(f"[INFO] Loaded: {n} images, {d} pixels each. Classes: {len(np.unique(y))}")

# === 2) Hai kiểu chuẩn hóa ===
# (A) ĐÚNG CHO PCA: chuẩn hóa theo cột (zero-mean toàn dataset theo từng pixel)
mean_global = X.mean(axis=0)
Xc_axis0 = X - mean_global

# (B) SAI CHO PCA: chuẩn hóa từng ảnh (mỗi ảnh có mean riêng = 0)
mean_per_img = X.mean(axis=1, keepdims=True)
Xc_axis1 = X - mean_per_img

# === 3) PCA bằng SVD cho cả hai cách ===
def pca_svd(Xc, k=None):
    # Xc đã zero-mean theo cách nào đó
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # Xc = U S Vt
    if k is None: 
        k = min(100, Vt.shape[0])  # mặc định lấy nhiều nhất 100 PC
    comps = Vt[:k]                 # (k, d)
    # Tọa độ PC: Xc @ V_k^T = U_k S_k
    Z = Xc @ comps.T               # (n, k)
    # Phương sai giải thích:
    var = (S**2) / (Xc.shape[0] - 1)
    exp_var = var[:k]
    exp_ratio = exp_var / var.sum()
    return comps, Z, exp_var, exp_ratio

K = 100  # số thành phần chính để hiển thị/trích xuất
comps0, Z0, var0, ratio0 = pca_svd(Xc_axis0, K)
comps1, Z1, var1, ratio1 = pca_svd(Xc_axis1, K)

print(f"[Axis=0] EVR (PC1..PC5): {np.round(ratio0[:5], 4)} | Total 50PC: {ratio0[:50].sum():.4f}")
print(f"[Axis=1] EVR (PC1..PC5): {np.round(ratio1[:5], 4)} | Total 50PC: {ratio1[:50].sum():.4f}")

# === 4) Vẽ minh họa ===
H, W = IMG_SIZE

# 4.1 Mean face (toàn tập) và một ảnh mẫu
plt.figure(figsize=(8,3))
plt.subplot(1,3,1); plt.imshow(X[0].reshape(H,W), cmap='gray'); plt.title('Ảnh mẫu'); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(mean_global.reshape(H,W), cmap='gray'); plt.title('Mean face (axis=0)'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow((X[0]-X[0].mean()).reshape(H,W), cmap='gray'); plt.title('Ảnh trừ mean riêng (axis=1)'); plt.axis('off')
plt.tight_layout(); plt.show()

# 4.2 Eigenfaces (axis=0): 12 thành phần đầu
efaces0 = comps0.reshape((K, H, W))
rows, cols = 3, 4
plt.figure(figsize=(10,7))
for i in range(rows*cols):
    plt.subplot(rows, cols, i+1)
    plt.imshow(efaces0[i], cmap='gray')
    plt.title(f"PC{i+1}")
    plt.axis('off')
plt.suptitle("Eigenfaces (axis=0 zero-mean)", y=0.98)
plt.tight_layout(); plt.show()

# 4.3 So sánh chiếu 2D (PC1–PC2) giữa hai cách chuẩn hóa
# Mã màu theo lớp (tối đa 10 lớp để rõ; nếu nhiều lớp, chỉ lấy 10 lớp đầu)
classes = sorted(np.unique(y))
palette = {c: i for i, c in enumerate(classes[:10])}
colors0 = np.array([palette.get(c, 0) for c in y])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(Z0[:,0], Z0[:,1], c=colors0, s=18, alpha=0.8)
plt.title("PC1–PC2 | zero-mean theo CỘT (axis=0)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.scatter(Z1[:,0], Z1[:,1], c=colors0, s=18, alpha=0.8)
plt.title("PC1–PC2 | zero-mean từng ẢNH (axis=1)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()

# 4.4 Biểu đồ tích lũy phương sai giải thích (EVR)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(ratio0[:K]), label='Axis=0 (đúng cho PCA)')
plt.plot(np.cumsum(ratio1[:K]), label='Axis=1 (sai mục tiêu PCA)', linestyle='--')
plt.xlabel("Số thành phần chính (k)")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("So sánh EVR tích lũy")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
