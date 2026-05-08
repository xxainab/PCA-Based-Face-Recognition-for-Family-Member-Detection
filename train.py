import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle

# =========================
# SETTINGS
# =========================
TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"
IMG_SIZE  = 100

# =========================
# LOAD FACE CASCADE
# =========================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# =========================
# DATA STORAGE
# =========================
X = []
y = []
label_map = {}
current_label = 0

# =========================
# HELPER: preprocess one image -> face vector (or None)
# =========================
def extract_face_vector(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=8, minSize=(80, 80)
    )

    # If strict detection fails, try a looser pass
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )

    # If still nothing, use the whole image as the face
    if len(faces) == 0:
        face = gray
    else:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y_face, w, h = faces[0]
        
        face = gray[y_face:y_face+h, x:x+w]

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.equalizeHist(face)
    # NEW: Standardize pixel values
    face = face.astype(np.float32)
    face = (face - np.mean(face)) / (np.std(face) + 1e-5)
    return face.flatten()

# =========================
# LOAD TRAINING IMAGES
# =========================
for person_name in sorted(os.listdir(TRAIN_DIR)):
    person_path = os.path.join(TRAIN_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name
    print(f"\nLoading images for: {person_name}")
    loaded = 0

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        vec = extract_face_vector(image_path)
        if vec is None:
            print(f"  Could not read: {image_name}")
            continue
        X.append(vec)
        y.append(current_label)
        loaded += 1
        print(f"  Loaded: {image_name}")

    print(f"  -> {loaded} images loaded for {person_name}")
    current_label += 1

# =========================
# CONVERT TO NUMPY
# =========================
X = np.array(X)
y = np.array(y)
print("\nDataset Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# MEAN FACE
# =========================
mean_face = np.mean(X, axis=0)
mean_face_image = mean_face.reshape(IMG_SIZE, IMG_SIZE)
plt.imshow(mean_face_image, cmap='gray')
plt.title("Mean Face")
plt.axis('off')
plt.savefig("output/mean_face.png")
plt.show()

# =========================
# APPLY PCA
# =========================
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X)
print("\nPCA Applied")
print("Reduced Shape:", X_pca.shape)
print(f"Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# =========================
# EIGENFACES
# =========================
eigenfaces = pca.components_
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    if i >= len(eigenfaces):
        break
    eigenface = eigenfaces[i].reshape(IMG_SIZE, IMG_SIZE)
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("output/eigenfaces.png")
plt.show()

# =========================
# TRAIN KNN
# =========================
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_pca, y)
print("\nTraining Complete")

# =========================
# TRAINING ACCURACY
# =========================
train_preds = knn.predict(X_pca)
train_correct = np.sum(train_preds == y)
print(f"\nTraining Accuracy: {train_correct}/{len(y)} = {train_correct/len(y)*100:.1f}%")

# =========================
# SAVE MODELS
# =========================
with open("models/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)
with open("models/labels.pkl", "wb") as f:
    pickle.dump(label_map, f)
print("\nModels Saved Successfully")

# =========================
# TEST SET EVALUATION
# =========================
if not os.path.isdir(TEST_DIR):
    print("\nNo test directory found -- skipping test evaluation.")
else:
    print(f"\n{'='*55}")
    print("  TEST SET EVALUATION")
    print(f"{'='*55}")

    # Build reverse map: name -> label
    name_to_label = {v: k for k, v in label_map.items()}

    X_test, y_test, test_names = [], [], []

    for person_name in sorted(os.listdir(TEST_DIR)):
        person_path = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        if person_name not in name_to_label:
            print(f"  Warning: '{person_name}' not in training labels, skipping.")
            continue

        true_label = name_to_label[person_name]

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            vec = extract_face_vector(image_path)
            if vec is None:
                continue
            X_test.append(vec)
            y_test.append(true_label)
            test_names.append((person_name, image_name))

    if len(X_test) == 0:
        print("  No test images could be loaded.")
    else:
        X_test     = np.array(X_test)
        y_test     = np.array(y_test)
        X_test_pca = pca.transform(X_test)
        predictions  = knn.predict(X_test_pca)
        distances, _ = knn.kneighbors(X_test_pca)
        min_dists    = distances[:, 0]

        correct = 0
        print(f"\n  {'File':<45} {'True':<8} {'Pred':<10} {'Dist':>7}")
        print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*7}")

        for i, (person_name, image_name) in enumerate(test_names):
            pred_label = predictions[i]
            pred_name  = label_map[pred_label]
            dist       = min_dists[i]
            ok         = (pred_label == y_test[i])
            sym        = "v" if ok else "x"
            if ok:
                correct += 1
            fname = image_name if len(image_name) <= 44 else image_name[:41] + "..."
            print(f"  {sym} {fname:<45} {person_name:<8} {pred_name:<10} {dist:>7.1f}")

        total = len(y_test)
        acc   = correct / total * 100
        print(f"\n  Accuracy: {correct}/{total} = {acc:.1f}%")

        # Per-person breakdown
        print(f"\n  Per-person breakdown:")
        for lbl, name in label_map.items():
            mask = (y_test == lbl)
            person_total = np.sum(mask)
            if person_total == 0:
                continue
            person_correct = np.sum(predictions[mask] == lbl)
            print(f"    {name:<15} {person_correct}/{person_total} = "
                  f"{person_correct/person_total*100:.0f}%")

    print(f"{'='*55}\n")