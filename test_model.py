import os
import cv2
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report

TEST_DIR = "dataset/test"
IMG_SIZE = 100

# =========================
# LOAD MODELS
# =========================

with open("models/pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

with open("models/knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

with open("models/labels.pkl", "rb") as f:
    label_map = pickle.load(f)

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# reverse map: name -> label int
reverse_map = {v: k for k, v in label_map.items()}

X_test = []
y_test = []
skipped = 0

for person_name in sorted(os.listdir(TEST_DIR)):

    person_path = os.path.join(TEST_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    # Robustness: skip test folders not in training set
    if person_name not in reverse_map:
        print(f"WARNING: '{person_name}' in test set but not in training set — skipping")
        continue

    label = reverse_map[person_name]

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        if len(faces) == 0:
            print(f"  No face detected in {image_path}")
            skipped += 1
            continue

        fx, fy, fw, fh = faces[0]   # BUG FIX: renamed to fx,fy

        face = gray[fy:fy+fh, fx:fx+fw]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_vector = face.flatten()

        X_test.append(face_vector)
        y_test.append(label)

print(f"\nTest images loaded: {len(X_test)}")
print(f"Skipped (no face detected): {skipped}")

if len(X_test) == 0:
    print("ERROR: No test images loaded. Check your test directory structure.")
    exit()

X_test = np.array(X_test)

# Project test faces into the SAME eigenspace as training
X_test_pca = pca.transform(X_test)

predictions = knn.predict(X_test_pca)

accuracy = accuracy_score(y_test, predictions)
print(f"\nOverall Accuracy: {accuracy * 100:.1f}%")

# Per-person breakdown helps find which person is causing confusion
print("\nPer-person report:")
target_names = [label_map[i] for i in sorted(label_map.keys())]
print(classification_report(y_test, predictions, target_names=target_names))

# Show distances for debugging Abdr vs Zai confusion
print("Sample distances (to tune Unknown threshold):")
distances, _ = knn.kneighbors(X_test_pca)
for i in range(min(10, len(X_test))):
    true_name = label_map[y_test[i]]
    pred_name = label_map[predictions[i]]
    correct = "OK" if y_test[i] == predictions[i] else "WRONG"
    print(f"  {true_name:10s} -> {pred_name:10s}  dist={distances[i][0]:.1f}  [{correct}]")