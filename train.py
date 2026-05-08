import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# SETTINGS
# =========================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMG_SIZE = 100
N_COMPONENTS = 20

# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# =========================
# CREATE OUTPUT FOLDERS
# =========================

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# =========================
# FACE EXTRACTION
# =========================

def extract_face(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(80, 80)
    )

    if len(faces) > 0:

        faces = sorted(
            faces,
            key=lambda f: f[2] * f[3],
            reverse=True
        )

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

    else:

        face = gray

    face = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE)
    )

    face = cv2.equalizeHist(face)

    return face.flatten()

# =========================
# TRAIN PERSON MODELS
# =========================

models = {}

print("\n====================================")
print(" TRAINING PCA MODELS")
print("====================================\n")

for person_name in sorted(os.listdir(TRAIN_DIR)):

    person_path = os.path.join(
        TRAIN_DIR,
        person_name
    )

    if not os.path.isdir(person_path):
        continue

    print(f"\nTraining model for: {person_name}")

    X = []

    for image_name in os.listdir(person_path):

        image_path = os.path.join(
            person_path,
            image_name
        )

        vec = extract_face(image_path)

        if vec is not None:
            X.append(vec)
            print(f"  Loaded: {image_name}")

    X = np.array(X)

    print("  Shape:", X.shape)

    # =========================
    # PCA
    # =========================

    n_comp = min(
        N_COMPONENTS,
        len(X)
    )

    pca = PCA(n_components=n_comp)

    X_pca = pca.fit_transform(X)

    # =========================
    # SAVE MODEL
    # =========================

    models[person_name] = {

        "pca": pca,
        "train_vectors": X,
        "train_pca": X_pca

    }

    # =========================
    # MEAN FACE
    # =========================

    mean_face = pca.mean_.reshape(
        IMG_SIZE,
        IMG_SIZE
    )

    plt.imshow(
        mean_face,
        cmap='gray'
    )

    plt.title(
        f"Mean Face - {person_name}"
    )

    plt.axis('off')

    plt.savefig(
        f"output/mean_{person_name}.png"
    )

    plt.close()

    # =========================
    # EIGENFACES
    # =========================

    eigenfaces = pca.components_

    cols = min(5, len(eigenfaces))
    rows = int(np.ceil(len(eigenfaces)/cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(10, 5)
    )

    axes = np.array(axes).reshape(-1)

    for i in range(len(eigenfaces)):

        eigenface = eigenfaces[i].reshape(
            IMG_SIZE,
            IMG_SIZE
        )

        axes[i].imshow(
            eigenface,
            cmap='gray'
        )

        axes[i].set_title(
            f"E{i+1}"
        )

        axes[i].axis('off')

    for j in range(len(eigenfaces), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    plt.savefig(
        f"output/eigenfaces_{person_name}.png"
    )

    plt.close()

    print(
        f"  Variance Explained: "
        f"{sum(pca.explained_variance_ratio_)*100:.1f}%"
    )

# =========================
# SAVE ALL MODELS
# =========================

with open(
    "models/person_models.pkl",
    "wb"
) as f:

    pickle.dump(models, f)

print("\nModels Saved Successfully")

# =========================
# TEST EVALUATION
# =========================

print("\n====================================")
print(" TEST SET EVALUATION")
print("====================================\n")

THRESHOLD = 8000

overall_correct = 0
overall_total = 0

for true_person in sorted(os.listdir(TEST_DIR)):

    person_path = os.path.join(
        TEST_DIR,
        true_person
    )

    if not os.path.isdir(person_path):
        continue

    print(f"\nTesting: {true_person}")

    correct = 0
    total = 0

    for image_name in os.listdir(person_path):

        image_path = os.path.join(
            person_path,
            image_name
        )

        vec = extract_face(image_path)

        if vec is None:
            continue

        best_person = "Unknown"
        best_error = float('inf')

        # =========================
        # CHECK AGAINST ALL MODELS
        # =========================

        for person_name, model in models.items():

            pca = model["pca"]

            projected = pca.transform(
                vec.reshape(1, -1)
            )

            reconstructed = pca.inverse_transform(
                projected
            )

            error = np.linalg.norm(
                vec - reconstructed
            )

            if error < best_error:

                best_error = error
                best_person = person_name

        if best_error > THRESHOLD:
            best_person = "Unknown"

        ok = (best_person == true_person)

        if ok:
            correct += 1
            overall_correct += 1

        total += 1
        overall_total += 1

        symbol = "v" if ok else "x"

        print(
            f"{symbol} {image_name:<30} "
            f"Pred: {best_person:<10} "
            f"Error: {best_error:.1f}"
        )

    accuracy = correct / total * 100

    print(
        f"\n{true_person} Accuracy: "
        f"{correct}/{total} = {accuracy:.1f}%"
    )

overall_accuracy = overall_correct / overall_total * 100

print("\n====================================")
print(
    f"OVERALL ACCURACY: "
    f"{overall_correct}/{overall_total} "
    f"= {overall_accuracy:.1f}%"
)
print("====================================")