import cv2
import pickle
import numpy as np

IMG_SIZE = 100

# =========================
# LOAD MODELS
# =========================

with open(
    "models/person_models.pkl",
    "rb"
) as f:

    models = pickle.load(f)

# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

THRESHOLD = 5000

print("\n====================================")
print(" PCA FACE RECOGNITION STARTED")
print(" Press Q to Quit")
print("====================================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
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

        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
        )

        face = cv2.equalizeHist(face)

        vec = face.flatten()

        # =========================
        # COMPARE WITH ALL MODELS
        # =========================

        best_person = "Unknown"
        best_error = float('inf')

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

            print(
                f"{person_name} Error: "
                f"{error:.1f}"
            )

            if error < best_error:

                best_error = error
                best_person = person_name

        # =========================
        # UNKNOWN DETECTION
        # =========================

        if best_error > THRESHOLD:

            best_person = "Unknown"

        print(
            f"\nPrediction: {best_person} "
            f"| Error: {best_error:.1f}\n"
        )

        # =========================
        # DRAW
        # =========================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{best_person}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Err: {best_error:.0f}",
            (x, y+h+30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "PCA Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()