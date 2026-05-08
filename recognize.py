
import cv2
import numpy as np
import pickle
import os

IMG_SIZE = 100
# TUNE THIS: Based on the distance printed in your console
THRESHOLD = 65
# Load Models
with open("models/pca_model.pkl", "rb") as f: pca = pickle.load(f)
with open("models/knn_model.pkl", "rb") as f: knn = pickle.load(f)
with open("models/labels.pkl", "rb") as f: label_map = pickle.load(f)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

print("Starting Robot Vision... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray_equalized, 1.2, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # 1. Preprocess (MUST match train.py)
        face_roi = gray_equalized[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_resized = cv2.equalizeHist(face_resized)
        
        # Z-Score Normalization
        face_norm = face_resized.astype(np.float32)
        face_norm = (face_norm - np.mean(face_norm)) / (np.std(face_norm) + 1e-5)
        
        face_vector = face_norm.flatten().reshape(1, -1)

        # 2. Project and Calculate Distance
        face_pca = pca.transform(face_vector)
        dist, ind = knn.kneighbors(face_pca, n_neighbors=1)
        distance_val = dist[0][0]

        # 3. Decision Logic (This is where person_name is defined)
        if distance_val < THRESHOLD:
            prediction = knn.predict(face_pca)
            person_name = label_map[prediction[0]]
            color = (0, 255, 0) # Green for Family
        else:
            person_name = "Unknown"
            color = (0, 0, 255) # Red for Unknown

        # 4. DEBUG: Now person_name exists, so we can print it safely
        print(f"DEBUG -> Name: {person_name} | Distance: {int(distance_val)}")
        
        # 5. UI Feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{person_name} ({int(distance_val)})", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Robot Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()