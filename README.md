# Pet Robot Face Recognition: Per-Person PCA Subspace System

A real-time, vision-based facial recognition pipeline built for a pet robot to identify its owner and designated family members while reliably detecting and rejecting unknown individuals (strangers). 

Instead of a single global classifier, this system implements a **Per-Person PCA Architecture**, training separate mathematical eigenspaces for each individual and performing classification based on **Euclidean Reconstruction Error Optimization**.

---

## 🚀 Key Features
* **Face Preprocessing Pipeline:** Integrated Haar Cascade face detection, automatic grayscale conversion, tight bounding-box cropping, and Histogram Equalization for illumination consistency.
* **Per-Person Eigenspaces:** Independent PCA models trained for each class, generating personalized Mean Faces and unique sets of Eigenfaces (Principal Components).
* **Reconstruction-Error Matching:** Outlier and stranger detection bypassing traditional classifiers via $L_2$ norm reconstruction validation.
* **Real-time Live Inference:** Low-latency webcam stream tracking with color-coded bounding boxes and on-screen statistical distance scores.
* **Evaluation Metrics:** Built-in reporting module tracking per-person accuracy breakdown and total test-set performance metrics.

---

## 📐 Architecture & Mathematical Overview

The system operates across a dual-module framework: training specialized models and executing live subspace inference.

### 1. Vector Representation & Feature Mapping
Each detected face is tightly cropped, normalized, and resized to a standardized $100 \times 100$ spatial matrix before being flattened into a high-dimensional image vector:
$$x \in \mathbb{R}^{10000}$$

### 2. Subspace Projection (Compression)
For each family member, an independent PCA model is trained. The raw input vector $x$ is mean-centered using that specific individual's private Mean Face ($\mu$) and projected onto their localized Eigenface matrix ($W^T$), condensing the 10,000 pixels into an optimized feature coordinate $z$:
$$z = W^T(x - \mu)$$

### 3. Reconstruction & Outlier Rejection ($L_2$ Norm)
During runtime, the live input vector $x$ is projected and decompressed back into the original image space across **every** saved model using its inverse transform:
$$\hat{x} = Wz + \mu$$

The system evaluates similarity by computing the **Euclidean Reconstruction Error** between the original face ($x$) and the reconstructed matrix ($\hat{x}$):
$$\text{Error} = \|x - \hat{x}\|_2 = \sqrt{\sum_{i=1}^{n} (x_i - \hat{x}_i)^2}$$

If a stranger stands in front of the camera, their features cannot be accurately synthesized by a known family member's specialized eigenfaces. If the minimum error across all available models exceeds a calibrated threshold, the profile is securely flagged as **Unknown**.

---

## 📁 Repository Directory Structure

```text
├── dataset/
│   ├── train/
│   │   ├── Abdr/
│   │   ├── Noor/
│   │   └── Zai/
│   └── test/
│       ├── Abdr/
│       ├── Noor/
│       └── Zai/
├── models/
│   └── person_models.pkl      # Serialized custom PCA profiles dictionary
├── output/
│   ├── mean_*.png             # Saved per-person Mean Face plots
│   └── eigenfaces_*.png       # Visualized Principal Component grids
├── haarcascade_frontalface_default.xml
├── train.py                   # Model compilation & test set verification
├── recognize.py               # Real-time live camera tracking engine
└── requirements.txt           # Python dependency manifest
```
## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/roboproj.git](https://github.com/YourUsername/roboproj.git)
cd roboproj
```
### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
### 3. Setup Your Dataset
Place your training and testing images inside the dataset/ directory following the structure shown above. For optimal individual profile generation, ensure images feature varied angles, expressions, and clear centering.

### 💻 Running the Project
### Phase 1: Train the Models
Compile the independent eigenspaces, plot the diagnostic mathematical faces, and review the validation report by running:
```bash
python train.py
```
### Phase 2: Live Webcam Inference
Launch the real-time tracking engine for your robot:

```Bash
python recognize.py
```
Press Q to close the video window safely and release your webcam.

### ⚙️ Calibration & Parameter Tuning
* N_COMPONENTS = 20 (in train.py): Dictates the number of top principal components saved. Increase this if unique features are blending; decrease it if the model is capturing too much background variance.

* THRESHOLD = 6000 (in recognize.py): The absolute baseline boundary filtering strangers.

* If known family members are constantly misclassified as "Unknown", increase this value.

* If strangers are triggering family labels (Zai, Noor, Abdr), decrease this value until the error gap isolates them.
