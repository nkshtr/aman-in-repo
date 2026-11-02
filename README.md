# 🛣️ Smart Road Damage & Pothole Detection Web App

A beginner-friendly, AI-powered web application built using **Python 3**, **Flask**, and **YOLOv8** (`ultralytics`). It automatically detects road surface hazards (potholes, longitudinal cracks, transverse cracks, alligator cracks) from uploaded photos, calculates their severity, grades overall road condition, and provides actionable maintenance recommendations.

Built as an engineering and school project with clean, simple, and well-commented code.

---

## 📸 Key Features

- **One-Click AI Detection:** Upload any road photo (JPG, JPEG, PNG) to detect cracks and potholes.
- **Side-by-Side Visual Inspection:** Compare the raw original image side-by-side with the YOLOv8 annotated result.
- **Automatic Severity Grading:** Calculates the percentage of road surface covered by each defect and assesses severity:
  - **Low:** Defect covers under 5% of the image.
  - **Medium:** Defect covers 5% to 15% of the image.
  - **High:** Defect covers over 15% of the image.
  - **Pothole Escalation:** Potholes are bumped by +1 severity tier because of their sudden collision and tire-damage hazard.
- **Overall Road Condition:**
  - 🟢 **Good:** 0 damages detected ("Road looks fine!").
  - 🟡 **Fair:** Only Low-severity minor defects found.
  - 🟠 **Poor:** At least one Medium-severity defect found.
  - 🔴 **Critical:** At least one High-severity defect OR 5+ potholes detected.
- **Repair Recommendations:** Generates clear 1–2 sentence maintenance instructions tailored to the road condition.
- **Optional Location Tagging:** Enter street names or landmarks to track specific road segments.
- **Built-in Sample Gallery:** Test the app immediately using 4 pre-loaded road photos without uploading your own.
- **Inspection History (SQLite):** Automatically records every inspection (timestamp, location, damage count, condition, thumbnails) in a local lightweight SQLite database (`road_damage.db`).
- **Download Annotated Results:** Export high-resolution annotated images with bounding boxes and confidence badges.
- **Friendly Validation & Error Handling:** Clear feedback for missing files, invalid formats, corrupt files, or uploads exceeding 5 MB.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+ / Flask 3.x
- **Computer Vision & AI:** YOLOv8 Nano (`ultralytics`)
- **Image Processing:** OpenCV (`opencv-python`) & Pillow (`PIL`)
- **Database:** SQLite (Python standard library `sqlite3`, zero external setup)
- **Frontend:** Semantic HTML5, Vanilla CSS3 (Flexbox & Grid), Vanilla JavaScript

---

## 📂 Project Structure

```text
├── app.py                  # Flask web routes, file upload validation, history tracking
├── detector.py             # YOLOv8 model loading, inference & bounding-box drawing
├── severity.py             # Severity calculation & road condition rules
├── database.py             # SQLite database helper for inspection history
├── train.py                # Optional training script for fine-tuning on custom datasets
├── requirements.txt        # Python dependency list
├── README.md               # Project documentation & presentation guide
├── models/
│   └── road_damage_yolov8n.pt  # Pretrained YOLOv8 Nano weights (~6.2 MB)
├── samples/                # Built-in test images for instant demonstration
│   ├── sample_clean_road.jpg
│   ├── sample_pothole.jpg
│   ├── sample_longitudinal_crack.jpg
│   └── sample_multiple_cracks.jpg
├── static/
│   ├── css/
│   │   └── style.css       # Clean, responsive CSS (cards, tables, color-coded badges)
│   ├── js/
│   │   └── main.js         # Client-side validation, drag-and-drop & loading spinner
│   ├── uploads/            # Temporary storage for uploaded user images
│   └── results/            # Saved AI-annotated output images
└── templates/
    ├── base.html           # Master layout with navigation and footer
    ├── index.html          # Upload form, instructions & sample images gallery
    ├── result.html         # Side-by-side view, condition badge & defect table
    ├── history.html        # Table of past inspections with thumbnails
    └── 404.html            # Friendly 404 error page
```

---

## 🚀 How to Install and Run Locally

### 1. Clone or Download the Project
Open a terminal in the project directory:
```bash
cd "d:\Jaishiv Friend"
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Or on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Your Browser
Navigate to:
```text
http://127.0.0.1:5000
```

> **Note on Model Weights:** The pretrained model file `models/road_damage_yolov8n.pt` (~6.2 MB) is automatically verified on startup. If missing, `detector.py` will automatically download it from Hugging Face on the first run.

---

## 🧠 Model Information & Origin

- **Model Architecture:** YOLOv8 Nano (`yolov8n`), 3.2 million parameters.
- **Source:** Hugging Face repository `ozair23/yolov8-road-damage-detector`.
- **Training Dataset:** Pretrained on the international **RDD2022 (Road Damage Dataset 2022)**, an authoritative benchmark dataset consisting of thousands of windshield camera road images collected across Japan, India, the Czech Republic, Norway, and the United States.
- **Classes Recognized:**
  1. `Pothole` — Structural cavities in the asphalt.
  2. `longitudinal crack` — Cracks running parallel to the direction of traffic (wheel paths).
  3. `transverse crack` — Cracks running perpendicular to the road centerline (thermal/shrinkage cracks).
  4. `alligator crack` — Interconnected web-like cracks caused by structural fatigue and subgrade failure.
  5. `other corruption` / `Surface Degradation` — Ravelling, rutting, or peeling of the pavement seal.

---

## 📐 How Detection and Severity Work

### 1. Single Forward Pass Detection
Unlike older multi-stage methods (R-CNN), YOLO (You Only Look Once) divides an image into a grid and predicts bounding box coordinates $(x_1, y_1, x_2, y_2)$ and class confidence probabilities in **a single forward pass**. This enables inference in ~30–60 milliseconds on CPU.

### 2. Relative Bounding Box Area
The physical severity of road damage is strongly correlated with its spatial footprint. We compute the relative area covered by each bounding box:
$$\text{Area Percentage} = \frac{(x_2 - x_1) \times (y_2 - y_1)}{\text{Image Width} \times \text{Image Height}} \times 100$$

### 3. Severity Classification
- **Low Severity:** Box occupies **under 5%** of the frame (minor hairline crack).
- **Medium Severity:** Box occupies **5% to 15%** of the frame (moderate fissure).
- **High Severity:** Box occupies **over 15%** of the frame (extensive structural failure).
- **Pothole Escalation Rule:** Because a pothole is a 3D structural hole rather than a 2D line, it poses an immediate hazard to vehicle tires and motorcycle balance. Any pothole is therefore automatically bumped **+1 severity grade** (Low $\rightarrow$ Medium; Medium $\rightarrow$ High).

### 4. Overall Road Condition
- **Good:** 0 damages detected.
- **Fair:** All detected defects are Low severity.
- **Poor:** At least one Medium severity defect.
- **Critical:** At least one High severity defect OR 5+ potholes.

---

## 🔬 Honest Model Evaluation on Real Samples

As required by good scientific evaluation, here is the honest test result on real sample images:

| Test Image | Expected Ground Truth | Actual YOLOv8 Output (conf=0.20) | Accuracy Assessment |
| :--- | :--- | :--- | :--- |
| **`sample_pothole.jpg`** | Pothole on asphalt | Pothole detected (29.5% conf, Low area escalated to **Medium severity**) | **Accurate.** Bounding box tightly encapsulates the pothole cavity. |
| **`sample_multiple_cracks.jpg`** | Multiple road fissures | 2 Longitudinal Cracks (36.7% and 27.4% conf) + Surface Degradation | **Accurate.** Detected the long linear cracks correctly; categorized as **High severity**. |
| **`sample_longitudinal_crack.jpg`** | Single continuous crack | Longitudinal Crack (25.4% conf, 23.3% area) | **Accurate.** Identified as **High severity** longitudinal crack. |
| **`sample_clean_road.jpg`** | Smooth highway | At 20% conf: faint false positive ("Surface Degradation" 24.7%). At 25% conf: **0 defects (Good)**. | **Partially Accurate.** Illustrates how road lane markings or asphalt color shifts can sometimes register low confidence false positives. Raising confidence to 25% correctly identifies it as **Good**. |

---

## ⚠️ Known Limitations & Viva Talking Points

1. **Camera Angle & Perspectives:** The model was trained on dashcam/windshield perspectives. Top-down bird's-eye drone photos or close-up macro shots from a phone might yield lower confidence.
2. **Shadows & Tree Branches:** Tree shadows across sunny roads can have linear edges that an AI model can sometimes mistake for longitudinal cracks.
3. **Wet Roads & Reflections:** Standing water after rain produces road reflections that can obscure crack boundaries or look like dark potholes.
4. **Lighting & Motion Blur:** Nighttime photos or images taken from fast-moving vehicles with motion blur decrease detection accuracy.

---

## 🎓 School Viva Questions & Answers

**Q1: Why did you use YOLOv8 instead of building a CNN from scratch?**
> *Answer:* Building a basic CNN from scratch would only classify whether an image contains damage or not (binary classification). YOLOv8 is an object detection model that does both localization (drawing exact bounding boxes) and multi-class classification simultaneously in real time.

**Q2: What is Transfer Learning?**
> *Answer:* Transfer learning is the process of taking a neural network that has already learned low-level visual features (edges, textures, gradients from millions of images in the COCO dataset) and fine-tuning its final detection layers on a specialized dataset like RDD2022 road damage.

**Q3: Why is model loading done at startup instead of inside the upload route?**
> *Answer:* Loading neural network weights into RAM takes 1–3 seconds. If we loaded it inside `upload_image()`, every user request would suffer an unnecessary delay. Loading once at startup makes subsequent predictions run in milliseconds.
