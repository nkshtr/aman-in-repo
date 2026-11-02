"""
app.py
------
Main Flask web application for Smart Road Damage Detection.

Provides routes for:
  - Home page with upload form and quick test samples
  - Upload handling with strict validation (JPG/PNG only, 5 MB limit)
  - Result visualization with side-by-side comparison & severity table
  - History tracking backed by SQLite
  - Download of annotated images

Designed to be beginner-friendly with plain-English comments for school project presentation.
"""

import os
import time
import shutil
from werkzeug.utils import secure_filename
from PIL import Image
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    abort
)

import detector
import severity
import database


# Initialize Flask application
app = Flask(__name__)
app.secret_key = "road-damage-detection-secret-key-for-school-project"

# Configuration constants
UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULTS_FOLDER = os.path.join("static", "results")
SAMPLES_FOLDER = "samples"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 5
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024  # 5 MB max upload

# Ensure necessary storage directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

# Initialize the SQLite database on application start
database.init_db()

# Load YOLOv8 model ONCE when the application starts (avoids reloading on every request)
MODEL_PATH = os.path.join("models", "road_damage_yolov8n.pt")
print("[App] Initializing YOLOv8 Road Damage Detection model...")
model = detector.load_model(MODEL_PATH)
print("[App] Model ready for inference.")


def allowed_file(filename):
    """
    Checks if a given filename has a permitted image extension (.jpg, .jpeg, .png).
    
    Parameters:
        filename (str): Name of the uploaded file.
        
    Returns:
        bool: True if extension is permitted, False otherwise.
    """
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def is_valid_image_file(file_path):
    """
    Verifies that the uploaded file is a genuine, uncorrupted image using Pillow.
    
    Parameters:
        file_path (str): Path to the saved file on disk.
        
    Returns:
        bool: True if readable as an image, False if corrupt or invalid.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Pillow verification of image header integrity
        return True
    except Exception:
        return False


def get_available_samples():
    """
    Scans the samples/ directory and returns metadata for any test images.
    
    Returns:
        list of dict: Sample images with filename, friendly title, and description.
    """
    samples_info = {
        "sample_clean_road.jpg": {
            "title": "Clean Highway (Good)",
            "desc": "Paved asphalt road surface with no visible defects."
        },
        "sample_pothole.jpg": {
            "title": "Road with Pothole",
            "desc": "Asphalt road section showing a clear pothole cavity."
        },
        "sample_multiple_cracks.jpg": {
            "title": "Severe Cracking",
            "desc": "Longitudinal and alligator cracks running across road surface."
        },
        "sample_longitudinal_crack.jpg": {
            "title": "Longitudinal Crack",
            "desc": "Linear longitudinal crack along traffic wheel path."
        }
    }
    
    available = []
    if os.path.exists(SAMPLES_FOLDER):
        for fname in sorted(os.listdir(SAMPLES_FOLDER)):
            if allowed_file(fname):
                info = samples_info.get(fname, {
                    "title": fname.replace("_", " ").rsplit(".", 1)[0].title(),
                    "desc": "Sample test road image."
                })
                available.append({
                    "filename": fname,
                    "title": info["title"],
                    "desc": info["desc"]
                })
    return available


@app.route("/")
def index():
    """
    Renders the Home Page.
    
    Displays the file upload interface, short instructions,
    and a gallery of built-in sample road photos for quick testing.
    """
    samples = get_available_samples()
    return render_template("index.html", samples=samples)


@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Handles user image uploads.
    
    Validates the file, runs the YOLOv8 road damage detection model,
    calculates severity and road condition, stores the record in SQLite,
    and renders the result page.
    """
    # Check if a file was included in the POST request
    if "image" not in request.files:
        flash("Please select an image file to upload.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]

    # Check if user submitted form without picking a file
    if file.filename == "":
        flash("No file was selected. Please choose a JPG or PNG road image.", "error")
        return redirect(url_for("index"))

    # Check file extension
    if not allowed_file(file.filename):
        flash("Invalid file format! Only JPG, JPEG, and PNG images are supported.", "error")
        return redirect(url_for("index"))

    # Generate safe unique filenames using a timestamp
    clean_name = secure_filename(file.filename)
    timestamp_prefix = str(int(time.time() * 1000))
    upload_filename = f"upload_{timestamp_prefix}_{clean_name}"
    annotated_filename = f"annotated_{timestamp_prefix}_{clean_name}"

    upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
    annotated_path = os.path.join(RESULTS_FOLDER, annotated_filename)

    # Save uploaded file to disk
    file.save(upload_path)

    # Verify that the saved file is a valid, uncorrupted image
    if not is_valid_image_file(upload_path):
        # Remove corrupted file
        if os.path.exists(upload_path):
            os.remove(upload_path)
        flash("The uploaded file appears to be corrupted or not a valid image. Please try another file.", "error")
        return redirect(url_for("index"))

    # Read optional location input
    location = request.form.get("location", "").strip()
    if not location:
        location = "Not specified"

    # Read confidence threshold (default: 0.20 / 20%)
    try:
        conf_val = float(request.form.get("confidence", 0.20))
        # Keep within sensible bounds [0.10, 0.90]
        conf_threshold = max(0.10, min(0.90, conf_val))
    except (ValueError, TypeError):
        conf_threshold = 0.20

    # Run YOLOv8 detection
    try:
        detections, img_width, img_height = detector.detect_damage(
            model=model,
            image_path=upload_path,
            output_annotated_path=annotated_path,
            conf_threshold=conf_threshold
        )
    except Exception as e:
        flash(f"An error occurred while analyzing the image: {str(e)}", "error")
        return redirect(url_for("index"))

    # Calculate overall road condition and repair recommendation
    condition, badge_class, recommendation = severity.get_overall_road_condition(detections)

    # Save record to SQLite history
    inspection_id = database.add_inspection(
        location=location,
        damage_count=len(detections),
        condition=condition,
        recommendation=recommendation,
        original_image=upload_filename,
        annotated_image=annotated_filename
    )

    # Render result page
    return render_template(
        "result.html",
        original_image=upload_filename,
        annotated_image=annotated_filename,
        detections=detections,
        damage_count=len(detections),
        condition=condition,
        badge_class=badge_class,
        recommendation=recommendation,
        location=location,
        inspection_id=inspection_id,
        img_width=img_width,
        img_height=img_height,
        confidence_used=int(conf_threshold * 100)
    )


@app.route("/test-sample/<filename>")
def test_sample(filename):
    """
    Runs quick detection on a selected sample image without requiring manual re-upload.
    
    Enables one-click instant testing during project demonstrations.
    """
    clean_name = secure_filename(filename)
    sample_source_path = os.path.join(SAMPLES_FOLDER, clean_name)

    if not os.path.exists(sample_source_path):
        flash("Sample image not found.", "error")
        return redirect(url_for("index"))

    # Copy sample to uploads directory with a unique timestamp
    timestamp_prefix = str(int(time.time() * 1000))
    upload_filename = f"sample_{timestamp_prefix}_{clean_name}"
    annotated_filename = f"annotated_{timestamp_prefix}_{clean_name}"

    upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
    annotated_path = os.path.join(RESULTS_FOLDER, annotated_filename)

    shutil.copy(sample_source_path, upload_path)

    # Run detection with default 0.20 confidence
    detections, img_width, img_height = detector.detect_damage(
        model=model,
        image_path=upload_path,
        output_annotated_path=annotated_path,
        conf_threshold=0.20
    )

    # Calculate road condition
    condition, badge_class, recommendation = severity.get_overall_road_condition(detections)

    # Friendly location label for sample
    sample_title = clean_name.replace("sample_", "").replace(".jpg", "").replace(".png", "").replace("_", " ").title()
    location = f"Sample Demo ({sample_title})"

    # Save to SQLite database
    inspection_id = database.add_inspection(
        location=location,
        damage_count=len(detections),
        condition=condition,
        recommendation=recommendation,
        original_image=upload_filename,
        annotated_image=annotated_filename
    )

    return render_template(
        "result.html",
        original_image=upload_filename,
        annotated_image=annotated_filename,
        detections=detections,
        damage_count=len(detections),
        condition=condition,
        badge_class=badge_class,
        recommendation=recommendation,
        location=location,
        inspection_id=inspection_id,
        img_width=img_width,
        img_height=img_height,
        confidence_used=20
    )


@app.route("/samples/<filename>")
def serve_sample(filename):
    """
    Serves sample road images for thumbnail display on the home page.
    """
    clean_name = secure_filename(filename)
    return send_from_directory(SAMPLES_FOLDER, clean_name)


@app.route("/download/<filename>")
def download_result(filename):
    """
    Allows the user to download the annotated result image to their computer.
    """
    clean_name = secure_filename(filename)
    file_path = os.path.join(RESULTS_FOLDER, clean_name)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(RESULTS_FOLDER, clean_name, as_attachment=True)


@app.route("/history")
def history():
    """
    Renders the inspection history page listing all past road analyses.
    """
    records = database.get_all_inspections()
    return render_template("history.html", records=records)


@app.route("/history/<int:inspection_id>")
def view_history_item(inspection_id):
    """
    Re-displays the full result page for a specific past inspection.
    """
    record = database.get_inspection_by_id(inspection_id)
    if not record:
        flash("Inspection record not found.", "error")
        return redirect(url_for("history"))

    # Derive badge class
    cond = record["condition"]
    badge_class = f"condition-{cond.lower()}"

    # Re-run lightweight metadata extraction or display summary
    return render_template(
        "result.html",
        original_image=record["original_image"],
        annotated_image=record["annotated_image"],
        detections=[],  # Historical summary view
        damage_count=record["damage_count"],
        condition=record["condition"],
        badge_class=badge_class,
        recommendation=record["recommendation"],
        location=record["location"],
        inspection_id=record["id"],
        img_width=None,
        img_height=None,
        is_historical=True,
        timestamp=record["timestamp"]
    )


@app.route("/history/clear", methods=["POST"])
def clear_history_route():
    """
    Clears all inspection history records from the SQLite database.
    """
    database.clear_history()
    flash("Inspection history has been cleared.", "info")
    return redirect(url_for("history"))


@app.errorhandler(413)
def file_too_large(e):
    """
    Friendly error handler when an uploaded file exceeds the 5 MB limit.
    """
    flash(f"File too large! The maximum allowed image size is {MAX_FILE_SIZE_MB} MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    """
    Friendly 404 error handler.
    """
    return render_template("404.html"), 404


if __name__ == "__main__":
    # Runs the Flask server on local development port 5000
    print("\n=======================================================")
    print(" Smart Road Damage Detection Web App")
    print(" Running locally at: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
