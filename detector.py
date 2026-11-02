"""
detector.py
-----------
Handles YOLOv8 model loading, inference, and visualization of bounding boxes.

Designed to be beginner-friendly with plain-English comments for school project presentation.
"""

import os
import cv2
from ultralytics import YOLO
from severity import calculate_box_area_percentage, get_damage_severity


# Define friendly display names for the model's dataset classes
CLASS_NAME_MAP = {
    "alligator crack": "Alligator Crack",
    "transverse crack": "Transverse Crack",
    "longitudinal crack": "Longitudinal Crack",
    "other corruption": "Surface Degradation",
    "pothole": "Pothole",
    "Pothole": "Pothole"
}

# Define distinct BGR colors for each damage category to make bounding boxes easy to distinguish
CATEGORY_COLORS = {
    "Pothole": (34, 34, 220),            # Red
    "Alligator Crack": (0, 140, 255),     # Orange
    "Longitudinal Crack": (0, 215, 255),  # Yellow
    "Transverse Crack": (180, 50, 200),   # Purple/Magenta
    "Surface Degradation": (210, 160, 40) # Slate/Teal
}

# Severity badge background colors (BGR)
SEVERITY_COLORS = {
    "Low": (34, 197, 94),       # Green
    "Medium": (234, 179, 8),    # Yellow/Amber
    "High": (239, 68, 68)       # Red
}


def load_model(model_path="models/road_damage_yolov8n.pt"):
    """
    Loads the YOLOv8 model from the disk into memory.
    
    If the model weight file does not exist locally, this function automatically
    downloads it from Hugging Face (pretrained on RDD2022).
    
    This function is called once when the Flask server starts,
    so that we don't waste time reloading weights on every user upload.
    
    Parameters:
        model_path (str): File path to the trained .pt weight file.
        
    Returns:
        YOLO: Initialized ultralytics YOLO model object.
    """
    if not os.path.exists(model_path):
        print(f"[Detector] Model file not found at '{model_path}'.")
        print("[Detector] Downloading pretrained YOLOv8 Road Damage model from Hugging Face (~6 MB)...")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        import urllib.request
        model_url = "https://huggingface.co/ozair23/yolov8-road-damage-detector/resolve/main/best.pt"
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[Detector] Pretrained weights downloaded successfully to '{model_path}'!")

    print(f"[Detector] Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    print(f"[Detector] Model loaded successfully with classes: {model.names}")
    return model


def format_class_name(raw_name):
    """
    Cleans up raw class names from the dataset into human-friendly strings.
    
    Parameters:
        raw_name (str): The raw label from model.names (e.g. 'alligator crack').
        
    Returns:
        str: Friendly capitalized label (e.g. 'Alligator Crack').
    """
    clean_key = str(raw_name).strip()
    return CLASS_NAME_MAP.get(clean_key, clean_key.capitalize())


def detect_damage(model, image_path, output_annotated_path, conf_threshold=0.20):
    """
    Runs YOLOv8 road damage detection on an input image and generates an annotated output image.
    
    Steps performed:
      1. Reads the image from disk to get dimensions.
      2. Passes the image through YOLOv8 using model.predict().
      3. For each detected bounding box:
         - Retrieves class label and confidence score.
         - Computes relative bounding box area (%).
         - Evaluates damage severity (Low, Medium, High).
      4. Draws high-contrast bounding boxes and label badges onto the image.
      5. Saves the final annotated image to output_annotated_path.
      
    Parameters:
        model (YOLO): The loaded YOLOv8 model.
        image_path (str): Path to the user's uploaded image.
        output_annotated_path (str): Destination path to save the annotated image.
        conf_threshold (float): Minimum confidence threshold (default 0.20 / 20%).
        
    Returns:
        tuple: (detections, image_width, image_height)
               where detections is a list of dictionary records.
    """
    # Step 1: Read the image using OpenCV to get dimensions and prepare for drawing
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file. It may be corrupted or in an unsupported format.")
        
    image_height, image_width = img.shape[:2]

    # Step 2: Run inference with YOLOv8
    # conf sets the minimum confidence cutoff; verbose=False silences console clutter
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)
    first_result = results[0]

    detections = []

    # Step 3: Iterate through detected bounding boxes
    if first_result.boxes is not None and len(first_result.boxes) > 0:
        for box in first_result.boxes:
            # Class ID and friendly name
            cls_id = int(box.cls[0])
            raw_class_name = model.names.get(cls_id, f"Class {cls_id}")
            damage_type = format_class_name(raw_class_name)

            # Confidence percentage
            confidence = round(float(box.conf[0]) * 100.0, 1)

            # Bounding box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
            
            # Boundary check to stay within image edges
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))
            bbox = [x1, y1, x2, y2]

            # Calculate area covered relative to the whole image
            area_percentage = calculate_box_area_percentage(bbox, image_width, image_height)

            # Calculate severity level (Low, Medium, High)
            severity = get_damage_severity(damage_type, area_percentage)

            # Store detection metadata
            detections.append({
                "damage_type": damage_type,
                "confidence": confidence,
                "area_percentage": area_percentage,
                "severity": severity,
                "bbox": bbox
            })

            # Step 4: Draw bounding box and label banner onto the image
            color = CATEGORY_COLORS.get(damage_type, (0, 200, 255))
            
            # Box thickness scales slightly with image size so it looks good on both small and 4K images
            thickness = max(2, int(min(image_width, image_height) / 350))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Prepare badge text: e.g. "Pothole 78% [Medium]"
            label_text = f"{damage_type} {confidence:.0f}% [{severity}]"
            
            # Compute font scale based on resolution
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(image_width, image_height) / 1200.0)
            text_thickness = max(1, int(font_scale * 2))
            
            text_size, baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            text_w, text_h = text_size

            # Position of the label box (place above box if space permits, else inside)
            label_y1 = max(0, y1 - text_h - 10)
            label_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 10
            label_x2 = min(image_width, x1 + text_w + 10)

            # Draw filled rectangle for label background
            cv2.rectangle(img, (x1, label_y1), (label_x2, label_y2), color, -1)
            
            # Draw white text over the colored background
            text_pos_y = label_y2 - 5 if y1 - text_h - 10 >= 0 else label_y2 - 5
            cv2.putText(
                img,
                label_text,
                (x1 + 5, text_pos_y),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                lineType=cv2.LINE_AA
            )

    # Step 5: Save the annotated image to output directory
    os.makedirs(os.path.dirname(output_annotated_path), exist_ok=True)
    cv2.imwrite(output_annotated_path, img)

    return detections, image_width, image_height
