"""
severity.py
-----------
Contains the business logic for calculating damage severity and determining
the overall road condition based on YOLO detection results.

Designed to be beginner-friendly with plain-English comments for school project presentation.
"""


def calculate_box_area_percentage(bbox, image_width, image_height):
    """
    Calculates what percentage of the full image a bounding box covers.
    
    Parameters:
        bbox (list or tuple): [x1, y1, x2, y2] pixel coordinates.
        image_width (int): Total width of the image in pixels.
        image_height (int): Total height of the image in pixels.
        
    Returns:
        float: Percentage of the image covered (0.0 to 100.0).
    """
    x1, y1, x2, y2 = bbox
    box_width = max(0.0, float(x2 - x1))
    box_height = max(0.0, float(y2 - y1))
    box_area = box_width * box_height
    
    total_image_area = float(image_width * image_height)
    if total_image_area <= 0:
        return 0.0
        
    percentage = (box_area / total_image_area) * 100.0
    return round(percentage, 2)


def get_damage_severity(damage_type, area_percentage):
    """
    Determines whether a single damage detection is Low, Medium, or High severity.
    
    Severity Rules:
      1. Base severity from relative size:
         - Low: covers under 5% of the total image area.
         - Medium: covers between 5% and 15% of the image area.
         - High: covers more than 15% of the image area.
         
      2. Pothole escalation rule:
         - Potholes present a sudden, severe impact hazard to vehicle tires and suspensions.
         - Therefore, potholes are treated as one level more serious than cracks of the same size:
           * Low size pothole (< 5%) -> Escalated to Medium severity.
           * Medium size pothole (5% - 15%) -> Escalated to High severity.
           * High size pothole (> 15%) -> High severity.
           
    Parameters:
        damage_type (str): Name of the detected damage (e.g., 'Pothole', 'longitudinal crack').
        area_percentage (float): Percentage of the image covered by the bounding box.
        
    Returns:
        str: 'Low', 'Medium', or 'High'
    """
    # Step 1: Calculate the base severity based on size percentage
    if area_percentage < 5.0:
        base_severity = "Low"
    elif area_percentage <= 15.0:
        base_severity = "Medium"
    else:
        base_severity = "High"

    # Step 2: Check if damage is a pothole
    is_pothole = "pothole" in damage_type.lower()
    
    # Step 3: Escalate severity if it is a pothole
    if is_pothole:
        if base_severity == "Low":
            return "Medium"
        elif base_severity == "Medium":
            return "High"
        else:
            return "High"

    return base_severity


def get_overall_road_condition(detections):
    """
    Evaluates all detected damages to determine the overall condition of the road.
    
    Road Condition Rules:
      - Good: 0 damages detected.
      - Fair: All detected damages are of 'Low' severity.
      - Poor: At least one damage is of 'Medium' severity (and no High severity).
      - Critical: At least one damage is 'High' severity, OR there are 5 or more potholes.
      
    Parameters:
        detections (list): List of detection dictionaries. Each dictionary contains:
                           'damage_type', 'confidence', 'bbox', 'area_percentage', 'severity'.
                           
    Returns:
        tuple: (condition_name, badge_color_code, recommendation_text)
               e.g. ('Good', 'good', 'Road is in great shape...')
    """
    total_count = len(detections)
    
    # Rule 1: No damages detected -> Road is Good
    if total_count == 0:
        condition = "Good"
        badge_class = "condition-good"
        recommendation = "The road surface appears to be in good condition with no visible structural defects. Continue routine periodic monitoring."
        return condition, badge_class, recommendation

    # Count potholes and severity levels
    pothole_count = 0
    has_high = False
    has_medium = False

    for item in detections:
        sev = item.get("severity", "Low")
        dmg_type = item.get("damage_type", "").lower()
        
        if "pothole" in dmg_type:
            pothole_count += 1
            
        if sev == "High":
            has_high = True
        elif sev == "Medium":
            has_medium = True

    # Rule 2: Critical condition
    # Triggered if any damage is High severity OR there are 5+ potholes
    if has_high or pothole_count >= 5:
        condition = "Critical"
        badge_class = "condition-critical"
        if pothole_count >= 5:
            recommendation = "Severe safety risk! High density of potholes detected. Immediate emergency road closure or prioritized patching is strongly advised."
        else:
            recommendation = "High severity road damage detected! Urgent structural repair is required to prevent vehicle damage and traffic accidents."
        return condition, badge_class, recommendation

    # Rule 3: Poor condition
    # Triggered if at least one damage is Medium severity
    if has_medium:
        condition = "Poor"
        badge_class = "condition-poor"
        recommendation = "Moderate road deterioration detected. Schedule maintenance and localized resurfacing within the next few weeks before defects worsen."
        return condition, badge_class, recommendation

    # Rule 4: Fair condition
    # All detected damages are Low severity
    condition = "Fair"
    badge_class = "condition-fair"
    recommendation = "Minor superficial cracking observed. Preventive sealant application is recommended to stop moisture infiltration and freeze-thaw damage."
    return condition, badge_class, recommendation
