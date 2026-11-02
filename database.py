"""
database.py
-----------
Handles SQLite storage for past road damage inspections.

Uses Python's built-in sqlite3 module with zero external database dependencies.
Designed to be beginner-friendly with plain-English comments for school project presentation.
"""

import sqlite3
import datetime
import os


DB_FILE = "road_damage.db"


def get_connection(db_path=DB_FILE):
    """
    Creates and returns a connection to the SQLite database.
    
    Parameters:
        db_path (str): File path to the SQLite database file.
        
    Returns:
        sqlite3.Connection: Database connection configured with Row factory.
    """
    conn = sqlite3.connect(db_path)
    # Enable dict-like access to row columns by name
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_FILE):
    """
    Initializes the database table if it doesn't already exist.
    
    Creates table:
      inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        location TEXT,
        damage_count INTEGER,
        condition TEXT,
        recommendation TEXT,
        original_image TEXT,
        annotated_image TEXT
      )
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location TEXT,
            damage_count INTEGER NOT NULL,
            condition TEXT NOT NULL,
            recommendation TEXT,
            original_image TEXT NOT NULL,
            annotated_image TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_inspection(location, damage_count, condition, recommendation, original_image, annotated_image, db_path=DB_FILE):
    """
    Saves a new inspection analysis record into the database.
    
    Parameters:
        location (str): Name of the street or landmark (defaults to 'Not specified').
        damage_count (int): Number of detected damage objects.
        condition (str): Overall road condition ('Good', 'Fair', 'Poor', 'Critical').
        recommendation (str): 1-2 sentence repair recommendation.
        original_image (str): Filename of the saved original image.
        annotated_image (str): Filename of the saved annotated image.
        db_path (str): Database file path.
        
    Returns:
        int: The auto-generated ID of the newly inserted inspection record.
    """
    # Current date and time formatted nicely: YYYY-MM-DD HH:MM
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    clean_location = location.strip() if location and location.strip() else "Not specified"
    
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inspections (
            timestamp, location, damage_count, condition, recommendation, original_image, annotated_image
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        clean_location,
        int(damage_count),
        condition,
        recommendation,
        original_image,
        annotated_image
    ))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_all_inspections(db_path=DB_FILE):
    """
    Retrieves all past inspection records, newest first.
    
    Parameters:
        db_path (str): Database file path.
        
    Returns:
        list of dict: List of past analysis records.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, location, damage_count, condition, recommendation, original_image, annotated_image
        FROM inspections
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    # Convert sqlite3.Row objects into standard Python dictionaries
    records = [dict(row) for row in rows]
    conn.close()
    return records


def get_inspection_by_id(inspection_id, db_path=DB_FILE):
    """
    Retrieves a single inspection record by its ID.
    
    Parameters:
        inspection_id (int): ID of the inspection record.
        db_path (str): Database file path.
        
    Returns:
        dict or None: The inspection record if found, otherwise None.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, location, damage_count, condition, recommendation, original_image, annotated_image
        FROM inspections
        WHERE id = ?
    """, (inspection_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def clear_history(db_path=DB_FILE):
    """
    Deletes all inspection records from the database.
    
    Useful during testing or when resetting the project demo.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM inspections")
    conn.commit()
    conn.close()
