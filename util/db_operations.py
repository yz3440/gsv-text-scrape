from .model import StreetViewProcessResult
import sqlite3
from typing import List
from enum import Enum


def setup_database(db_path: str = "gsv.db"):
    """Setup the database with required tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create OCR result table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ocr_result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pano_id TEXT,
            text TEXT,
            confidence REAL,
            yaw REAL,
            pitch REAL,
            width REAL,
            height REAL,
            engine TEXT,
            FOREIGN KEY (pano_id) REFERENCES search_panoramas(pano_id)
        )
    """
    )

    # Add computed_ocr column to search_panoramas if it doesn't exist
    cursor.execute("PRAGMA table_info(search_panoramas)")
    columns = [column[1] for column in cursor.fetchall()]
    if "computed_ocr" not in columns:
        cursor.execute(
            "ALTER TABLE search_panoramas ADD COLUMN computed_ocr BOOLEAN DEFAULT FALSE"
        )

    # Add download_attempted column if it doesn't exist
    if "download_attempted" not in columns:
        cursor.execute(
            "ALTER TABLE search_panoramas ADD COLUMN download_attempted INTEGER DEFAULT 0"
        )

    conn.commit()
    conn.close()


def get_n_pano_id_without_ocr(
    connection,
    n: int,
) -> List[str]:
    # get n random panorama_id that has computed_ocr = False
    cur = connection.cursor()
    cur.execute(
        "SELECT pano_id FROM search_panoramas WHERE (computed_ocr = 0 or computed_ocr is NULL) ORDER BY RANDOM() LIMIT ?",
        (n,),
    )
    res = cur.fetchall()
    return [r[0] for r in res]


def add_one_to_download_count(panorama_id: str, connection):
    cur = connection.cursor()
    cur.execute(
        "UPDATE search_panoramas SET download_attempted = download_attempted + 1 WHERE pano_id = ?",
        (panorama_id,),
    )
    connection.commit()


def insert_ocr_result(
    connection,
    streetview_process_result: StreetViewProcessResult,
):
    # First, check if computed_ocr is already True
    cur = connection.cursor()
    cur.execute(
        "SELECT computed_ocr FROM search_panoramas WHERE pano_id = ?",
        (streetview_process_result.panorama_id,),
    )
    result = cur.fetchone()

    if result is None:
        print(
            f"No record found for panorama_id: {streetview_process_result.panorama_id}"
        )
        return

    if result[0]:  # If computed_ocr is already True
        print(
            f"OCR already computed for panorama_id: {streetview_process_result.panorama_id}"
        )
        return

    try:
        # Start a new transaction
        cur = connection.cursor()

        # INSERT OCR RESULTS
        for sphere_ocr_result in streetview_process_result.all_sphere_ocr_results:
            cur.execute(
                "INSERT INTO ocr_result (pano_id, text, confidence, yaw, pitch, width, height, engine) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    streetview_process_result.panorama_id,
                    sphere_ocr_result.text,
                    sphere_ocr_result.confidence,
                    sphere_ocr_result.yaw,
                    sphere_ocr_result.pitch,
                    sphere_ocr_result.width,
                    sphere_ocr_result.height,
                    sphere_ocr_result.engine,
                ),
            )

        # SET computed_ocr = True
        cur.execute(
            "UPDATE search_panoramas SET computed_ocr = 1 WHERE pano_id = ?",
            (streetview_process_result.panorama_id,),
        )

        connection.commit()

        print(
            f"Transaction committed successfully for panorama_id: {streetview_process_result.panorama_id}"
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        connection.rollback()
        raise
