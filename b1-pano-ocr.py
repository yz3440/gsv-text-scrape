from util.streetview_ocr import (
    download_and_ocr_google_streetview_from_id,
    get_streetview_image,
    ocr_google_streetview_from_id,
)
from util.db_operations import (
    get_n_pano_id_without_ocr,
    insert_ocr_result,
    add_one_to_download_count,
    setup_database,
)
import panoocr as po
import json
import signal
from functools import wraps
import sqlite3
from typing import Any

from dotenv import load_dotenv
import os
import sys
import argparse
import uuid
import time


# Add timeout decorator for Unix systems
def timeout_handler(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds"
                )

            # Set the timeout handler
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator


@timeout_handler(5)
def get_streetview_image_with_timeout(pano_id):
    return get_streetview_image(pano_id)


parser = argparse.ArgumentParser(description="Run OCR process")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument(
    "--ocr-engine",
    choices=["macocr", "florence2", "paddleocr", "easyocr"],
    default="macocr",
    help="Choose OCR engine",
)
parser.add_argument("--save-result", action="store_true", help="Save OCR result")

args = parser.parse_args()

DEBUG_MODE = args.debug
OCR_ENGINE_NAME = args.ocr_engine
SAVE_RESULT = True if DEBUG_MODE else args.save_result

UNIQUE_ID = uuid.uuid4()
print(f"UUID / {UNIQUE_ID}")

TEMP_DIR = f"./temp/{UNIQUE_ID}"
os.makedirs(TEMP_DIR, exist_ok=True)

print(f"DEBUG_MODE: {DEBUG_MODE}")
print(f"OCR_ENGINE_NAME: {OCR_ENGINE_NAME}")
print(f"SAVE_RESULT: {SAVE_RESULT}")

load_dotenv()
DATABASE_PATH = os.getenv("DATABASE_PATH", "gsv.db")


def clear_console():
    if os.name == "nt":
        _ = os.system("cls")
    else:
        _ = os.system("clear")


def start_debug_process(co, ocr_engine, duplication_detection_engine, perspectives):
    pano_ids = [
        "UPnf7-KYannHcUI03oHZ5A",  # TIMES SQUARE
        "kg7vOGyQopfBnJWU1KSouA",  # CHINATOWN
    ]

    for pano_id in pano_ids:
        result = download_and_ocr_google_streetview_from_id(
            pano_id,
            perspectives,
            ocr_engine,
            duplication_detection_engine,
        )

        if SAVE_RESULT:
            result.save_to_dir("./temp")


def main(co: sqlite3.Connection):
    if OCR_ENGINE_NAME == "macocr":
        OCR_ENGINE = po.create_ocr_engine(
            po.OCREngineType.MACOCR,
            {
                "language_preference": [
                    po.MacOCRLanguageCode.ENGLISH_US,
                ],
                "recognition_level": po.MacOCRRecognitionLevel.ACCURATE,
            },
        )
    elif OCR_ENGINE_NAME == "florence2":
        OCR_ENGINE = po.create_ocr_engine(
            po.OCREngineType.FLORENCE,
            {},
        )
    elif OCR_ENGINE_NAME == "paddle":
        OCR_ENGINE = po.create_ocr_engine(
            engine_type=po.OCREngineType.PADDLEOCR,
            config={
                "language_preference": po.PaddleOCRLanguageCode.ENGLISH,
                "recognize_upside_down": False,
                "use_v4_server": True,
            },
        )
    elif OCR_ENGINE_NAME == "easyocr":
        OCR_ENGINE = po.create_ocr_engine(
            engine_type=po.OCREngineType.EASYOCR,
            config={},
        )
    else:
        raise ValueError("Invalid OCR engine")

    DUPLICATION_DETECTION_ENGINE = po.SphereOCRDuplicationDetectionEngine()
    PERSPECTIVES = po.DEFAULT_IMAGE_PERSPECTIVES

    if DEBUG_MODE:
        start_debug_process(co, OCR_ENGINE, DUPLICATION_DETECTION_ENGINE, PERSPECTIVES)
    else:
        while True:
            try:
                if co is None:
                    co = sqlite3.connect(DATABASE_PATH)
                start_process(
                    co, OCR_ENGINE, DUPLICATION_DETECTION_ENGINE, PERSPECTIVES
                )
            except Exception as e:
                print(f"Error: {e}")
                continue


def start_process(co, ocr_engine, duplication_detection_engine, perspectives):
    sum_download_time = 0.0000001
    sum_e2p_time = 0.0000001
    sum_ocr_time = 0.0000001
    sum_duplication_removal_time = 0.0000001
    sum_total_time = 0.0000001
    sum_success = 0

    N = 100

    print(f"Getting {N} panoramas in all boroughs")

    pano_ids = get_n_pano_id_without_ocr(co, N)

    for i, pano_id in enumerate(pano_ids):
        if sum_success > 0:
            avg_download_time = sum_download_time / sum_success
            avg_e2p_time = sum_e2p_time / sum_success
            avg_ocr_time = sum_ocr_time / sum_success
            avg_duplication_removal_time = sum_duplication_removal_time / sum_success
            avg_total_time = sum_total_time / sum_success
            print(f"t_dl\tt_e2p\tt_ocr\tt_dup\tt_total")
            print(
                f"{avg_download_time:.1f}\t{avg_e2p_time:.1f}\t{avg_ocr_time:.1f}\t{avg_duplication_removal_time:.1f}\t{avg_total_time:.1f}"
            )
            pano_per_minute = 60 / avg_total_time
            print(f"Panoramas per minute: {pano_per_minute:.3f}")

        print(f"Processing {i + 1}/{N}")
        try:
            download_time = 0
            current_time = time.time()
            print(f"{pano_id}\tDownloading")
            add_one_to_download_count(pano_id, co)

            try:
                panorama_pil_image = get_streetview_image_with_timeout(pano_id)
            except TimeoutError:
                print(f"{pano_id}\tDownload timed out after 5 seconds, skipping...")
                continue

            download_time += time.time() - current_time
            result = ocr_google_streetview_from_id(
                pano_id,
                panorama_pil_image,
                perspectives,
                ocr_engine,
                duplication_detection_engine,
            )

            insert_ocr_result(co, result)
            sum_download_time += result.download_time + download_time
            sum_e2p_time += result.e2p_time
            sum_ocr_time += result.ocr_time
            sum_duplication_removal_time += result.duplication_removal_time
            sum_total_time += result.total_time
            sum_success += 1

            if SAVE_RESULT:
                result.save_to_dir("./temp")

        except Exception as e:
            print(f"{pano_id}\tError: {e}")

        finally:
            clear_console()
            print(f"UUID / {UNIQUE_ID}")


if __name__ == "__main__":
    print("Setting up database")
    setup_database(DATABASE_PATH)

    print("Connecting to database")
    CONNECTION = sqlite3.connect(DATABASE_PATH)
    try:
        main(CONNECTION)
    finally:
        print("Closing database connection")
        CONNECTION.close()

        print("Removing temp directory")
        if os.name == "nt":
            os.system(f"rmdir /s /q {TEMP_DIR}")
        else:
            os.system(f"rm -rf {TEMP_DIR}")
