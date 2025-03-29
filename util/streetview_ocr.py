from streetlevel import streetview
from typing import List
import panoocr as po
import os, json
import time
from PIL import Image
from itertools import chain
from .model import StreetViewProcessResult


def flatten_2d_list_itertools(lst):
    return list(chain(*lst))


def get_streetview_image(pano_id: str) -> Image:
    pano = streetview.find_panorama_by_id(pano_id)
    panorama_pil_image = streetview.get_panorama(pano=pano)
    return panorama_pil_image


def ocr_google_streetview_from_id(
    pano_id: str,
    panorama_pil_image: Image,
    perspectives: List[po.PerspectiveMetadata],
    ocr_engine: po.OCREngine,
    duplication_detection_engine: po.SphereOCRDuplicationDetectionEngine,
) -> StreetViewProcessResult:
    begin_time = time.time()

    panorama_image = po.PanoramaImage(pano_id, panorama_pil_image)

    perspective_count = len(perspectives)

    e2p_time = 0
    ocr_time = 0
    duplication_removal_time = 0

    all_sphere_ocr_results_for_each_perspective = []

    # Process each perspective
    print(f"{pano_id}\t ({perspective_count})")
    for i, perspective in enumerate(perspectives):
        print(f"{i}", end=" ", flush=True)
        # Equirectangular to Perspective
        current_time = time.time()
        perspective_image = panorama_image.generate_perspective_image(perspective)
        perspective_pil_image = perspective_image.get_perspective_image()
        e2p_time += time.time() - current_time

        # Recognize Text
        current_time = time.time()
        flat_ocr_results = ocr_engine.recognize(perspective_pil_image)
        sphere_ocr_results = [
            flat_ocr_result.to_sphere(
                horizontal_fov=perspective.horizontal_fov,
                vertical_fov=perspective.vertical_fov,
                yaw_offset=perspective.yaw_offset,
                pitch_offset=perspective.pitch_offset,
            )
            for flat_ocr_result in flat_ocr_results
        ]
        ocr_time += time.time() - current_time

        all_sphere_ocr_results_for_each_perspective.append(sphere_ocr_results)
    print()
    # Remove duplications
    print(f"{pano_id}\tRemoving Duplications")
    current_time = time.time()
    for i in range(0, perspective_count):
        first_perspective_index = i
        second_perspective_index = 0 if i == perspective_count - 1 else i + 1

        (new_ocr_results_first_frame, new_ocr_results_second_frame) = (
            duplication_detection_engine.remove_duplication_for_two_lists(
                all_sphere_ocr_results_for_each_perspective[first_perspective_index],
                all_sphere_ocr_results_for_each_perspective[second_perspective_index],
            )
        )

        all_sphere_ocr_results_for_each_perspective[first_perspective_index] = (
            new_ocr_results_first_frame
        )
        all_sphere_ocr_results_for_each_perspective[second_perspective_index] = (
            new_ocr_results_second_frame
        )

    all_sphere_ocr_results_no_duplication = flatten_2d_list_itertools(
        all_sphere_ocr_results_for_each_perspective
    )

    duplication_removal_time += time.time() - current_time

    total_time = time.time() - begin_time

    return StreetViewProcessResult(
        panorama_id=pano_id,
        all_sphere_ocr_results=all_sphere_ocr_results_no_duplication,
        streetview_image=panorama_pil_image,
        download_time=0,
        e2p_time=e2p_time,
        ocr_time=ocr_time,
        duplication_removal_time=duplication_removal_time,
        total_time=total_time,
    )


def download_and_ocr_google_streetview_from_id(
    pano_id: str,
    perspectives: List[po.PerspectiveMetadata],
    ocr_engine: po.OCREngine,
    duplication_detection_engine: po.SphereOCRDuplicationDetectionEngine,
) -> StreetViewProcessResult:

    download_time = 0
    current_time = time.time()

    pano = streetview.find_panorama_by_id(pano_id)
    panorama_pil_image = streetview.get_panorama(pano=pano)
    download_time += time.time() - current_time

    result = ocr_google_streetview_from_id(
        pano_id,
        panorama_pil_image,
        perspectives,
        ocr_engine,
        duplication_detection_engine,
    )

    result.download_time = download_time

    return result
