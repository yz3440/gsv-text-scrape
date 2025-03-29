import os
import json
from typing import List
from PIL import Image
import panoocr as po
from dataclasses import dataclass


@dataclass
class StreetViewProcessResult:
    panorama_id: str
    all_sphere_ocr_results: List[po.SphereOCRResult]
    streetview_image: Image.Image
    download_time: float
    e2p_time: float
    ocr_time: float
    duplication_removal_time: float
    total_time: float


    def save_to_dir(self, directory: str, filename: str | None = None):
        if filename is None:
            filename = self.panorama_id

        # save ocr results
        all_sphere_ocr_result_dicts = [
            ocr_result.to_dict() for ocr_result in self.all_sphere_ocr_results
        ]
        all_sphere_ocr_filename = f"{filename}_ocr.json"
        with open(os.path.join(directory, all_sphere_ocr_filename), "w") as f:
            json.dump(all_sphere_ocr_result_dicts, f)

        # save streetview image
        streetview_image_filename = f"{filename}.jpg"
        self.streetview_image.save(os.path.join(directory, streetview_image_filename))
