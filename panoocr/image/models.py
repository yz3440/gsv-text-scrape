import numpy as np
from dataclasses import dataclass
from PIL import Image
from typing import List, Any, Union, Optional
from . import e2p


@dataclass
class PerspectiveMetadata:
    pixel_width: int
    pixel_height: int
    horizontal_fov: float
    vertical_fov: float
    yaw_offset: float
    pitch_offset: float

    def __str__(self):
        return f"PerspectiveMetadata(pixel_width={self.pixel_width}, pixel_height={self.pixel_height}, horizontal_fov={self.horizontal_fov}, vertical_fov={self.vertical_fov}, yaw_offset={self.yaw_offset}, pitch_offset={self.pitch_offset})"

    def to_file_suffix(self):
        return f"{self.pixel_width}_{self.pixel_height}_{self.horizontal_fov}_{self.vertical_fov}_{self.yaw_offset}_{self.pitch_offset}"


class PerspectiveImage:
    source_panorama_image_array: np.ndarray
    panorama_id: str
    perspective_metadata: PerspectiveMetadata

    perspective_image_array: np.ndarray = None
    perspective_image: Image.Image = None

    def __init__(
        self,
        panorama_id: str,
        source_panorama_image_array: np.ndarray,
        perspective_metadata: PerspectiveMetadata,
    ):

        self.source_panorama_image_array = source_panorama_image_array
        self.panorama_id = panorama_id
        self.perspective_metadata = perspective_metadata

        self.perspective_image_array = e2p.e2p(
            e_img=self.source_panorama_image_array,
            fov_deg=(
                self.perspective_metadata.horizontal_fov,
                self.perspective_metadata.vertical_fov,
            ),
            u_deg=self.perspective_metadata.yaw_offset,
            v_deg=self.perspective_metadata.pitch_offset,
            out_hw=(
                self.perspective_metadata.pixel_height,
                self.perspective_metadata.pixel_width,
            ),
            in_rot_deg=0,
            mode="bilinear",
        )

        self.perspective_image = Image.fromarray(self.perspective_image_array)

    def get_perspective_metadata(self):
        return self.perspective_metadata

    def get_perspective_image_array(self):
        return self.perspective_image_array

    def get_perspective_image(self):
        return self.perspective_image


class PanoramaImage:
    panorama_id: str
    loaded_image: Image.Image | None = None
    loaded_image_array: np.ndarray | None = None

    def __init__(
        self,
        panorama_id: str,
        image: Union[str, Image.Image, np.ndarray],
    ):
        self.panorama_id = panorama_id
        if isinstance(image, str):
            self.loaded_image = Image.open(image)
            self.loaded_image_array = np.array(self.loaded_image)

        elif isinstance(image, Image.Image):
            self.loaded_image = image
            self.loaded_image_array = np.array(self.loaded_image)

        elif isinstance(image, np.ndarray):
            self.loaded_image_array = image
            self.loaded_image = Image.fromarray(self.loaded_image_array)

        else:
            raise ValueError(
                "Input image must be a path to an image or a PIL Image object"
            )

    def generate_perspective_image(self, perspective: PerspectiveMetadata):
        if self.loaded_image is None:
            raise ValueError("Image has not been loaded")

        perspective_image = PerspectiveImage(
            source_panorama_image_array=self.loaded_image_array,
            panorama_id=self.panorama_id,
            perspective_metadata=perspective,
        )
        return perspective_image
