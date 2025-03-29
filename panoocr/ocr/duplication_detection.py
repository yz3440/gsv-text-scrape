from .models import SphereOCRResult
from dataclasses import dataclass
from typing import List

import geopandas as gpd
from shapely.geometry import Polygon
import textdistance


@dataclass
class RegionIntersection:
    region_1_area: float
    region_2_area: float
    intersection_area: float
    intersection_ratio: float


DEFAULT_MIN_TEXT_SIMILARITY = 0.5
DEFAULT_MIN_INTERSECTION_RATIO_FOR_SIMILAR_TEXT = 0.5
DEFAULT_MIN_TEXT_OVERLAP = 0.5
DEFAULT_MIN_INTERSECTION_RATIO_FOR_OVERLAPPING_TEXT = 0.15
DEFAULT_MIN_INTERSECTION_RATIO = 0.1


class SphereOCRDuplicationDetectionEngine:
    min_text_similarity: float
    min_intersection_ratio_for_similar_text: float
    min_text_overlap: float
    min_intersection_ratio_for_overlapping_text: float
    min_intersection_ratio: float

    def __init__(
        self,
        min_text_similarity=DEFAULT_MIN_TEXT_SIMILARITY,
        min_intersection_ratio_for_similar_text=DEFAULT_MIN_INTERSECTION_RATIO_FOR_SIMILAR_TEXT,
        min_text_overlap=DEFAULT_MIN_TEXT_OVERLAP,
        min_intersection_ratio_for_overlapping_text=DEFAULT_MIN_INTERSECTION_RATIO_FOR_OVERLAPPING_TEXT,
        min_intersection_ratio=DEFAULT_MIN_INTERSECTION_RATIO,
    ):
        self.min_text_similarity = min_text_similarity
        self.min_intersection_ratio_for_similar_text = (
            min_intersection_ratio_for_similar_text
        )
        self.min_text_overlap = min_text_overlap
        self.min_intersection_ratio_for_overlapping_text = (
            min_intersection_ratio_for_overlapping_text
        )
        self.min_intersection_ratio = min_intersection_ratio

    def __sphere_ocr_to_polygon(self, sphere_ocr: SphereOCRResult) -> Polygon:
        bounding_box = {
            "left": sphere_ocr.yaw - sphere_ocr.width / 2,
            "right": sphere_ocr.yaw + sphere_ocr.width / 2,
            "top": sphere_ocr.pitch + sphere_ocr.height / 2,
            "bottom": sphere_ocr.pitch - sphere_ocr.height / 2,
        }

        points = [
            (bounding_box["left"], bounding_box["top"]),
            (bounding_box["right"], bounding_box["top"]),
            (bounding_box["right"], bounding_box["bottom"]),
            (bounding_box["left"], bounding_box["bottom"]),
        ]

        return Polygon(points)

    def __polygon_to_gdf(self, polygon: Polygon) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon])
        gdf.to_crs(3857, inplace=True)
        return gdf

    def __sphere_ocr_to_gdf(self, sphere_ocr: SphereOCRResult) -> gpd.GeoDataFrame:
        return self.__polygon_to_gdf(self.__sphere_ocr_to_polygon(sphere_ocr))

    def __get_intersection(
        self, gdf_1: gpd.GeoDataFrame, gdf_2: gpd.GeoDataFrame
    ) -> RegionIntersection | None:
        gdf_intersection = gpd.overlay(gdf_1, gdf_2, how="intersection")

        if gdf_intersection.empty:
            return None

        region_1_area = gdf_1.area.values[0]
        region_2_area = gdf_2.area.values[0]
        intersection_area = intersection_area = gdf_intersection.area.values[0]

        intersection_ratio = intersection_area / min(region_1_area, region_2_area)

        return RegionIntersection(
            region_1_area=region_1_area,
            region_2_area=region_2_area,
            intersection_area=intersection_area,
            intersection_ratio=intersection_ratio,
        )

    def __intersect_ocr_results(
        self,
        ocr_results_1: SphereOCRResult,
        ocr_results_2: SphereOCRResult,
    ) -> RegionIntersection | None:

        gdf_1 = self.__sphere_ocr_to_gdf(ocr_results_1)
        gdf_2 = self.__sphere_ocr_to_gdf(ocr_results_2)

        intersection = self.__get_intersection(gdf_1, gdf_2)

        if intersection is None:
            return None
        return intersection

    def __get_texts_similarity(self, text_1: str, text_2: str) -> float:
        return textdistance.levenshtein.normalized_similarity(text_1, text_2)

    def __get_texts_overlap(self, text_1: str, text_2: str) -> float:
        return textdistance.overlap.normalized_similarity(text_1, text_2)

    def check_duplication(
        self, ocr_results_1: SphereOCRResult, ocr_results_2: SphereOCRResult
    ) -> bool:
        text_similarity = self.__get_texts_similarity(
            ocr_results_1.text, ocr_results_2.text
        )

        text_overlap = self.__get_texts_overlap(ocr_results_1.text, ocr_results_2.text)

        # if the texts are both not similar and the overlap is not high, then they are not duplicates
        if (text_similarity < self.min_text_similarity) and (
            text_overlap < self.min_text_overlap
        ):
            return False

        intersection = self.__intersect_ocr_results(ocr_results_1, ocr_results_2)
        if intersection is None:
            return False

        if intersection.intersection_ratio < self.min_intersection_ratio:
            return False

        # Check overlap
        if (
            text_overlap >= self.min_text_overlap
            and intersection.intersection_ratio
            >= self.min_intersection_ratio_for_overlapping_text
        ):
            return True

        # Check similarity
        if (
            text_similarity >= self.min_text_similarity
            and intersection.intersection_ratio
            >= self.min_intersection_ratio_for_similar_text
        ):
            return True

        return False

    # def elect_candidate_among_ocr_results(
    #     self, ocr_results: List[SphereOCRResult]
    # ) -> SphereOCRResult:
    #     if len(ocr_results) == 0:
    #         return []

    #     candidate_index = 0
    #     candidate = ocr_results[candidate_index]

    #     # favor the longer text
    #     for i in range(1, len(ocr_results)):
    #         if len(ocr_results[i].text) > len(candidate.text):
    #             # remove the previous candidate
    #             ocr_results[candidate_index] = None

    #             # set the new candidate
    #             candidate = ocr_results[i]
    #             candidate_index = i

    #     return ocr_results

    def remove_duplication_for_two_lists(
        self,
        ocr_results_0: List[SphereOCRResult],
        ocr_results_1: List[SphereOCRResult],
    ) -> bool:

        duplications = []
        for i, ocr_result_0 in enumerate(ocr_results_0):
            for j, ocr_result_1 in enumerate(ocr_results_1):
                if self.check_duplication(ocr_result_0, ocr_result_1):
                    duplications.append([i, j])

        indices_to_remove_from_ocr_results_0 = []
        indices_to_remove_from_ocr_results_1 = []

        for duplication in duplications:
            candidate_ocr_results_0 = ocr_results_0[duplication[0]]
            candidate_ocr_results_1 = ocr_results_1[duplication[1]]

            if len(candidate_ocr_results_0.text) == len(candidate_ocr_results_1.text):
                if (
                    candidate_ocr_results_0.confidence
                    < candidate_ocr_results_1.confidence
                ):
                    indices_to_remove_from_ocr_results_0.append(duplication[0])
                else:
                    indices_to_remove_from_ocr_results_1.append(duplication[1])

            elif len(candidate_ocr_results_0.text) > len(candidate_ocr_results_1.text):
                indices_to_remove_from_ocr_results_1.append(duplication[1])
            else:
                indices_to_remove_from_ocr_results_0.append(duplication[0])

        indices_to_remove_from_ocr_results_0 = list(
            set(indices_to_remove_from_ocr_results_0)
        )
        indices_to_remove_from_ocr_results_0.sort(reverse=True)
        indices_to_remove_from_ocr_results_1 = list(
            set(indices_to_remove_from_ocr_results_1)
        )
        indices_to_remove_from_ocr_results_1.sort(reverse=True)

        for index in indices_to_remove_from_ocr_results_0:
            ocr_results_0.pop(index)

        for index in indices_to_remove_from_ocr_results_1:
            ocr_results_1.pop(index)

        return ocr_results_0, ocr_results_1
