"""Helper script for counting fruits in the image registered in flight."""

import cv2
import numpy as np
import os
from typing import List, Tuple

# TODO
# 1. Detekcja śmigła...

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images_eval"

h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255
light = 100

Patch = np.ndarray
PatchCoords = Tuple[int, int, int, int]


def mask_plants(hsv: np.ndarray) -> np.ndarray:
    """Generate mask for plants based on HSV image.

    Args:
        hsv (np.ndarray): HSV image

    Returns:
        np.ndarray: Mask for plants
    """
    # TODO Przenieść progi do gory
    # Wykrywanie bialych obszarow
    mask_white = cv2.inRange(hsv, (0, 0, 250), (5, 3, 255))

    # TODO Przenieść progi do gory
    # Wykrywanie zielonych obszarow
    mask_green = cv2.inRange(hsv, (50, 0, 45), (60, 180, 255))

    mask_wg = cv2.bitwise_or(mask_white, mask_green)

    # cv2.imshow('Mask white and green',mask_wg)

    # Eksperyment
    kernel_1_25 = np.ones((1, 25), np.uint8)
    kernel_25_1 = np.ones((25, 1), np.uint8)

    th_RGB_F = cv2.dilate(mask_wg, kernel_1_25)
    th_RGB_F = cv2.dilate(th_RGB_F, kernel_25_1)

    th_RGB_F = cv2.medianBlur(th_RGB_F, 7)
    th_RGB_F = np.uint8(th_RGB_F / 255)

    # cv2.imshow("Mask after median blur",mask_wg)
    return th_RGB_F


def mask_rotors(hsv: np.ndarray) -> np.ndarray:
    """Generate mask for rotors based on HSV image.

    Args:
        hsv (np.ndarray): HSV image

    Returns:
        np.ndarray: Mask for rotors
    """
    # TODO do ew. korekty
    mask_blue = cv2.inRange(hsv, (113, 75, 0), (122, 255, 255))
    mask_red = cv2.inRange(hsv, (0, 230, 0), (2, 255, 255))
    mask_rotors = cv2.bitwise_or(mask_blue, mask_red)

    return mask_rotors


def get_patches(
    img_color: np.ndarray, img_depth: np.ndarray, odom: str
) -> Tuple[List[Patch], List[PatchCoords], np.ndarray]:
    """Extract patches from the image.

    Args:
        img_color (np.ndarray): RGB color image
        img_depth (np.ndarray): Depth image
        odom (str): Odometry data

    Returns:
        Tuple[List[Patch], List[PatchCoords], np.ndarray]: Extracted patches \
            (HSV image cutout), patches coordinates, and rotated color image
    """
    # Rotate the image based on the roll angle
    roll = float(odom.split(" ")[3])

    roll_deg = -1 * roll / np.pi * 180
    dimensions = img_color.shape
    cols = int(dimensions[1] * 1.2)
    rows = int(dimensions[0] * 1.2)
    rot_matrix = cv2.getRotationMatrix2D(
        (float(dimensions[0] / 2), float(dimensions[1] / 2)), roll_deg, 1
    )
    img_color_rotated = cv2.warpAffine(img_color, rot_matrix, (cols, rows))
    img_depth_rotated = cv2.warpAffine(img_depth, rot_matrix, (cols, rows))
    # DR = cv2.cvtColor(DR, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("IR", img_color_rotated)
    # cv2.imshow("DR", img_depth_rotated * 10)

    # Depth image segmentation - looking for plants (squares)
    hsv = cv2.cvtColor(img_color_rotated, cv2.COLOR_BGR2HSV)
    plants_mask = mask_plants(hsv)

    # img_depth_rotated = cv2.cvtColor(img_depth_rotated, cv2.COLOR_BGR2GRAY)
    result = cv2.multiply(plants_mask, img_depth_rotated)
    result = cv2.medianBlur(result, 5)
    # cv2.imshow("B", plants_mask * 10)

    plant_mask = np.uint8(result == 3) * 255
    # cv2.imshow("B", plant_mask)

    rotors_mask = mask_rotors(hsv)
    # cv2.imshow("Rotors", rotors_mask)

    # Analysis of the segmented image
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(plant_mask)
    patches: List[Patch] = []
    patches_coords: List[PatchCoords] = []
    for i in range(1, num_labels):
        left, top, width, height, area = stats[i]

        # Magic correction
        left = left + 12
        width = width - 24
        height = height - 12
        # TODO Dodatkowe warunki a ten sprawdzic
        # TODO Korekcja ramki
        sqare_ratio = width / height
        # print(f"SQUARE = {sqare_ratio} AREA = {area}")

        if area > 100 * 100 and sqare_ratio > 1 and sqare_ratio < 1.3:
            # NOTE: There should be no rotors in the fov
            rotor_rotor = rotors_mask[top : top + height, left : left + width]
            # cv2.imshow("RR", rotor_rotor)

            num_labels_r, labels_r, stats_r, centroids_r = (
                cv2.connectedComponentsWithStats(rotor_rotor)
            )

            rotors_in_fov = False
            if num_labels_r > 1:
                for rr in range(1, num_labels_r):
                    _, _, _, _, area_r = stats_r[rr]
                    # TODO a jak będzie jakiś szum ?
                    if area_r > 100:
                        # print(f"Rotor area {area_r}")
                        rotors_in_fov = True

            if not rotors_in_fov:
                # TODO Polaczyc z dylatacja
                # Reczna korekcja (wynika z rozmiaru dylatacji)
                # cv2.rectangle(
                #     img_color_rotated,
                #     (left, top),
                #     (left + width, top + height),
                #     255,
                #     2,
                # )
                patches.append(hsv[top : top + height, left : left + width])
                patches_coords.append((top, top + height, left, left + width))

    return patches, patches_coords, img_color_rotated


if __name__ == "__main__":
    # Read images from the folder
    # NOTE: We use set to remove duplicates
    bed_ids = list(
        {
            name.split("_")[0][0]
            for name in os.listdir(IMAGES_FOLDER_PATH)
            if not name.startswith(".")
        }
    )
    bed_sides = ["0", "1"]

    for bed_id in bed_ids:
        for bed_side in bed_sides:
            unique_id = f"{bed_id}{bed_side}"

            # Get all files with name starting with id
            files = [
                name
                for name in os.listdir(IMAGES_FOLDER_PATH)
                if name.startswith(unique_id)
            ]

            no_images = len(files) // 3
            for i in range(no_images):
                img_color = cv2.imread(
                    f"{IMAGES_FOLDER_PATH}/{unique_id}_{i}_eval_color.png"
                )
                img_depth = cv2.imread(
                    f"{IMAGES_FOLDER_PATH}/{unique_id}_{i}_eval_depth.png",
                    cv2.IMREAD_GRAYSCALE,
                )
                with open(
                    f"{IMAGES_FOLDER_PATH}/{unique_id}_{i}_eval_odom.txt", "r"
                ) as f:
                    odom = f.readline()

                _, patches_coords, img_rotated = get_patches(img_color, img_depth, odom)

                for p in patches_coords:
                    top, bottom, left, right = p
                    cv2.rectangle(
                        img_rotated,
                        (left, top),
                        (right, bottom),
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Image rotated", img_rotated)
                cv2.waitKey(0)
