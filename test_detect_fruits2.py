import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from scripts.utils.inflight_image_analysis import get_patches
from scripts.utils.detect_fruits import process_patch
from scripts.utils.types import PlantType
from scripts.utils.plant_bed import PlantBed, PlantSideCount
from typing import NewType
from icuas24_competition.msg import BedImageData, BedView, BedViewArray
from scripts.utils.positions import POINTS_OF_INTEREST, PointOfInterest

from scripts.evaluator import Plant as ev_Plant, PlantBed as ev_PlantBed

import csv

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images_eval"

h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255
light = 100

Patch = np.ndarray
PatchCoords = Tuple[int, int, int, int]

plant_beds: Dict[int, PlantBed] = {}
bed_view_errors: Dict[Tuple[int, int], float] = {}
bed_view_poses: np.ndarray = None
bed_view_encoding: Dict[int, Tuple[int, int]] = None

roll_error_coefficient = 0.3
pitch_error_coefficient = 0.1
yaw_error_coefficient = 0.6

ROLL_IDX = 3
PITCH_IDX = 4
YAW_IDX = 5
PROXIMITY_THRESHOLD = 0.2  # 0.1




def get_fruit_count(plant_beds) -> int:
    """Get the total fruit count.

    Returns:
        int: The total fruit count.
    """
    fruit_sum = 0
    for bed_id in plant_beds.keys():
        no_fruits = plant_beds[bed_id].get_bed_fruit_count(fruit_type)
        fruit_sum += no_fruits

    return fruit_sum


def _get_yaw_error(yaw1: float, yaw2: float) -> float:
    """Calculate the yaw error between two angles in radians.

    Args:
        yaw1 (float): The first angle [radians].
        yaw2 (float): The second angle [radians].

    Returns:
        float: The yaw error between the two angles in radians.
    """
    yaw_diff = np.abs(yaw1 - yaw2)

    return min(yaw_diff, 2 * np.pi - yaw_diff)


def _calculate_angle_error(
        roll_error: float, pitch_error: float, yaw_error: float
    ):
    return (
        roll_error_coefficient * roll_error
        + pitch_error_coefficient * pitch_error
        + yaw_error_coefficient * yaw_error
    )



if __name__ == "__main__":

    path_to_beds_csv = '/root/sim_ws/src/icuas24_competition/beds.csv'
    csv_plant_beds = []
    with open(path_to_beds_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # NOTE: We remove the last character from the plant type to remove the
            # trailing 's'
            left = ev_Plant(
                PlantType(row[1][:-1].upper()),
                int(row[2]),
                int(row[3]),
                int(row[4]),
            )
            centre = ev_Plant(
                PlantType(row[5][:-1].upper()),
                int(row[6]),
                int(row[7]),
                int(row[8]),
            )
            right = ev_Plant(
                PlantType(row[9][:-1].upper()),
                int(row[10]),
                int(row[11]),
                int(row[12]),
            )
            new_plant_bed = ev_PlantBed(left, centre, right)
            csv_plant_beds.append(new_plant_bed)

    # print(csv_plant_beds[0].left.plant_type)
    # exit()


    # Read images from the folder
    # NOTE: We use set to remove duplicates
    bed_ids = list(
        {
            name.split("_")[0][:-1]
            for name in os.listdir(IMAGES_FOLDER_PATH)
            if not name.startswith(".")
        }
    )
    bed_sides = ["0", "1"]
    bed_sides_num = [0, 1]

    # bed_ids = [13]
    bed_ids = [int(bed_id) for bed_id in bed_ids]

    bed_view_poses_list: List[PointOfInterest] = []
    bed_view_encoding = {}
    bed_images = {}
    bed_view: BedView
    idx_b = 0
    for k, bed_id in enumerate(bed_ids):
        for bed_side in bed_sides_num:
            bed_view_poses_list.append(
                POINTS_OF_INTEREST[bed_id][bed_side]
            )
            bed_view_encoding[idx_b] = (
                bed_id,
                bed_side,
            )
            bed_images[(bed_id, bed_side)] = 0
            idx_b += 1
    
    bed_view_poses = np.array(bed_view_poses_list)

    for bed_id in bed_ids:
        for bed_side in bed_sides:
            unique_id = f"{bed_id}{bed_side}_"
            bed_side = int(bed_side)

            # Get all files with name starting with id
            files = [
                name
                for name in os.listdir(IMAGES_FOLDER_PATH)
                if name.startswith(unique_id)
            ]

            no_images = len(files) // 3
            print(f"Processing {no_images} images for {unique_id}")
            for i in range(no_images):
                img_color = cv2.imread(
                    f"{IMAGES_FOLDER_PATH}/{unique_id}{i}_eval_color.png"
                )
                img_depth = cv2.imread(
                    f"{IMAGES_FOLDER_PATH}/{unique_id}{i}_eval_depth.png",
                    cv2.IMREAD_GRAYSCALE,
                )
                with open(
                    f"{IMAGES_FOLDER_PATH}/{unique_id}{i}_eval_odom.txt", "r"
                ) as f:
                    odom = f.readline()

                odom_data = np.asarray(odom.split()).astype(np.float64)
                # print(odom_data)


                distances = np.linalg.norm(
                    bed_view_poses[:, :3] - odom_data[:3], axis=1
                )
                # rospy.logdebug(f"[Photo Logger] Distances: {distances}")

                # Get the indices of the closest bed views
                closest_idx = np.argwhere(distances < PROXIMITY_THRESHOLD)
                # rospy.logdebug(f"[Photo Logger] Closest {len(closest_idx)} "
                #                f"indices: {closest_idx}")

                # Check if the UAV is close to any bed view
                if len(closest_idx) == 0:
                    continue

                # Check if we have more than one closest bed view
                # If so, we take the one with the smallest yaw error
                closest_idx = closest_idx.flatten()

                if len(closest_idx) > 1:
                    yaw = odom_data[YAW_IDX]
                    yaw_errors = [
                        _get_yaw_error(yaw, bed_view_poses[idx, YAW_IDX])
                        for idx in closest_idx
                    ]
                    closest_idx = closest_idx[np.argmin(yaw_errors)]
                else:
                    closest_idx = closest_idx[0]
                bed_view = bed_view_encoding[closest_idx]


                roll_diff = np.abs(odom_data[ROLL_IDX])
                pitch_diff = np.abs(odom_data[PITCH_IDX])
                yaw_diff = _get_yaw_error(
                    odom_data[YAW_IDX], bed_view_poses[closest_idx, YAW_IDX]
                )

                # Decide if the current image is better than the previous one(s)
                bed_view = (bed_id, bed_side)
                current_error = _calculate_angle_error(
                    roll_diff,
                    pitch_diff,
                    yaw_diff
                )
                if bed_view not in bed_view_errors:
                    bed_view_errors[bed_view] = current_error

                if current_error > bed_view_errors[bed_view]:
                    continue


                patches, patches_coords, img_rotated = get_patches(
                    img_color, img_depth, odom
                )

                for p in patches_coords:
                    top, bottom, left, right = p
                    cv2.rectangle(
                        img_rotated,
                        (left, top),
                        (right, bottom),
                        (0, 255, 0),
                        2,
                    )


                for k, (patch, patch_coords) in enumerate(zip(patches, patches_coords)):
                    fruit_count, fruit_type, fruit_centres = process_patch(patch)
                    # Mark plants
                    cv2.rectangle(
                        img_rotated,
                        (patch_coords[2], patch_coords[0]),
                        (patch_coords[3], patch_coords[1]),
                        (255, 0, 0),
                        2,
                    )
                    # Mark fruits
                    for centre in fruit_centres:
                        cv2.circle(
                            img_rotated,
                            (
                                int(
                                    centre[1] * (patch_coords[3] - patch_coords[2])
                                    + patch_coords[2]
                                ),
                                int(
                                    centre[0] * (patch_coords[1] - patch_coords[0])
                                    + patch_coords[0]
                                ),
                            ),
                            5,
                            (0, 255, 0),
                            -1,
                        )

                    # Check fruit type
                    plant_type = PlantType.EMPTY
                    if fruit_type == 0:
                        plant_type = PlantType.TOMATO
                    elif fruit_type == 1:
                        plant_type = PlantType.EGGPLANT
                    elif fruit_type == 2:
                        plant_type = PlantType.PEPPER

                    # Save the obtained data
                    plant_side = PlantSideCount(
                        fruit_count=fruit_count,
                        fruit_position=fruit_centres,
                        fruit_type=plant_type,
                    )

                    # Add the plant to the plant bed if it does not exist
                    if bed_id not in plant_beds:
                        plant_beds[bed_id] = PlantBed()

                    # Reverse the index if the bed side is 1
                    idx = k if bed_side == 0 else len(patches) - k - 1

                    
                    # Gather the data
                    plant_beds[bed_id].set_plant(
                        idx,
                        bed_side,
                        plant_side.fruit_count,
                        plant_side.fruit_position.copy(),
                        plant_side.fruit_type,
                    )

                cv2.imshow(f"Image rotated_{bed_id}_{i}", img_rotated)
                cv2.waitKey(1)
                cv2.destroyAllWindows()

        print(f"Bed {bed_id}")
        for fruit_type in [PlantType.TOMATO, PlantType.EGGPLANT, PlantType.PEPPER]:
            fruit_sum = plant_beds[bed_id].get_bed_fruit_count(fruit_type)
            fruit_right = plant_beds[bed_id].get_bed_fruit_count_right(fruit_type)
            fruit_left = plant_beds[bed_id].get_bed_fruit_count_left(fruit_type)

            print(f"Fruit type: {fruit_type}")
            print(f"Total fruit count: {fruit_sum}")
            print(f"Right fruit count: {fruit_right}")
            print(f"Left fruit count: {fruit_left}")

            
            if csv_plant_beds[bed_id - 1].left.plant_type == fruit_type:
                print(f"Fruit type: {fruit_type}")
                print(f"CSV Left fruit count: {csv_plant_beds[bed_id - 1].left.left_fruits}")
                print(f"CSV right fruit count: {csv_plant_beds[bed_id - 1].left.right_fruits}")
                print(f"CSV all fruit count: {csv_plant_beds[bed_id - 1].left.all_fruits}")

                if csv_plant_beds[bed_id - 1].left.left_fruits != fruit_left:
                    print(f"Left fruit count does not match")
                
                if csv_plant_beds[bed_id - 1].left.right_fruits != fruit_right:
                    print(f"Right fruit count does not match")
                
                if csv_plant_beds[bed_id - 1].left.all_fruits != fruit_sum:
                    print(f"All fruit count does not match")


            if csv_plant_beds[bed_id - 1].centre.plant_type == fruit_type:
                print(f"Fruit type: {fruit_type}")
                print(f"CSV left fruit count: {csv_plant_beds[bed_id - 1].centre.left_fruits}")
                print(f"CSV right fruit count: {csv_plant_beds[bed_id - 1].centre.right_fruits}")
                print(f"CSV all fruit count: {csv_plant_beds[bed_id - 1].centre.all_fruits}")

                if csv_plant_beds[bed_id - 1].centre.left_fruits != fruit_left:
                    print(f"Left fruit count does not match")

                if csv_plant_beds[bed_id - 1].centre.right_fruits != fruit_right:
                    print(f"Right fruit count does not match")

                if csv_plant_beds[bed_id - 1].centre.all_fruits != fruit_sum:
                    print(f"All fruit count does not match")

            if csv_plant_beds[bed_id - 1].right.plant_type == fruit_type:
                print(f"Fruit type: {fruit_type}")
                print(f"CSV left fruit count: {csv_plant_beds[bed_id - 1].right.left_fruits}")
                print(f"CSV right fruit count: {csv_plant_beds[bed_id - 1].right.right_fruits}")
                print(f"CSV all fruit count: {csv_plant_beds[bed_id - 1].right.all_fruits}")

                if csv_plant_beds[bed_id - 1].right.left_fruits != fruit_left:
                    print(f"Left fruit count does not match")

                if csv_plant_beds[bed_id - 1].right.right_fruits != fruit_right:
                    print(f"Right fruit count does not match")

                if csv_plant_beds[bed_id - 1].right.all_fruits != fruit_sum:
                    print(f"All fruit count does not match")

        


    current_fruit_count = get_fruit_count(plant_beds)
