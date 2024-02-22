import os
import cv2
import numpy as np
from typing import List, Tuple
from scripts.utils.inflight_image_analysis import get_patches
from scripts.utils.detect_fruits import process_patch

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images_eval"

h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255
light = 100

Patch = np.ndarray
PatchCoords = Tuple[int, int, int, int]

automatic = True

if __name__ == "__main__":
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

    # Automatic
    if (automatic):
        for bed_id in bed_ids:
            for bed_side in bed_sides:
                unique_id = f"{bed_id}{bed_side}_"

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

                    patches, patches_coords, img_rotated = get_patches(img_color, img_depth, odom)

                    for p in patches_coords:
                        top, bottom, left, right = p
                        cv2.rectangle(
                            img_rotated,
                            (left, top),
                            (right, bottom),
                            (0, 255, 0),
                            2,
                        )
                    
                    for z, (patch, patch_coords) in enumerate(zip(patches, patches_coords)):
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


                    cv2.imshow(f"Image rotated_{unique_id}_{i}", img_rotated)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    else:

        name = "171_25"

        img_color = cv2.imread(
                        f"{IMAGES_FOLDER_PATH}/{name}_eval_color.png"
        )
        img_depth = cv2.imread(
            f"{IMAGES_FOLDER_PATH}/{name}_eval_depth.png",
            cv2.IMREAD_GRAYSCALE,
        )
        with open(
            f"{IMAGES_FOLDER_PATH}/{name}_eval_odom.txt", "r"
        ) as f:
            odom = f.readline()

        patches, patches_coords, img_rotated = get_patches(img_color, img_depth, odom)

        for p in patches_coords:
            top, bottom, left, right = p
            cv2.rectangle(
                img_rotated,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2,
            )
        
        for zz, (patch, patch_coords) in enumerate(zip(patches, patches_coords)):
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


        cv2.imshow(f"Image rotated_", img_rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Manual