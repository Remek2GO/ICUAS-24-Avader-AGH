#!/usr/bin/env python
"""Node to analyze images and count fruits in the plant beds."""

import cv2
import os
import sys
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from icuas24_competition.msg import AnalyzerResult, BedImageData
from scripts.utils.detect_fruits import process_patch
from scripts.utils.inflight_image_analysis import get_patches
from scripts.utils.plant_bed import PlantBed, PlantSideCount
from scripts.utils.types import PlantType

bridge = CvBridge()


class PhotoAnalyzer:
    """Class to analyze the photos of the plants."""

    def __init__(self, frequency: float, eval_mode: bool = False):
        """Initialize the PhotoAnalyzer class.

        Args:
            frequency (float): The frequency of the node.
        """
        self.rate = rospy.Rate(frequency)
        self.bed_image_data_queue: List[BedImageData] = []
        self.result_image: Image = None

        self.plant_beds: Dict[int, PlantBed] = {}
        self.fruit_type: PlantType = None

        self.bed_view_errors: Dict[Tuple[int, int], float] = {}
        self.roll_error_coefficient = 0.3
        self.pitch_error_coefficient = 0.1
        self.yaw_error_coefficient = 0.6

        self.eval_mode = eval_mode

        # ROS publishers and subscribers
        self.pub_current_fruit_count = rospy.Publisher(
            "/avader/current_fruit_count", Int32, queue_size=10
        )
        self.pub_output_image = rospy.Publisher(
            "/avader/output_image", Image, queue_size=10
        )

        if self.eval_mode:
            self.pub_analyzer_result = rospy.Publisher(
                "/evaluator/analyzer_result", AnalyzerResult, queue_size=10
            )

        rospy.Subscriber("/red/plants_beds", String, self._fruit_type_clb)
        rospy.Subscriber("/avader/bed_image_data", BedImageData, self._image_data_clb)

    def _calculate_angle_error(
        self, roll_error: float, pitch_error: float, yaw_error: float
    ):
        return (
            self.roll_error_coefficient * roll_error
            + self.pitch_error_coefficient * pitch_error
            + self.yaw_error_coefficient * yaw_error
        )

    def _fruit_type_clb(self, msg: String):
        fruit_name = msg.data.split(" ")[0]
        self.fruit_type = PlantType(fruit_name.upper())

    def _image_data_clb(self, msg: BedImageData):
        self.bed_image_data_queue.append(msg)
        rospy.logdebug(
            f"[Photo Analyzer] New image data received. Queue size: "
            f"{len(self.bed_image_data_queue)}"
        )

    def get_fruit_count(self) -> int:
        """Get the total fruit count.

        Returns:
            int: The total fruit count.
        """
        fruit_sum = 0
        for bed_id in self.plant_beds.keys():
            no_fruits = self.plant_beds[bed_id].get_bed_fruit_count(self.fruit_type)
            fruit_sum += no_fruits

        return fruit_sum

    def run(self):
        """Run the node."""
        while not rospy.is_shutdown():
            # NOTE: The rate.sleep() is intentionally placed at the beginning of the
            # loop to facilitate using the continue statement
            self.rate.sleep()

            # Check if there is any image data to process
            if len(self.bed_image_data_queue) > 0:
                rospy.logdebug(
                    f"[Photo Analyzer] Processing the image data, "
                    f"queue size: {len(self.bed_image_data_queue)}"
                )
                bed_image_data = self.bed_image_data_queue.pop(0)

                # Decide if the current image is better than the previous one(s)
                bed_view = (bed_image_data.bed_id, bed_image_data.bed_side)
                current_error = self._calculate_angle_error(
                    bed_image_data.roll_error,
                    bed_image_data.pitch_error,
                    bed_image_data.yaw_error,
                )
                if bed_view not in self.bed_view_errors:
                    self.bed_view_errors[bed_view] = current_error

                # Add the plant to the plant bed if it does not exist
                if bed_image_data.bed_id not in self.plant_beds:
                    self.plant_beds[bed_image_data.bed_id] = PlantBed()

                if current_error > self.bed_view_errors[bed_view]:
                    rospy.logdebug(
                        f"[Photo Analyzer] Bed view error: {current_error} > "
                        f"{self.bed_view_errors[bed_view]}"
                    )
                    continue

                # Process the image
                img_color = bridge.imgmsg_to_cv2(bed_image_data.img_color, "bgr8")
                img_depth = bridge.imgmsg_to_cv2(bed_image_data.img_depth, "8UC1")
                # TODO: Odom data as a string is a temporary solution to ensure the
                # compatibility with previous code
                odom_data = f"{bed_image_data.odom_data.x} "
                odom_data += f"{bed_image_data.odom_data.y} "
                odom_data += f"{bed_image_data.odom_data.z} "
                odom_data += f"{bed_image_data.odom_data.roll} "
                odom_data += f"{bed_image_data.odom_data.pitch} "
                odom_data += f"{bed_image_data.odom_data.yaw}"
                patches, patches_coords, img_rotated = get_patches(
                    img_color, img_depth, odom_data
                )

                # Sort patches based on their location
                z_patches = zip(patches, patches_coords)
                z_patches_sort = sorted(z_patches, key=lambda x: x[1][2])

                # Process each patch
                for i, (patch, patch_coords) in enumerate(z_patches_sort):
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

                    # Reverse the index if the bed side is 1
                    idx = i if bed_image_data.bed_side == 0 else len(patches) - i - 1

                    # Gather the data
                    self.plant_beds[bed_image_data.bed_id].set_plant(
                        idx,
                        bed_image_data.bed_side,
                        plant_side.fruit_count,
                        plant_side.fruit_position.copy(),
                        plant_side.fruit_type,
                    )

                # Publish image with detection results
                self.result_image = bridge.cv2_to_imgmsg(img_rotated, "bgr8")

                # Publish the current fruit count
                current_fruit_count = self.get_fruit_count()
                self.pub_current_fruit_count.publish(current_fruit_count)

                # Update lowest error
                self.bed_view_errors[bed_view] = current_error
                rospy.logdebug(
                    f"[Photo Analyzer] Bed view error updated: "
                    f"{self.bed_view_errors[bed_view]}"
                )

                # NOTE: Evaluation only
                if self.eval_mode:
                    # Publish the result
                    analyzer_result = AnalyzerResult(
                        bed_id=bed_image_data.bed_id,
                        bed_side=bed_image_data.bed_side,
                        fruit_sum=self.plant_beds[
                            bed_image_data.bed_id
                        ].get_bed_fruit_count(self.fruit_type),
                        fruit_right=self.plant_beds[
                            bed_image_data.bed_id
                        ].get_bed_fruit_count_right(self.fruit_type),
                        fruit_left=self.plant_beds[
                            bed_image_data.bed_id
                        ].get_bed_fruit_count_left(self.fruit_type),
                    )
                    self.pub_analyzer_result.publish(analyzer_result)

            # Publish the result image
            if self.result_image is not None:
                self.pub_output_image.publish(self.result_image)


if __name__ == "__main__":
    myargs = rospy.myargv(argv=sys.argv)
    if len(myargs) < 2:
        print("Usage: photo_analyzer.py <frequency>")
        sys.exit(1)
    frequency = float(myargs[1])
    log_level = rospy.DEBUG if "--debug" in myargs else rospy.INFO
    eval_mode = "--eval" in myargs

    rospy.init_node("photo_analyzer", log_level=log_level)
    rospy.loginfo(f"[Photo Analyzer] Node started with params:\n" f"\t{frequency} Hz")

    photo_analyzer = PhotoAnalyzer(frequency, eval_mode)
    photo_analyzer.run()
