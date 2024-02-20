#!/usr/bin/env python
"""Photo taker node to take photos of the plants."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
from typing import Dict, List, Tuple

import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

from icuas24_competition.msg import BedImageData, BedViewArray
from scripts.utils.positions import POINTS_OF_INTEREST

bridge = CvBridge()

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images_eval"
PROXIMITY_THRESHOLD = 0.2  # 0.1
ROLL_IDX = 3
ROLL_THRESHOLD = 6 * np.pi / 180
YAW_IDX = 5
YAW_THRESHOLD = 3 * np.pi / 180


class PhotoLogger:
    """Class to log all photos of the plants."""

    def __init__(self, frequency: float, max_images: int):
        """Initialize the PhotoTaker class.

        Args:
            frequency (float): The frequency of the node.
        """
        self.current_color_path: str = None
        self.current_depth_path: str = None
        self.current_color_msg: Image = None
        self.current_depth_msg: Image = None
        self.current_odom: Odometry = None
        self.rate = rospy.Rate(frequency)
        self.max_images = max_images
        self.bed_view_order: List[Tuple[int, int]] = None
        self.bed_images: Dict[Tuple[int, int], int] = None
        self.current_bed_view_idx: int = None

        # ROS publishers and subscribers
        self.pub_bed_image_data = rospy.Publisher(
            "/avader/bed_image_data", BedImageData, queue_size=10
        )

        rospy.Subscriber(
            "/avader/bed_views",
            BedViewArray,
            self._bed_views_clb,
        )
        rospy.Subscriber("/red/camera/depth/image_raw", Image, self._image_depth_clb)
        rospy.Subscriber("/red/camera/color/image_raw", Image, self._image_color_clb)
        rospy.Subscriber("/red/odometry", Odometry, self._odom_clb)

    def _bed_views_clb(self, msg: BedViewArray):
        if len(msg.bed_views) == 0:
            rospy.logerr("[Photo Logger] Received no bed views.")
            return
        self.bed_view_order = [
            (bed_view.bed_id, bed_view.bed_side) for bed_view in msg.bed_views
        ]
        self.bed_images = {
            (bed_view.bed_id, bed_view.bed_side): 0 for bed_view in msg.bed_views
        }
        self.current_bed_view_idx = 0
        rospy.logdebug(f"[Photo Logger] Bed views received: {self.bed_view_order}")

    def _get_yaw_error(self, yaw1: float, yaw2: float) -> float:
        """Calculate the yaw error between two angles in radians.

        Args:
            yaw1 (float): The first angle [radians].
            yaw2 (float): The second angle [radians].

        Returns:
            float: The yaw error between the two angles in radians.
        """
        yaw_diff = np.abs(yaw1 - yaw2)
        return min(yaw_diff, 2 * np.pi - yaw_diff)

    def _image_color_clb(self, msg: Image):
        self.current_color_msg = msg

    def _image_depth_clb(self, msg: Image):
        self.current_depth_msg = msg

    def _odom_clb(self, msg: Odometry):
        self.current_odom = msg

    def is_data_initialized(self):
        """Check if the data is initialized.

        Returns:
            bool: True if the data is initialized, False otherwise.
        """
        return (
            self.current_color_msg is not None
            and self.current_depth_msg is not None
            and self.current_odom is not None
            and self.current_bed_view_idx is not None
        )

    def run(self):
        """Run the photo logger node.

        This method is the main loop of the photo logger node.
        """
        # Wait for the data initialization
        while not rospy.is_shutdown() and not self.is_data_initialized():
            self.rate.sleep()

        # Main loop
        rospy.loginfo("[Photo Logger] Entering main loop.")
        while not rospy.is_shutdown() and self.current_bed_view_idx < len(
            self.bed_view_order
        ):
            # Get distances from odom_position to all points of interest
            odom_data = np.array(
                [
                    self.current_odom.pose.pose.position.x,
                    self.current_odom.pose.pose.position.y,
                    self.current_odom.pose.pose.position.z,
                    *euler_from_quaternion(
                        [
                            self.current_odom.pose.pose.orientation.x,
                            self.current_odom.pose.pose.orientation.y,
                            self.current_odom.pose.pose.orientation.z,
                            self.current_odom.pose.pose.orientation.w,
                        ]
                    ),
                ]
            )

            # Check if the UAV is close to the next bed view than the current one
            bed_view = self.bed_view_order[self.current_bed_view_idx]
            distance = np.linalg.norm(
                odom_data[:3]
                - np.array(POINTS_OF_INTEREST[bed_view[0]][bed_view[1]][:3])
            )
            yaw_diff = self._get_yaw_error(
                odom_data[YAW_IDX],
                POINTS_OF_INTEREST[bed_view[0]][bed_view[1]][YAW_IDX],
            )
            if self.current_bed_view_idx < len(self.bed_view_order) - 1:
                next_bed_view = self.bed_view_order[self.current_bed_view_idx + 1]
                distance_next = np.linalg.norm(
                    odom_data[:3]
                    - np.array(
                        POINTS_OF_INTEREST[next_bed_view[0]][next_bed_view[1]][:3]
                    )
                )
                yaw_diff_next = self._get_yaw_error(
                    odom_data[YAW_IDX],
                    POINTS_OF_INTEREST[next_bed_view[0]][next_bed_view[1]][YAW_IDX],
                )
                if distance_next < distance:
                    self.current_bed_view_idx += 1
                    bed_view = next_bed_view
                    distance = distance_next
                    yaw_diff = yaw_diff_next
                elif distance_next == distance:
                    if yaw_diff_next < yaw_diff:
                        self.current_bed_view_idx += 1
                        bed_view = next_bed_view
                        distance = distance_next
                        yaw_diff = yaw_diff_next

            rospy.logdebug(
                f"[Photo Logger] Searching ({bed_view[0]}, {bed_view[1]}): "
                f"Distance: {distance}, Yaw diff: {yaw_diff}, "
                f"Roll: {odom_data[ROLL_IDX]}"
            )
            # Write images only if the UAV is close to a point of interest
            if (
                distance < PROXIMITY_THRESHOLD
                and odom_data[ROLL_IDX] < ROLL_THRESHOLD
                and yaw_diff < YAW_THRESHOLD
            ):
                img_color = bridge.imgmsg_to_cv2(self.current_color_msg, "bgr8")
                img_depth = bridge.imgmsg_to_cv2(self.current_depth_msg, "8UC1")

                bed_id = bed_view[0]
                bed_side = bed_view[1]
                img_idx = self.bed_images[(bed_id, bed_side)]

                unique_id = f"{bed_id}{bed_side}_{img_idx}_eval"
                path = f"{IMAGES_FOLDER_PATH}/{unique_id}"
                path_color = f"{path}_color.png"
                path_depth = f"{path}_depth.png"
                path_odom = f"{path}_odom.txt"

                cv2.imwrite(path_color, img_color)
                cv2.imwrite(path_depth, img_depth)
                with open(path_odom, "w") as out_file:
                    odom_txt = (
                        f"{odom_data[0]} {odom_data[1]} {odom_data[2]} "
                        f"{odom_data[3]} {odom_data[4]} {odom_data[5]}"
                    )
                    out_file.write(odom_txt)

                self.bed_images[(bed_id, bed_side)] += 1
                enaugh_data = self.bed_images[(bed_id, bed_side)] >= self.max_images
                if enaugh_data:
                    self.current_bed_view_idx += 1

                img_data_msg = BedImageData()
                img_data_msg.bed_id = bed_id
                img_data_msg.bed_side = bed_side
                img_data_msg.enaugh_data = enaugh_data
                img_data_msg.img_color = self.current_color_msg
                img_data_msg.img_depth = self.current_depth_msg
                img_data_msg.odom_data.x = odom_data[0]
                img_data_msg.odom_data.y = odom_data[1]
                img_data_msg.odom_data.z = odom_data[2]
                img_data_msg.odom_data.roll = odom_data[3]
                img_data_msg.odom_data.pitch = odom_data[4]
                img_data_msg.odom_data.yaw = odom_data[5]
                self.pub_bed_image_data.publish(img_data_msg)

                rospy.logdebug(
                    f"[Photo Logger] ({bed_id}, {bed_side}) Data saved: "
                    f"{self.bed_images[(bed_id, bed_side)]}/{self.max_images}"
                )
            self.rate.sleep()
        rospy.loginfo("[Photo Logger] Finished.")


if __name__ == "__main__":
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) < 2:
        rospy.logerr("[Photo Logger] Usage: photo_logger.py <frequency> <max_images>")
        sys.exit(1)
    frequency = float(myargv[1])
    max_images = int(myargv[2])

    rospy.init_node("photo_logger")
    rospy.loginfo(
        f"[Photo Logger] Node started with params:\n"
        f"\tFrequency: {frequency} Hz\n"
        f"\tMax images: {max_images}"
    )

    photo_logger = PhotoLogger(frequency, max_images)
    photo_logger.run()
