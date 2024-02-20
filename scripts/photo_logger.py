#!/usr/bin/env python
"""Photo taker node to take photos of the plants."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import numpy as np
from typing import Dict, Tuple

import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

from icuas24_competition.msg import BedImageData, DestinationBedView
from scripts.utils.positions import POINTS_OF_INTEREST

bridge = CvBridge()

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images_eval"
NO_BEDS = 27
NO_SIDES = 2
PROXIMITY_THRESHOLD = 0.1  # 0.1
ROLL_IDX = 3
ROLL_THRESHOLD = np.pi / 180
YAW_IDX = 5
YAW_THRESHOLD = np.pi / 180


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
        self.bed_images: Dict[Tuple[int, int], int] = {
            (bed_id, bed_side): 0
            for bed_id in range(1, NO_BEDS + 1)
            for bed_side in range(NO_SIDES)
        }
        self.current_bed_view: Tuple[int, int] = None
        self.max_images = max_images

        # ROS publishers and subscribers
        self.pub_bed_image_data = rospy.Publisher(
            "/avader/bed_image_data", BedImageData, queue_size=10
        )

        rospy.Subscriber(
            "/avader/destination_bed_view",
            DestinationBedView,
            self._destination_bed_view_clb,
        )
        rospy.Subscriber("/red/camera/depth/image_raw", Image, self._image_depth_clb)
        rospy.Subscriber("/red/camera/color/image_raw", Image, self._image_color_clb)
        rospy.Subscriber("/red/odometry", Odometry, self._odom_clb)

    def _destination_bed_view_clb(self, msg: DestinationBedView):
        self.current_bed_view = (msg.bed_id, msg.bed_side)

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
            and self.current_bed_view is not None
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
        while not rospy.is_shutdown():
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

            # Get the point of interest
            poi = POINTS_OF_INTEREST[self.current_bed_view[0]][self.current_bed_view[1]]
            poi_diff = odom_data - np.array(poi)
            distance = np.linalg.norm(poi_diff[:3])
            roll_diff = np.abs(poi_diff[ROLL_IDX])
            yaw_diff = np.abs(poi_diff[YAW_IDX])

            # Write images only if the UAV is close to a point of interest
            if (
                distance < PROXIMITY_THRESHOLD
                and (
                    roll_diff < ROLL_THRESHOLD or roll_diff > 2 * np.pi - ROLL_THRESHOLD
                )
                and (yaw_diff < YAW_THRESHOLD or yaw_diff > 2 * np.pi - YAW_THRESHOLD)
            ):
                img_color = bridge.imgmsg_to_cv2(self.current_color_msg, "bgr8")
                img_depth = bridge.imgmsg_to_cv2(self.current_depth_msg, "8UC1")

                bed_id = self.current_bed_view[0]
                bed_side = self.current_bed_view[1]
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

                img_data_msg = BedImageData()
                img_data_msg.bed_id = bed_id
                img_data_msg.bed_side = bed_side
                img_data_msg.enaugh_data = (
                    self.bed_images[(bed_id, bed_side)] >= self.max_images
                )
                img_data_msg.img_color = self.current_color_msg
                img_data_msg.img_depth = self.current_depth_msg
                self.pub_bed_image_data.publish(img_data_msg)

                rospy.loginfo(
                    f"[Photo Logger] ({bed_id}, {bed_side}) Data saved: "
                    f"{self.bed_images[(bed_id, bed_side)]}/{self.max_images}"
                )
            self.rate.sleep()


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
