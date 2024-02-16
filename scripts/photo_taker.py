#!/usr/bin/env python
"""Photo taker node to take photos of the plants."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Transform
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from typing import Tuple

from icuas24_competition.msg import ImageForAnalysis, UavSetpoint
from scripts.utils.positions import PointOfInterest, POINTS_OF_INTEREST
from scripts.utils.types import Setpoint

bridge = CvBridge()

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images"
PROXIMITY_THRESHOLD = 0.1
YAW_THRESHOLD = np.pi / 180
MAX_IMAGES = 5
FRAMES_TO_SKIP = 10

DEBUG_MODE = False


class PhotoTaker:
    """Class to take photos of the plants."""

    def __init__(self, frequency: float):
        """Initialize the PhotoTaker class.

        Args:
            frequency (float): The frequency of the node.
        """
        self.current_color_path: str = None
        self.current_depth_path: str = None
        self.current_color_msg: Image = None
        self.current_depth_msg: Image = None
        self.current_odom: Odometry = None
        self.take_photo: bool = False
        self.take_photo_setpoint: Setpoint = None

        self.rate = rospy.Rate(frequency)

        # Rospy subscribers and publishers
        # rospy.Subscriber("/red/plants_beds", String, self._plants_beds_clb)
        rospy.Subscriber("/take_photo", UavSetpoint, self._take_photo_clb)
        rospy.Subscriber("/red/camera/depth/image_raw", Image, self._image_depth_clb)
        rospy.Subscriber("/red/camera/color/image_raw", Image, self._image_color_clb)
        rospy.Subscriber("red/odometry", Odometry, self._odom_clb)
        self.pub_image_taken = rospy.Publisher(
            "/image_for_analysis", ImageForAnalysis, queue_size=10
        )
        self.pub_move_on = rospy.Publisher("/move_on", Bool, queue_size=10)
        self.pub_position_hold = rospy.Publisher(
            "/position_hold/trajectory", MultiDOFJointTrajectoryPoint, queue_size=10
        )

    def _image_color_clb(self, msg: Image):
        self.current_color_msg = msg

    def _image_depth_clb(self, msg: Image):
        self.current_depth_msg = msg

    def _odom_clb(self, msg: Odometry):
        self.current_odom = msg

    def _take_photo_clb(self, msg: UavSetpoint):
        # Read setpoint
        self.take_photo_setpoint = Setpoint(
            x=msg.x, y=msg.y, z=msg.z, roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw
        )

        # Set flag to take a photo
        self.take_photo = True

    def is_close_to_position(self) -> Tuple[bool, int, int]:
        """Check if the drone is close enaugh to the point of interest.

        Returns:
            `bool`: `True` if the drone is close enaugh to the point of interest, \
                `False` otherwise.
        """
        odom_position = [
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
        poi: PointOfInterest = [
            self.take_photo_setpoint.x,
            self.take_photo_setpoint.y,
            self.take_photo_setpoint.z,
            0.0,
            0.0,
            self.take_photo_setpoint.yaw,
        ]

        # Get the id of the bed side
        bed_side = -1
        if poi[-1] == 0.0:
            bed_side = 0
        else:
            bed_side = 1

        # Get bed id
        bed_id = -1
        for poi_id, (bed_side_0, bed_side_1) in POINTS_OF_INTEREST.items():
            if bed_side == 0:
                if np.allclose(bed_side_0[:3], poi[:3], rtol=0.0, atol=0.1):
                    bed_id = poi_id
                    break
            else:
                if np.allclose(bed_side_1[:3], poi[:3], rtol=0.0, atol=0.1):
                    bed_id = poi_id
                    break

        distance = np.linalg.norm(np.array(odom_position[:3]) - np.array(poi[:3]))
        yaw_diff = np.abs(odom_position[-1] - poi[-1])

        if DEBUG_MODE:
            rospy.loginfo(
                f"[Photo Taker] ({bed_id}, {bed_side}) Distance: {distance}, \
                    Yaw diff: |{odom_position[-1]} - {poi[-1]}| = {yaw_diff}"
            )
        return (
            distance < PROXIMITY_THRESHOLD
            and (yaw_diff < YAW_THRESHOLD or yaw_diff > 2 * np.pi - YAW_THRESHOLD),
            bed_id,
            bed_side,
        )

    def run(self):
        """Run the photo taker node.

        This method is the main loop of the photo taker node. It checks if the drone \
            is close enaugh to the point of interest and takes a photo if it is. \
            Otherwise, it publishes the position hold setpoint to move the drone \
            closer to the point of interest.
        """
        while not rospy.is_shutdown():
            if self.take_photo:
                close_enaugh, bed_id, bed_side = self.is_close_to_position()
                if bed_id == -1 or bed_side == -1:
                    self.take_photo = False
                    self.pub_move_on.publish(Bool(True))
                else:
                    if close_enaugh:
                        self.take_photo = False
                        self.pub_move_on.publish(Bool(True))

                        # Get images from messages
                        img_color = bridge.imgmsg_to_cv2(self.current_color_msg, "bgr8")
                        img_depth = bridge.imgmsg_to_cv2(self.current_depth_msg, "8UC1")

                        unique_id = f"{bed_id}{bed_side}_manual"
                        path = f"{IMAGES_FOLDER_PATH}/{unique_id}"

                        # if DEBUG_MODE:
                        cv2.imwrite(f"{path}_color.png", img_color)
                        cv2.imwrite(f"{path}_depth.png", img_depth)

                        img_msg = ImageForAnalysis()
                        img_msg.img_path_color = f"{path}_color.png"
                        img_msg.img_path_depth = f"{path}_depth.png"
                        img_msg.bed_id = np.uint8(bed_id)
                        img_msg.bed_side = np.uint8(bed_side)
                        img_msg.img_id = 10
                        img_msg.pose = self.current_odom.pose.pose
                        self.pub_image_taken.publish(img_msg)
                    else:
                        # Publish position hold setpoint
                        transform = Transform()
                        transform.translation.x = self.take_photo_setpoint.x
                        transform.translation.y = self.take_photo_setpoint.y
                        transform.translation.z = self.take_photo_setpoint.z
                        x, y, z, w = quaternion_from_euler(
                            self.take_photo_setpoint.roll,
                            self.take_photo_setpoint.pitch,
                            self.take_photo_setpoint.yaw,
                        )
                        transform.rotation.x = x
                        transform.rotation.y = y
                        transform.rotation.z = z
                        transform.rotation.w = w
                        point = MultiDOFJointTrajectoryPoint()
                        point.transforms = []
                        point.transforms.append(transform)
                        self.pub_position_hold.publish(point)
            self.rate.sleep()


if __name__ == "__main__":
    myargv = rospy.myargv(argv=sys.argv)
    frequency = 50.0
    if len(myargv) < 2:
        rospy.logwarn(
            f"[Photo Taker] Frequency not provided, using default value {frequency} Hz"
        )
    else:
        frequency = float(myargv[1])

    rospy.init_node("photo_taker")
    rospy.loginfo("[Photo Taker] Node started")

    photo_taker = PhotoTaker(frequency)
    photo_taker.run()
