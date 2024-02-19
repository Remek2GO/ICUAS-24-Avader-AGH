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

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images_eval"
PROXIMITY_THRESHOLD = 0.1   #0.1
YAW_THRESHOLD = np.pi / 180   #pi / 180
MAX_IMAGES = 5
FRAMES_TO_SKIP = 10

DEBUG_MODE = False

class PhotoLogger:
    """Class to log all photos of the plants."""

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
        self.rate = rospy.Rate(frequency)

        # Rospy subscribers and publishers
        # rospy.Subscriber("/red/plants_beds", String, self._plants_beds_clb)
        rospy.Subscriber("/red/camera/depth/image_raw", Image, self._image_depth_clb)
        rospy.Subscriber("/red/camera/color/image_raw", Image, self._image_color_clb)
        rospy.Subscriber("red/odometry", Odometry, self._odom_clb)
   

    def _image_color_clb(self, msg: Image):
        self.current_color_msg = msg

    def _image_depth_clb(self, msg: Image):
        self.current_depth_msg = msg

    def _odom_clb(self, msg: Odometry):
        self.current_odom = msg

   

    def run(self):
        """Run the photo logger node.

        This method is the main loop of the photo logger node. 
        """
        ii = 0
        while not rospy.is_shutdown():
            # Get images from messages
            if ( self.current_color_msg != None):
                img_color = bridge.imgmsg_to_cv2(self.current_color_msg, "bgr8")
                img_depth = bridge.imgmsg_to_cv2(self.current_depth_msg, "8UC1")

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
                

                unique_id = f"{ii}_eval"
                path = f"{IMAGES_FOLDER_PATH}/{unique_id}"

                # if DEBUG_MODE:
                cv2.imwrite(f"{path}_color.png", img_color)
                cv2.imwrite(f"{path}_depth.png", img_depth)

                f = open(f"{path}_odom.txt","w")
                t = str(odom_position[0]) +"  "+str(odom_position[1])+"  " + str(odom_position[2])+"  " + str(odom_position[3])+"  " + str(odom_position[4])+"  " + str(odom_position[5])
                f.write(t)
                print(t)
                
                f.close()
                ii +=1

            self.rate.sleep()




if __name__ == "__main__":
    myargv = rospy.myargv(argv=sys.argv)
    frequency = 50.0
    if len(myargv) < 2:
        rospy.logwarn(
            f"[Photo Logger] Frequency not provided, using default value {frequency} Hz"
        )
    else:
        frequency = float(myargv[1])

    rospy.init_node("photo_logger")
    rospy.loginfo("[Photo Logger] Node started")

    photo_logger = PhotoLogger(frequency)
    photo_logger.run()
