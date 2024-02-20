#!/usr/bin/env python
"""Node to analyze images and count fruits in the plant beds."""

import cv2
import os
import sys
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from icuas24_competition.msg import BedImageData
from scripts.utils.detect_fruits import process_patch, FRUIT_TYPE_ENCODING
from scripts.utils.inflight_image_analysis import get_patches

bridge = CvBridge()


class PhotoAnalyzer:
    """Class to analyze the photos of the plants."""

    def __init__(self, frequency: float):
        """Initialize the PhotoAnalyzer class.

        Args:
            frequency (float): The frequency of the node.
        """
        self.rate = rospy.Rate(frequency)
        self.bed_image_data_queue: List[BedImageData] = []
        self.result_image: Image = None

        # ROS publishers and subscribers
        self.pub_output_image = rospy.Publisher(
            "/avader/output_image", Image, queue_size=10
        )

        rospy.Subscriber("/avader/bed_image_data", BedImageData, self._image_data_clb)

    def _image_data_clb(self, msg: BedImageData):
        self.bed_image_data_queue.append(msg)

    def run(self):
        """Run the node."""
        while not rospy.is_shutdown():
            if len(self.bed_image_data_queue) > 0:
                rospy.logdebug("[Photo Analyzer] Processing the image data")
                bed_image_data = self.bed_image_data_queue.pop(0)
                img_color = bridge.imgmsg_to_cv2(bed_image_data.img_color, "bgr8")
                img_depth = bridge.imgmsg_to_cv2(bed_image_data.img_depth, "8UC1")
                # TODO: It is temporary solution to ensure compatibility with the
                # previous code
                odom_data = f"{bed_image_data.odom_data.x} "
                odom_data += f"{bed_image_data.odom_data.y} "
                odom_data += f"{bed_image_data.odom_data.z} "
                odom_data += f"{bed_image_data.odom_data.roll} "
                odom_data += f"{bed_image_data.odom_data.pitch} "
                odom_data += f"{bed_image_data.odom_data.yaw}"
                patches, patches_coords, img_rotated = get_patches(
                    img_color, img_depth, odom_data
                )
                for patch, patch_coords in zip(patches, patches_coords):
                    fruit_count, fruit_type, fruit_centre = process_patch(patch)
                    cv2.rectangle(
                        img_rotated,
                        (patch_coords[2], patch_coords[0]),
                        (patch_coords[3], patch_coords[1]),
                        (255, 0, 0),
                        2,
                    )
                self.result_image = bridge.cv2_to_imgmsg(img_rotated, "bgr8")

            # Publish the result image
            if self.result_image is not None:
                self.pub_output_image.publish(self.result_image)

            self.rate.sleep()


if __name__ == "__main__":
    myargs = rospy.myargv(argv=sys.argv)
    if len(myargs) < 2:
        print("Usage: photo_analyzer.py <frequency>")
        sys.exit(1)
    frequency = float(myargs[1])

    rospy.init_node("photo_analyzer", log_level=rospy.DEBUG)
    rospy.loginfo(f"[Photo Analyzer] Node started with params:\n" f"\t{frequency} Hz")

    photo_analyzer = PhotoAnalyzer(frequency)
    photo_analyzer.run()
