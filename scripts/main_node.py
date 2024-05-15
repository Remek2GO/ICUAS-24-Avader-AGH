#!/usr/bin/python
"""Script to start the main node of the package."""

import rospy
from sensor_msgs.msg import CompressedImage, NavSatFix, Imu, PointCloud2

TOPIC_CAMERA = "/camera/color/image_raw/compressed"
TOPIC_GPS = "/hawkblue/mavros/global_position/global"
TOPIC_IMU = "/hawkblue/mavros/imu/data"
TOPIC_LIDAR = "/velodyne_points"


class MainNode:
    """Class for the main node of the package."""

    def __init__(self):
        """Initialize the main node."""
        # ROS subscribers
        rospy.Subscriber(TOPIC_CAMERA, CompressedImage, self._clb_camera)
        rospy.Subscriber(TOPIC_GPS, NavSatFix, self._clb_gps)
        rospy.Subscriber(TOPIC_IMU, Imu, self._clb_imu)
        rospy.Subscriber(TOPIC_LIDAR, PointCloud2, self._clb_lidar)

    def _clb_camera(self, msg: CompressedImage):
        """Process the camera image."""
        rospy.loginfo("Received camera image.")

    def _clb_gps(self, msg: NavSatFix):
        """Process the GPS data."""
        rospy.loginfo("Received GPS data.")

    def _clb_imu(self, msg: Imu):
        """Process the IMU data."""
        rospy.loginfo("Received IMU data.")

    def _clb_lidar(self, msg: PointCloud2):
        """Process the LiDAR data."""
        rospy.loginfo("Received LiDAR data.")


if __name__ == "__main__":
    rospy.init_node("main_node", log_level=rospy.INFO)
    main_node = MainNode()
    rospy.spin()
