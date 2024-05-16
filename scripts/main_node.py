#!/usr/bin/python
"""Script to start the main node of the package."""

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, NavSatFix, Image, Imu, PointCloud2

TOPIC_CAMERA = "/camera/color/image_raw/compressed"
TOPIC_FRUIT_DETECTIONS = "/fruit_detections"
TOPIC_GPS = "/hawkblue/mavros/global_position/global"
TOPIC_IMU = "/hawkblue/mavros/imu/data"
TOPIC_LIDAR = "/velodyne_points"


class MainNode:
    """Class for the main node of the package."""

    def __init__(self, frequency: float):
        """Initialize the main node."""
        self._camera_image: np.ndarray = None
        self._cv_bridge = CvBridge()
        self._gps_data: NavSatFix = None
        self._imu_data: Imu = None
        self._lidar_data: PointCloud2 = None
        self._rate = rospy.Rate(frequency)

        # ROS publishers
        self._pub_camera = rospy.Publisher(TOPIC_FRUIT_DETECTIONS, Image, queue_size=1)

        # ROS subscribers
        rospy.Subscriber(TOPIC_CAMERA, CompressedImage, self._clb_camera)
        rospy.Subscriber(TOPIC_GPS, NavSatFix, self._clb_gps)
        rospy.Subscriber(TOPIC_IMU, Imu, self._clb_imu)
        rospy.Subscriber(TOPIC_LIDAR, PointCloud2, self._clb_lidar)

    def _clb_camera(self, msg: CompressedImage):
        """Process the camera image."""
        self._camera_image = self._cv_bridge.compressed_imgmsg_to_cv2(msg)

    def _clb_gps(self, msg: NavSatFix):
        """Process the GPS data."""
        self._gps_data = msg

    def _clb_imu(self, msg: Imu):
        """Process the IMU data."""
        self._imu_data = msg

    def _clb_lidar(self, msg: PointCloud2):
        """Process the LiDAR data."""
        self._lidar_data = msg

    def get_camera_image(self):
        """Get the camera image."""
        if self._camera_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self._camera_image

    def publish_fruit_detections(self):
        """Publish the fruit detections in the image."""
        camera_image = self.get_camera_image()
        msg = self._cv_bridge.cv2_to_imgmsg(camera_image, "bgr8")
        self._pub_camera.publish(msg)

    def run_processing(self):
        """Run the processing."""
        rospy.loginfo("Running processing.")

        while not rospy.is_shutdown():
            self._rate.sleep()
            self.publish_fruit_detections()


if __name__ == "__main__":
    rospy.init_node("main_node", log_level=rospy.INFO)
    main_node = MainNode(frequency=50.0)
    main_node.run_processing()
