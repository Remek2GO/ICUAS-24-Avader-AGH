#!/usr/bin/python
"""Script to start the main node of the package."""

from cv_bridge import CvBridge

import cv2
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, NavSatFix, Image, Imu, PointCloud2
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, euler_matrix

TOPIC_CAMERA = "/camera/color/image_raw/compressed"
TOPIC_FRUIT_DETECTIONS = "/fruit_detections"
TOPIC_GPS = "/hawkblue/mavros/global_position/global"
TOPIC_IMU = "/hawkblue/mavros/imu/data"
TOPIC_LIDAR = "/velodyne_points"
TOPIC_ROTATED_LIDAR = "/rotated_lidar"


class MainNode:
    """Class for the main node of the package."""

    def __init__(self, frequency: float):
        """Initialize the main node."""
        self._camera_image: np.ndarray = None
        self._cv_bridge = CvBridge()
        self._current_pose: np.ndarray = None
        self._gps_data: NavSatFix = None
        self._imu_data: Imu = None
        self._initial_rpy: np.ndarray = None
        self._lidar_header: Header = None
        self._lidar_intensity: float = None
        self._lidar_points: np.ndarray = None
        self._lidar_ring: int = None
        self._rate = rospy.Rate(frequency)

        # ROS publishers
        self._pub_camera = rospy.Publisher(TOPIC_FRUIT_DETECTIONS, Image, queue_size=1)
        self._pub_rotated_lidar = rospy.Publisher(
            TOPIC_ROTATED_LIDAR, PointCloud2, queue_size=1
        )

        # ROS subscribers
        rospy.Subscriber(TOPIC_CAMERA, CompressedImage, self._clb_camera)
        rospy.Subscriber(TOPIC_GPS, NavSatFix, self._clb_gps)
        rospy.Subscriber(TOPIC_IMU, Imu, self._clb_imu)
        rospy.Subscriber(TOPIC_LIDAR, PointCloud2, self._clb_lidar)

    def _clb_camera(self, msg: CompressedImage):
        """Process the camera image."""
        distorted_img = self._cv_bridge.compressed_imgmsg_to_cv2(msg)
        # self._camera_image = self._cv_bridge.compressed_imgmsg_to_cv2(msg)

        # Undistort the image
        camera_matrix = np.array(
            [
                [672.0395020303501, 0.0, 642.4371572558833],
                [0.0, 313.0419989351929, 232.20148718757312],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        # dist_coeffs = np.array(
        #     [
        #         -0.732090958418714,
        #         0.8017114744356094,
        #         0.05425622628568444,
        #         0.0011515109968409918,
        #     ],
        #     dtype=np.float32,
        # )
        dist_coeffs = np.array(
            [0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (640, 480), 1, (640, 480)
        )
        self._camera_image = cv2.undistort(
            distorted_img, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        # self.publish_fruit_detections()

    def _clb_gps(self, msg: NavSatFix):
        """Process the GPS data."""
        self._gps_data = msg

    def rad2degree(self, radian):
        return radian * 180 / np.pi

    def _clb_imu(self, msg: Imu):
        """Process the IMU data."""
        self._imu_data = msg

        # Rotate the LiDAR data to the camera/ IMU frame
        lidar_pose = euler_matrix(np.pi,  np.pi / 18, 0)[:3, :3]
        if self._initial_rpy is None:
            self._initial_rpy = euler_from_quaternion(
                [
                    self._imu_data.orientation.x,
                    self._imu_data.orientation.y,
                    self._imu_data.orientation.z,
                    self._imu_data.orientation.w,
                ]
            )

            rospy.loginfo(
                f"Initial Roll: {self._initial_rpy[0]:.2f}, Pitch: {self._initial_rpy[1]:.2f}, Yaw: {self._initial_rpy[2]:.2f}"
            )
            rospy.loginfo(
                f"Initial Roll: {self.rad2degree(self._initial_rpy[0]):.2f}, Pitch: {self.rad2degree(self._initial_rpy[1]):.2f}, Yaw: {self.rad2degree(self._initial_rpy[2]):.2f}"
            )

        current_rpy = euler_from_quaternion(
            [
                self._imu_data.orientation.x,
                self._imu_data.orientation.y,
                self._imu_data.orientation.z,
                self._imu_data.orientation.w,
            ]
        )
        rpy_diff = np.array(current_rpy) - np.array(self._initial_rpy)
        rospy.loginfo(
            f"Roll: {rpy_diff[0]:.2f}, Pitch: {rpy_diff[1]:.2f}, Yaw: {rpy_diff[2]:.2f}"
        )
        # self._current_pose = (
        #     euler_matrix(rpy_diff[0], rpy_diff[1], rpy_diff[2])[:3, :3]
        # ) @ lidar_pose
        self._current_pose = lidar_pose

    def _clb_lidar(self, msg: PointCloud2):
        """Process the LiDAR data."""
        # Read and
        gen = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity", "ring")
        )
        points = list(gen)
        self._lidar_header = msg.header
        self._lidar_intensity = points[0][3]
        self._lidar_points = np.dot(self._current_pose, np.array(points)[:, :3].T).T
        self._lidar_ring = points[0][4]

        # self.publish_rotated_lidar()

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

    def publish_rotated_lidar(self):
        """Publish the rotated LiDAR data."""
        if self._lidar_points is not None:
            lidar_cloud = point_cloud2.create_cloud_xyz32(
                self._lidar_header,
                self._lidar_points,
            )
            self._pub_rotated_lidar.publish(lidar_cloud)

    def run_processing(self):
        """Run the processing."""
        rospy.loginfo("Running processing.")

        while not rospy.is_shutdown():
            self._rate.sleep()
            self.publish_fruit_detections()
            self.publish_rotated_lidar()


if __name__ == "__main__":
    rospy.init_node("main_node", log_level=rospy.INFO)
    main_node = MainNode(frequency=50.0)
    main_node.run_processing()
