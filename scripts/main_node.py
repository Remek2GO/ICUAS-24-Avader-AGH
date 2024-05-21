#!/usr/bin/python
"""Script to start the main node of the package."""

from cv_bridge import CvBridge

import numpy as np
import rospy
import math
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, NavSatFix, Image, Imu, PointCloud2
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, euler_matrix
from scipy.spatial.transform import Rotation as R

import message_filters

TOPIC_CAMERA = "/camera/color/image_raw/compressed"
TOPIC_FRUIT_DETECTIONS = "/fruit_detections"
TOPIC_GLOBAL_MAP = "/global_map"
TOPIC_GPS = "/hawkblue/mavros/global_position/global"
TOPIC_IMU = "/hawkblue/mavros/imu/data"
TOPIC_LIDAR = "/velodyne_points"
TOPIC_ROTATED_LIDAR = "/rotated_lidar"
EARTH_RADIUS = 6378137.0
e2 = 6.69437999014e-3  # eccentricity, WGS84


class MainNode:
    """Class for the main node of the package."""

    def __init__(self, frequency: float):
        """Initialize the main node."""
        self._camera_image: np.ndarray = None
        self._cv_bridge = CvBridge()
        self._current_rpy: np.ndarray = None
        self._current_pose: np.ndarray = None
        self._global_map: np.ndarray = None
        self._global_map_id: int = 0
        self._gps_data: NavSatFix = None
        self._imu_data: Imu = None
        self._initial_imu_rpy: np.ndarray = None
        self._lidar_header: Header = None
        self._lidar_intensity: float = None
        self._lidar_points: np.ndarray = None
        self._lidar_ring: int = None
        self._rate = rospy.Rate(frequency)

        # Sensors' poses
        self._lidar_pose = euler_matrix(np.pi, np.pi / 18, 0)[:3, :3]

        # ROS publishers
        self._pub_camera = rospy.Publisher(TOPIC_FRUIT_DETECTIONS, Image, queue_size=1)
        self._pub_global_map = rospy.Publisher(
            TOPIC_GLOBAL_MAP, PointCloud2, queue_size=1
        )
        self._pub_rotated_lidar = rospy.Publisher(
            TOPIC_ROTATED_LIDAR, PointCloud2, queue_size=1
        )

        # ROS subscribers
        # rospy.Subscriber(TOPIC_CAMERA, CompressedImage, self._clb_camera)
        rospy.Subscriber(TOPIC_GPS, NavSatFix, self._clb_gps)
        rospy.Subscriber(TOPIC_IMU, Imu, self._clb_imu)
        # rospy.Subscriber(TOPIC_LIDAR, PointCloud2, self._clb_lidar)

        # MW: Synchro TEST
        image_sub = message_filters.Subscriber(TOPIC_CAMERA, CompressedImage)
        lidar_sub = message_filters.Subscriber(TOPIC_LIDAR, PointCloud2)
        # imu_sub = message_filters.Subscriber(TOPIC_IMU, Imu)
        # gps_sub = message_filters.Subscriber(TOPIC_GPS, NavSatFix)

        self._time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [lidar_sub, image_sub], 10, 0.1, allow_headerless=False
        )
        self._time_synchronizer.registerCallback(self._clb_sync)

    # MW: Synchro TEST
    def _clb_sync(self, lidar: PointCloud2, img: CompressedImage):
        """Process the synchronized data."""
        self._clb_lidar(lidar)
        self._clb_camera(img)

    def _clb_camera(self, msg: CompressedImage):
        """Process the camera image."""
        # distorted_img = self._cv_bridge.compressed_imgmsg_to_cv2(msg)
        self._camera_image = self._cv_bridge.compressed_imgmsg_to_cv2(msg)

        # Undistort the image
        # camera_matrix = np.array(
        #     [
        #         [672.0395020303501, 0.0, 642.4371572558833],
        #         [0.0, 313.0419989351929, 232.20148718757312],
        #         [0.0, 0.0, 1.0],
        #     ],
        #     dtype=np.float32,
        # )
        # dist_coeffs = np.array(
        #     [
        #         -0.732090958418714,
        #         0.8017114744356094,
        #         0.05425622628568444,
        #         0.0011515109968409918,
        #     ],
        #     dtype=np.float32,
        # )
        # dist_coeffs = np.array(
        #     [0.0, 0.0, 0.0, 0.0],
        #     dtype=np.float32,
        # )
        # new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        #     camera_matrix, dist_coeffs, (640, 480), 1, (640, 480)
        # )
        # self._camera_image = cv2.undistort(
        #     distorted_img, camera_matrix, dist_coeffs, None, new_camera_matrix
        # )
        # self.publish_fruit_detections()

    def _clb_gps(self, msg: NavSatFix):
        """Process the GPS data."""
        self._gps_data = msg

    def _clb_imu(self, msg: Imu):
        """Process the IMU data."""
        self._imu_data = msg

        # Rotate the LiDAR data to the camera/ IMU frame
        if self._initial_imu_rpy is None:
            self._initial_imu_rpy = euler_from_quaternion(
                [
                    self._imu_data.orientation.x,
                    self._imu_data.orientation.y,
                    self._imu_data.orientation.z,
                    self._imu_data.orientation.w,
                ]
            )

            # rospy.loginfo(
            #     f"Initial Roll: {self._initial_rpy[0]:.2f}, "
            #     f"Pitch: {self._initial_rpy[1]:.2f}, "
            #     f"Yaw: {self._initial_rpy[2]:.2f}"
            # )
            rospy.loginfo(
                f"Initial Roll: {self.rad2degree(self._initial_imu_rpy[0]):.2f}, "
                f"Pitch: {self.rad2degree(self._initial_imu_rpy[1]):.6f}, "
                f"Yaw: {self.rad2degree(self._initial_imu_rpy[2]):.2f}"
            )

        current_imu_rpy = euler_from_quaternion(
            [
                self._imu_data.orientation.x,
                self._imu_data.orientation.y,
                self._imu_data.orientation.z,
                self._imu_data.orientation.w,
            ]
        )
        self._current_rpy = np.array(current_imu_rpy) - np.array(self._initial_imu_rpy)
        # rospy.loginfo(
        #     f"Roll: {self.current_rpy[0]:.2f}, "
        #     f"Pitch: {self.current_rpy[1]:.2f}, "
        #     f"Yaw: {self.current_rpy[2]:.2f}"
        # )
        # rospy.loginfo(
        #     f"Roll: {self.rad2degree(self._current_rpy[0]):.2f}, "
        #     f"Pitch: {self.rad2degree(self._current_rpy[1]):.2f}, "
        #     f"Yaw: {self.rad2degree(self._current_rpy[2]):.2f}"
        # )
        # self._current_pose = euler_matrix(
        #     self.current_rpy[0], self.current_rpy[1], self.current_rpy[2]
        # )[:3, :3] @ lidar_pose
        # self._current_pose = lidar_pose

    def _clb_lidar(self, msg: PointCloud2):
        """Process the LiDAR data."""
        # Read and
        gen = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity", "ring")
        )
        points = list(gen)
        self._lidar_header = msg.header
        self._lidar_intensity = points[0][3]
        self._lidar_points = np.dot(self._lidar_pose, np.array(points)[:, :3].T).T
        self._lidar_ring = points[0][4]

        # Update global map
        if self._current_rpy is not None:
            rot_matrix = euler_matrix(
                self._current_rpy[0], self._current_rpy[1], self._current_rpy[2]
            )[:3, :3]
            self._global_map = np.dot(rot_matrix, self._lidar_points.T).T

        # self.publish_rotated_lidar()

    def get_camera_image(self):
        """Get the camera image."""
        if self._camera_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self._camera_image

    def get_fruit_localization(self):
        """Get localization of fruit based on image and lidar"""
        fruit_center = [-7, 15, -2]

        if self._imu_data is not None:
            rotation_euler = euler_from_quaternion(
                [
                    self._imu_data.orientation.x,
                    self._imu_data.orientation.y,
                    self._imu_data.orientation.z,
                    self._imu_data.orientation.w,
                ]
            )
            lat = math.radians(self._gps_data.latitude)
            lon = math.radians(self._gps_data.longitude)
            sin_lat = math.sin(lat)
            cos_lat = math.cos(lat)
            cos_lon = math.cos(lon)
            sin_lon = math.sin(lon)

            rn = EARTH_RADIUS / math.sqrt(1 - e2 * sin_lat * sin_lat)
            X = (rn + self._gps_data.altitude) * cos_lat * cos_lon
            Y = (rn + self._gps_data.altitude) * cos_lat * sin_lon
            Z = self._gps_data.altitude

            YAW_OFFSET = 1.1868
            gps_position = np.array([X, Y, Z])
            rot_matrix = R.from_rotvec(
                [rotation_euler[0], rotation_euler[1]+YAW_OFFSET, rotation_euler[2]]
            ).as_matrix()

            fruit_localization = np.dot(rot_matrix, fruit_center) + gps_position
            return fruit_localization

    def publish_fruit_detections(self):
        """Publish the fruit detections in the image."""
        camera_image = self.get_camera_image()
        msg = self._cv_bridge.cv2_to_imgmsg(camera_image, "bgr8")
        self._pub_camera.publish(msg)

    def publish_global_map(self):
        """Publish the global map."""
        if self._global_map is not None:
            msg_header = Header()
            msg_header.stamp = rospy.Time.now()
            msg_header.frame_id = "velodyne"
            msg_header.seq = self._global_map_id
            global_map = point_cloud2.create_cloud_xyz32(
                msg_header,
                self._global_map,
            )
            self._pub_global_map.publish(global_map)
            self._global_map_id += 1

    def publish_rotated_lidar(self):
        """Publish the rotated LiDAR data."""
        if self._lidar_points is not None:
            lidar_cloud = point_cloud2.create_cloud_xyz32(
                self._lidar_header,
                self._lidar_points,
            )
            self._pub_rotated_lidar.publish(lidar_cloud)

    def rad2degree(self, radian: float) -> float:
        """Convert radians to degrees.

        Args:
            radian (float): The angle in radians.

        Returns:
            float: The angle in degrees.
        """
        return radian * 180 / np.pi

    def run_processing(self):
        """Run the processing."""
        rospy.loginfo("Running processing.")

        while not rospy.is_shutdown():
            self._rate.sleep()
            self.publish_fruit_detections()
            # self.publish_rotated_lidar()
            self.publish_global_map()
            self.get_fruit_localization()


if __name__ == "__main__":
    rospy.init_node("main_node", log_level=rospy.INFO)
    main_node = MainNode(frequency=50.0)
    main_node.run_processing()
