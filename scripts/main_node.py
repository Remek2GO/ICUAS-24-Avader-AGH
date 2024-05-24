#!/usr/bin/python
"""Script to start the main node of the package."""

from cv_bridge import CvBridge

import cv2
import numpy as np
import rospy
import math
from collections import deque
from geometry_msgs.msg import PointStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, NavSatFix, Image, Imu, PointCloud2
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, euler_matrix
import cv2
import message_filters

import image_geometry  # for PinholeCameraModel

TOPIC_CAMERA = "/camera/color/image_raw/compressed"
TOPIC_FRUIT_DETECTIONS = "/fruit_detections"
TOPIC_GLOBAL_MAP = "/global_map"
TOPIC_GPS = "/hawkblue/mavros/global_position/global"
TOPIC_GPS_POSITION = "/gps_position"
TOPIC_IMAGE_LIDAR_POINTS = "/image_lidar_points"
TOPIC_IMAGE_LIDAR_MAP = "/image_lidar_map"
TOPIC_IMU = "/hawkblue/mavros/imu/data"
TOPIC_IMU_POSITION = "/imu_position"
TOPIC_LIDAR = "/velodyne_points"
TOPIC_ROTATED_LIDAR = "/rotated_lidar"

EARTH_RADIUS = 6378137.0
E2 = 6.69437999014e-3  # eccentricity, WGS84
G = 9.81  # gravity acceleration, m/s^2
LIDAR_FRAME_WINDOW = 10


class MainNode:
    """Class for the main node of the package."""

    def __init__(self, frequency: float):
        """Initialize the main node."""
        self._camera_image: np.ndarray = None
        self._cv_bridge = CvBridge()
        self._current_pose: np.ndarray = None
        self._current_rpy: np.ndarray = None
        self._global_map: deque = deque([], maxlen=LIDAR_FRAME_WINDOW)
        self._global_map_id: int = 0
        self._gps_data: NavSatFix = None
        self._gps_position: np.ndarray = None
        self._gps_position_initial: np.ndarray = None
        self._gps_position_id: int = 0
        self._image_lidar_points: np.ndarray = None
        self._image_norm = np.zeros((480, 640, 3), dtype=np.uint8)
        self._imu_data: Imu = None
        self._imu_rpy_initial: np.ndarray = None
        self._imu_rpy_current: np.ndarray = None
        self._imu_position: np.ndarray = np.zeros(3)
        self._imu_velocity: np.ndarray = np.zeros(3)
        self._imu_position_id: int = 0
        self._lidar_header: Header = None
        self._lidar_image_points: np.ndarray = None
        self._lidar_image_points_norm: np.ndarray = None
        self._lidar_intensity: float = None
        self._lidar_points: np.ndarray = None
        self._lidar_ring: int = None
        self._lidar_rotated_id: int = 0
        self._rate = rospy.Rate(frequency)

        # Sensors' poses
        self._camera_intrinsics = np.array(
            [
                [598.0, 0.0, 348.0],
                [0.0, 598.0, 261.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._lidar_pose = euler_matrix(np.pi, np.pi / 18, 0)[:3, :3]
        self._lidar_translation = np.array([0.083, 0.0, 0.126])

        # ROS publishers
        self._pub_camera = rospy.Publisher(TOPIC_FRUIT_DETECTIONS, Image, queue_size=1)
        self._pub_global_map = rospy.Publisher(
            TOPIC_GLOBAL_MAP, PointCloud2, queue_size=1
        )
        self._pub_gps_position = rospy.Publisher(
            TOPIC_GPS_POSITION, PointStamped, queue_size=1
        )
        self._pub_imu_position = rospy.Publisher(
            TOPIC_IMU_POSITION, PointStamped, queue_size=1
        )
        self._pub_rotated_lidar = rospy.Publisher(
            TOPIC_ROTATED_LIDAR, PointCloud2, queue_size=1
        )
        self._pub_image_lidar_map = rospy.Publisher(
            TOPIC_IMAGE_LIDAR_MAP, PointCloud2, queue_size=1
        )
        self._pub_image_lidar_points = rospy.Publisher(
            TOPIC_IMAGE_LIDAR_POINTS, Image, queue_size=1
        )
        self._pub_image_lidar = rospy.Publisher(
            "/image_lidar", PointCloud2, queue_size=1
        )
        self._pub_norm_image = rospy.Publisher("/norm_image", Image, queue_size=1)

        # ROS subscribers
        # rospy.Subscriber(TOPIC_CAMERA, CompressedImage, self._clb_camera)
        rospy.Subscriber(TOPIC_GPS, NavSatFix, self._clb_gps)
        rospy.Subscriber(TOPIC_IMU, Imu, self._clb_imu)
        # rospy.Subscriber(TOPIC_LIDAR, PointCloud2, self._clb_lidar)

        # MW: Synchro TEST
        # gps_sub = message_filters.Subscriber(TOPIC_GPS, NavSatFix)
        image_sub = message_filters.Subscriber(TOPIC_CAMERA, CompressedImage)
        lidar_sub = message_filters.Subscriber(TOPIC_LIDAR, PointCloud2)
        # imu_sub = message_filters.Subscriber(TOPIC_IMU, Imu)

        self._time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [lidar_sub, image_sub], 10, 0.1, allow_headerless=False
        )
        self._time_synchronizer.registerCallback(self._clb_sync)

    # MW: Synchro TEST
    def _clb_sync(self, lidar: PointCloud2, img: CompressedImage):
        """Process the synchronized data."""
        # self._clb_gps(gps)
        self._clb_lidar_v2(lidar)
        self._clb_camera(img)

        # rospy.loginfo("Synchronized data received")

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
        if self._gps_position is None:
            self._gps_position_initial = self.get_gps_position()

            # Wait for the valid initial GPS position
            if self._gps_position_initial is None:
                return
        self._gps_position = self.get_gps_position() - self._gps_position_initial

    def _clb_imu(self, msg: Imu):
        """Process the IMU data."""
        # Get position from IMU data
        if self._imu_data is not None:
            dt = (msg.header.stamp - self._imu_data.header.stamp).to_sec()
            self._imu_position += self._imu_velocity * dt + 0.5 * (
                np.array(
                    [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z - G,
                    ]
                )
                * dt
                * dt
            )
            self._imu_velocity += (
                np.array(
                    [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z - G,
                    ]
                )
                * dt
            )

        # Update IMU data
        self._imu_data = msg

        # Rotate the LiDAR data to the camera/ IMU frame
        if self._imu_rpy_initial is None:
            self._imu_rpy_initial = euler_from_quaternion(
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
            # rospy.loginfo(
            #     f"Initial Roll: {self.rad2degree(self._initial_imu_rpy[0]):.2f}, "
            #     f"Pitch: {self.rad2degree(self._initial_imu_rpy[1]):.2f}, "
            #     f"Yaw: {self.rad2degree(self._initial_imu_rpy[2]):.2f}"
            # )

        self._imu_rpy_current = euler_from_quaternion(
            [
                self._imu_data.orientation.x,
                self._imu_data.orientation.y,
                self._imu_data.orientation.z,
                self._imu_data.orientation.w,
            ]
        )
        rospy.loginfo(
            f"Roll: {self.rad2degree(self._imu_rpy_current[0]):.2f}, "
            f"Pitch: {self.rad2degree(self._imu_rpy_current[1]):.2f}, "
            f"Yaw: {self.rad2degree(self._imu_rpy_current[2]):.2f}"
        )

        self._current_rpy = np.array(self._imu_rpy_current) - np.array(
            self._imu_rpy_initial
        )
        # rospy.loginfo(
        #     f"Roll: {self._current_rpy[0]:.2f}, "
        #     f"Pitch: {self._current_rpy[1]:.2f}, "
        #     f"Yaw: {self._current_rpy[2]:.2f}"
        # )

    def _clb_lidar_v2(self, msg: PointCloud2):
        """Process the LiDAR data."""
        # Read and
        gen = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity", "ring")
        )
        points = list(gen)
        self._lidar_header = msg.header
        self._lidar_intensity = points[0][3]
        self._lidar_points = np.dot(self._lidar_pose, np.array(points)[:, :3].T).T + np.dot(self._lidar_pose, self._lidar_translation)
        self._lidar_ring = points[0][4]

        # Update global map
        if self._current_rpy is not None:
            rot_matrix = euler_matrix(
                self._current_rpy[0], self._current_rpy[1], self._current_rpy[2]
            )[:3, :3]
            self._global_map = np.dot(rot_matrix, self._lidar_points.T).T


        # Filter the lidar data in X-axis 
        # self._lidar_points --> [x, y, z]  

        points_lidar = np.array(points)[:, :3]
        # Filter the lidar data in X-axis
        front_lidar_data = self._lidar_points[
            np.logical_and(self._lidar_points[:, 0] > 0, self._lidar_points[:, 0] < np.inf)
        ]
        # Filter the lidar data in Y-axis
        front_lidar_data = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -15, front_lidar_data[:, 1] < 15)
        ]

        # calculate distance at point y = 0 and z = 0 (in the lidar frame)
        distance_x = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -0.1, front_lidar_data[:, 1] < 0.1)
        ][:, 0]

        # calculate the distance of the lidar data and
        # calculate half size of image
        distance_lidar = np.mean(distance_x) / 2
        image_lidar_data = front_lidar_data[
            np.logical_and(
                front_lidar_data[:, 1] < distance_lidar,
                front_lidar_data[:, 1] > -1 * distance_lidar,
            )
        ]
        self._lidar_image_points  = image_lidar_data

        # print(np.min(image_lidar_data[:, 0]), np.max(image_lidar_data[:, 0]))
        # print(np.min(image_lidar_data[:, 1]), np.max(image_lidar_data[:, 1]))
        # print(np.min(image_lidar_data[:, 2]), np.max(image_lidar_data[:, 2]))
        
        height, width, channel = self.get_camera_image().shape
        # image = np.ones((height, width, channel), dtype=np.uint8) * 255

        # only above the ground
        # image_lidar_data = image_lidar_data[image_lidar_data[:, 2] < 0]

        print(image_lidar_data.shape)

        lidar_width_scale = (width - 1) / (
            np.max(image_lidar_data[:, 1]) - np.min(image_lidar_data[:, 1])
        )
        lidar_height_scale = (height//2 - 1) / (
            np.max(image_lidar_data[:, 2]) - np.min(image_lidar_data[:, 2])
        )

        # image_lidar_data_norm = np.array(
        #     [   image_lidar_data[:, 0],
        #         (image_lidar_data[:, 1])
        #         * lidar_width_scale,
        #         (image_lidar_data[:, 2])
        #         * lidar_height_scale,
        #     ]
        # ).T

        # how to use project lidar point to image
        image_lidar_data_norm = np.array(
            [
                (image_lidar_data[:, 1] - np.min(image_lidar_data[:, 1]))
                * lidar_width_scale,
                (image_lidar_data[:, 2] - np.min(image_lidar_data[:, 2]))
                * lidar_height_scale,
            ]
        ).T

        # display on the image
        
        self._image_norm = self.get_camera_image().copy()

        
        K = np.array([[672.0395020303501, 0.0, 642.4371572558833],
                        [0.0, 313.0419989351929, 232.20148718757312],
                        [0.0, 0.0, 1.0]])
        



        # for i in range(image_lidar_data_norm.shape[0]):
        #     x, y = int(image_lidar_data_norm[i, 0]), int(image_lidar_data_norm[i, 1])
        #     image[y, x] = [0, 0, 0]

        # msg = self._cv_bridge.cv2_to_imgmsg(image, "bgr8")
        # self._pub_norm_image.publish(msg)



        # self._lidar_image_points_norm = image_lidar_data_norm

        # rospy.loginfo(f"Number of points: {lidar_width_scale} {lidar_height_scale}")
        # rospy.loginfo(f"Shape: {image_lidar_data_norm.shape}")

        # print(np.min(image_lidar_data_norm[:, 0]), np.max(image_lidar_data_norm[:, 0]))
        # print(np.min(image_lidar_data_norm[:, 1]), np.max(image_lidar_data_norm[:, 1]))
        # print(np.min(image_lidar_data_norm[:, 2]), np.max(image_lidar_data_norm[:, 2]))

        # self.publish_rotated_lidar()

    def get_camera_image(self):
        """Get the camera image."""
        if self._camera_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self._camera_image

    def get_fruit_localization(self):
        """Get localization of fruit based on image and lidar."""
        fruit_center = [-7, 15, -2]

        gps_position = self.get_gps_position()
        if self._imu_data is not None and gps_position is not None:
            rotation_euler = euler_from_quaternion(
                [
                    self._imu_data.orientation.x,
                    self._imu_data.orientation.y,
                    self._imu_data.orientation.z,
                    self._imu_data.orientation.w,
                ]
            )

            yaw_offset = 1.1868
            rot_matrix = euler_matrix(
                rotation_euler[0], rotation_euler[1] + yaw_offset, rotation_euler[2]
            )[:3, :3]

            fruit_localization = np.dot(rot_matrix, fruit_center) + gps_position

            return fruit_localization

    def get_gps_position(self) -> np.ndarray:
        """Get the GPS position in meters.

        Returns:
            np.ndarray: The GPS position in meters.
        """
        if self._gps_data is None:
            return None
        lat = math.radians(self._gps_data.latitude)
        lon = math.radians(self._gps_data.longitude)
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        cos_lon = math.cos(lon)
        sin_lon = math.sin(lon)

        rn = EARTH_RADIUS / math.sqrt(1 - E2 * sin_lat * sin_lat)
        x = (rn + self._gps_data.altitude) * cos_lat * cos_lon
        y = (rn + self._gps_data.altitude) * cos_lat * sin_lon
        z = self._gps_data.altitude

        # Convert to IMU frame
        x_imu = y
        y_imu = -x
        z_imu = z

        return np.array([x_imu, y_imu, z_imu])

    def publish_fruit_detections(self):
        """Publish the fruit detections in the image."""
        camera_image = self.get_camera_image()
        msg = self._cv_bridge.cv2_to_imgmsg(camera_image, "bgr8")
        self._pub_camera.publish(msg)

    def publish_global_map(self):
        """Publish the global map."""
        if len(self._global_map) > 0:
            # Convert glonal map deque to PointCloud2
            all_points = np.concatenate(self._global_map, axis=0)

            # Create message
            msg_header = Header()
            msg_header.stamp = rospy.Time.now()
            msg_header.frame_id = "hawkblue/imu_link"
            msg_header.seq = self._global_map_id
            global_map = point_cloud2.create_cloud_xyz32(
                msg_header,
                all_points,
            )
            self._pub_global_map.publish(global_map)
            self._global_map_id += 1

    def publish_gps_position(self):
        """Publish the GPS position."""
        if self._gps_position is not None:
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "hawkblue/imu_link"
            msg.header.seq = self._gps_position_id
            msg.point.x = self._gps_position[0]
            msg.point.y = self._gps_position[1]
            msg.point.z = self._gps_position[2]
            self._pub_gps_position.publish(msg)
            self._gps_position_id += 1

    def publish_image_lidar_points(self):
        """Publish the Image LiDAR data."""
        if self._image_lidar_points is not None:
            msg = self._cv_bridge.cv2_to_imgmsg(self._image_lidar_points, "bgr8")
            self._pub_image_lidar_points.publish(msg)

    def publish_imu_position(self):
        """Publish the IMU position."""
        if self._imu_position is not None:
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "hawkblue/imu_link"
            msg.header.seq = self._imu_position_id
            msg.point.x = self._imu_position[0]
            msg.point.y = self._imu_position[1]
            msg.point.z = self._imu_position[2]
            self._pub_imu_position.publish(msg)
            self._imu_position_id += 1

    def publish_rotated_lidar(self):
        """Publish the rotated LiDAR data."""
        if self._lidar_points is not None:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "hawkblue/imu_link"
            header.seq = self._lidar_rotated_id
            lidar_cloud = point_cloud2.create_cloud_xyz32(
                header,
                self._lidar_points,
            )
            self._pub_rotated_lidar.publish(lidar_cloud)
            self._lidar_rotated_id += 1

    def publish_image_lidar_map(self):
        """Publish the Image LiDAR data."""
        if self._lidar_image_points is not None:
            lidar_cloud = point_cloud2.create_cloud_xyz32(
                self._lidar_header,
                self._lidar_image_points,
            )
            self._pub_image_lidar_map.publish(lidar_cloud)

    def publish_image_lidar(self):
        """Publish the Image LiDAR data."""
        if self._lidar_image_points_norm is not None:
            lidar_cloud = point_cloud2.create_cloud_xyz32(
                self._lidar_header,
                self._lidar_image_points_norm,
            )
            self._pub_image_lidar.publish(lidar_cloud)

    def publish_norm_image(self):
        """Publish the Image LiDAR data."""
        if self._image_norm is not None:
            msg = self._cv_bridge.cv2_to_imgmsg(self._image_norm, "bgr8")
            self._pub_norm_image.publish(msg)

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
            # self.publish_fruit_detections()
            # self.publish_rotated_lidar()
            self.publish_global_map()
            # self.get_fruit_localization()
            # self.publish_image_lidar_map()
            # self.publish_norm_image()
            self.publish_image_lidar_points()


if __name__ == "__main__":
    rospy.init_node("main_node", log_level=rospy.INFO)
    main_node = MainNode(frequency=50.0)
    main_node.run_processing()
