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
from std_msgs.msg import Header, Int32
from tf.transformations import euler_from_quaternion, euler_matrix
import message_filters
from typing import List, Tuple

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from scripts.ImageProcessing.Tracking_new import AnalyzeFrame, ObjectParameters

TOPIC_CAMERA = "/camera/color/image_raw/compressed"
TOPIC_FRUIT_COUNT_RED = "/red_fruit_count"
TOPIC_FRUIT_COUNT_YELLOW = "/yellow_fruit_count"
TOPIC_FRUIT_DETECTIONS = "/fruit_detections"
TOPIC_GLOBAL_MAP = "/global_map"
TOPIC_GPS = "/hawkblue/mavros/global_position/global"
TOPIC_GPS_POSITION = "/gps_position"
TOPIC_IMAGE_LIDAR_POINTS = "/image_lidar_points"
TOPIC_IMAGE_LIDAR_POINTS_V2 = "/image_lidar_points_v2"
TOPIC_IMU = "/hawkblue/mavros/imu/data"
TOPIC_IMU_POSITION = "/imu_position"
TOPIC_LIDAR = "/velodyne_points"

EARTH_RADIUS = 6378137.0
E2 = 6.69437999014e-3  # eccentricity, WGS84
G = 9.81  # gravity acceleration, m/s^2
LIDAR_FRAME_WINDOW = 10
N_CLOSEST_POINTS = 5
NEW_FRUIT_PROXIMITY_THRESHOLD = 0.1  # meters


class MainNode:
    """Class for the main node of the package."""

    def __init__(self, frequency: float):
        """Initialize the main node."""
        self._camera_image: np.ndarray = None
        self._cv_bridge = CvBridge()
        self._current_pose: np.ndarray = None
        self._current_rpy: np.ndarray = None
        self._fruits_xyz: np.ndarray = None
        self._fruits_red_count: int = 0
        self._fruits_yellow_count: int = 0
        self._global_map: deque = deque([], maxlen=LIDAR_FRAME_WINDOW)
        self._global_map_id: int = 0
        self._gps_data: NavSatFix = None
        self._gps_position: np.ndarray = None
        self._gps_position_initial: np.ndarray = None
        self._gps_position_id: int = 0
        self._image_lidar_points: np.ndarray = None
        self._image_lidar_points_v2: np.ndarray = None
        self._imu_data: Imu = None
        self._imu_rpy_initial: np.ndarray = None
        self._imu_rpy_current: np.ndarray = None
        self._imu_position: np.ndarray = np.zeros(3)
        self._imu_velocity: np.ndarray = np.zeros(3)
        self._imu_position_id: int = 0
        self._lidar_header: Header = None
        self._lidar_intensity: float = None
        self._lidar_points: np.ndarray = None
        self._lidar_points_distances: np.ndarray = None
        self._lidar_ring: int = None
        self._lidar_rotated_id: int = 0
        self._points_on_2d: np.ndarray = None
        self._rate = rospy.Rate(frequency)

        self._analyzer = AnalyzeFrame()

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
        self._lidar_in_imu_coords = np.array([0.067, 0.0, -0.246])
        self._camera_in_imu_coords = np.array([0.15, 0.0, -0.12])

        # ROS publishers
        self._pub_camera = rospy.Publisher(TOPIC_FRUIT_DETECTIONS, Image, queue_size=1)
        self._pub_red_fruit_count = rospy.Publisher(
            TOPIC_FRUIT_COUNT_RED, Int32, queue_size=1
        )
        self._pub_yellow_fruit_count = rospy.Publisher(
            TOPIC_FRUIT_COUNT_YELLOW, Int32, queue_size=1
        )
        self._pub_global_map = rospy.Publisher(
            TOPIC_GLOBAL_MAP, PointCloud2, queue_size=1
        )
        self._pub_gps_position = rospy.Publisher(
            TOPIC_GPS_POSITION, PointStamped, queue_size=1
        )
        self._pub_imu_position = rospy.Publisher(
            TOPIC_IMU_POSITION, PointStamped, queue_size=1
        )
        self._pub_image_lidar_points = rospy.Publisher(
            TOPIC_IMAGE_LIDAR_POINTS, Image, queue_size=1
        )
        self._pub_image_lidar_points_v2 = rospy.Publisher(
            TOPIC_IMAGE_LIDAR_POINTS_V2, Image, queue_size=1
        )

        # ROS subscribers
        # rospy.Subscriber(TOPIC_CAMERA, CompressedImage, self._clb_camera)
        # rospy.Subscriber(TOPIC_GPS, NavSatFix, self._clb_gps)
        # rospy.Subscriber(TOPIC_IMU, Imu, self._clb_imu)
        # rospy.Subscriber(TOPIC_LIDAR, PointCloud2, self._clb_lidar)

        # MW: Synchro TEST
        gps_sub = message_filters.Subscriber(TOPIC_GPS, NavSatFix)
        image_sub = message_filters.Subscriber(TOPIC_CAMERA, CompressedImage)
        lidar_sub = message_filters.Subscriber(TOPIC_LIDAR, PointCloud2)
        imu_sub = message_filters.Subscriber(TOPIC_IMU, Imu)

        self._time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [lidar_sub, image_sub, imu_sub, gps_sub], 10, 0.1, allow_headerless=False
        )
        self._time_synchronizer.registerCallback(self._clb_sync)

    # MW: Synchro TEST
    def _clb_sync(
        self, lidar: PointCloud2, img: CompressedImage, imu: Imu, gps: NavSatFix
    ):
        """Process the synchronized data."""
        self._clb_gps(gps)
        self._clb_imu(imu)
        self._clb_lidar_v2(lidar)
        # self._clb_lidar(lidar)
        self._clb_camera(img)
        rospy.loginfo("Synchronized data received")

    def _clb_camera(self, msg: CompressedImage):
        """Process the camera image."""
        # distorted_img = self._cv_bridge.compressed_imgmsg_to_cv2(msg)
        self._camera_image = self._cv_bridge.compressed_imgmsg_to_cv2(msg)

        self.publish_fruit_detections()
        self.publish_fruit_red_count()
        self.publish_fruit_yellow_count()

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
        # # Get position from IMU data
        # if self._imu_data is not None:
        #     dt = (msg.header.stamp - self._imu_data.header.stamp).to_sec()
        #     self._imu_position += self._imu_velocity * dt + 0.5 * (
        #         np.array(
        #             [
        #                 msg.linear_acceleration.x,
        #                 msg.linear_acceleration.y,
        #                 msg.linear_acceleration.z - G,
        #             ]
        #         )
        #         * dt
        #         * dt
        #     )
        #     self._imu_velocity += (
        #         np.array(
        #             [
        #                 msg.linear_acceleration.x,
        #                 msg.linear_acceleration.y,
        #                 msg.linear_acceleration.z - G,
        #             ]
        #         )
        #         * dt
        # )

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
        # rospy.loginfo(
        #     f"Roll: {self.rad2degree(self._imu_rpy_current[0]):.2f}, "
        #     f"Pitch: {self.rad2degree(self._imu_rpy_current[1]):.2f}, "
        #     f"Yaw: {self.rad2degree(self._imu_rpy_current[2]):.2f}"
        # )

        self._current_rpy = np.array(self._imu_rpy_current) - np.array(
            self._imu_rpy_initial
        )
        # rospy.loginfo(
        #     f"Roll: {self._current_rpy[0]:.2f}, "
        #     f"Pitch: {self._current_rpy[1]:.2f}, "
        #     f"Yaw: {self._current_rpy[2]:.2f}"
        # )

    def _clb_lidar(self, msg: PointCloud2):
        """Process the LiDAR data."""
        # Read and
        gen = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity", "ring")
        )
        points = list(gen)
        self._lidar_header = msg.header
        self._lidar_intensity = points[0][3]
        self._lidar_ring = points[0][4]

        # Update global map
        # if self._current_rpy is not None:
        #     rot_matrix = euler_matrix(
        #         self._current_rpy[0], self._current_rpy[1], self._current_rpy[2]
        #     )[:3, :3]
        #     self._global_map = np.dot(rot_matrix, self._lidar_points.T).T

        # Filter the lidar data in X-axis
        # self._lidar_points --> [x, y, z]
        points_lidar = np.array(points)[:, :3]
        # Filter the lidar data in X-axis
        front_lidar_data = points_lidar[
            np.logical_and(points_lidar[:, 0] > 0, points_lidar[:, 0] < np.inf)
        ]
        # Filter the lidar data in Y-axis
        points_lidar = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -15, front_lidar_data[:, 1] < 15)
        ]
        angle_x = np.arctan2(points_lidar[:, 1], points_lidar[:, 0])

        # calculate angle point from center of lidar and filter them
        points_lidar = points_lidar[
            np.logical_and(
                angle_x > np.radians(-33),
                angle_x < np.radians(33),
            )
        ]
        angle_y = np.arctan2(points_lidar[:, 2], points_lidar[:, 0])
        points_lidar = points_lidar[
            np.logical_and(
                angle_y > np.radians(-45),
                angle_y < np.radians(10),
            )
        ]

        K = np.array(
            [
                [556.451448389196, 0.0, 319.297706499726],
                [0.0, 555.404890964252, 223.046111850854],
                [0.0, 0.0, 1.0],
            ]
        )
        # K = self._camera_intrinsics

        D = np.array(
            [
                0.067875242949499,
                -0.160971333363663099,
                -0.008792577462950867,
                -0.000882174321070191,
                0.0,
            ]
        )

        X, Y, Z = -0.435, -0.054, 0.652  # -0.054
        ROLL, PITCH, YAW = 126.0, -1.0, -127.0

        tvec = np.array([X, Y, Z])
        rvec = np.array([np.radians(ROLL), np.radians(PITCH), np.radians(YAW)])

        image_cv2, _ = cv2.projectPoints(points_lidar, rvec, tvec, K, D)
        image_cv2 = np.squeeze(image_cv2)

        visible = np.logical_and.reduce(
            (
                image_cv2[:, 0] > 0,
                image_cv2[:, 0] < 480,
                image_cv2[:, 1] > 0,
                image_cv2[:, 1] < 640,
            )
        )

        image_cv2 = image_cv2[visible]

        image = self.get_camera_image().copy()
        for point in image_cv2[:]:
            image = cv2.circle(
                image, (int(point[1]), int(point[0])), 1, (0, 0, 255), -1
            )

        self._image_lidar_points = image

    def _clb_lidar_v2(self, msg: PointCloud2):
        """Process the LiDAR data."""
        # Read lidar point cloud
        gen = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity", "ring")
        )
        points = list(gen)
        # self._lidar_header = msg.header
        # self._lidar_intensity = points[0][3]
        # self._lidar_points = (
        #     np.dot(self._lidar_pose, np.array(points)[:, :3].T).T
        #     + self._lidar_in_imu_coords.T
        # )
        # self._lidar_ring = points[0][4]

        # if self._imu_rpy_current is not None and self._gps_position is not None:
        #     rot_matrix = euler_matrix(
        #         self._imu_rpy_current[0],
        #         self._imu_rpy_current[1],
        #         self._imu_rpy_current[2],
        #     )[:3, :3]
        #     local_map = np.dot(rot_matrix, self._lidar_points.T).T
        #     local_map += self._gps_position.T
        #     self._global_map.append(local_map)

        # Transform the lidar data to the camera frame
        lidar_rotation = euler_matrix(0.0, -np.pi / 18, 0.0)[:3, :3]
        lidar_translation = np.array([-0.083, 0.0, 0.126])
        points_lidar_in_camera_coords = (
            np.dot(lidar_rotation, np.array(points)[:, :3].T).T + lidar_translation.T
        )

        # Filter the lidar data in X-axis
        front_lidar_data = points_lidar_in_camera_coords[
            np.logical_and(
                points_lidar_in_camera_coords[:, 0] > 0,
                points_lidar_in_camera_coords[:, 0] < np.inf,
            )
        ]
        # Filter the lidar data in Y-axis
        front_lidar_data = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -20, front_lidar_data[:, 1] < 20)
        ]

        # calculate distance at point y = 0 and z = 0 (in the lidar frame)
        distance_x = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -0.1, front_lidar_data[:, 1] < 0.1)
        ][:, 0]

        # calculate the distance of the lidar data and
        # calculate half size of image
        mean_distance_lidar = np.mean(distance_x)

        # print(f"Mean distance: {mean_distance_lidar}")
        distance_lidar = mean_distance_lidar / 0.5
        image_lidar_data = front_lidar_data[
            np.logical_and(
                front_lidar_data[:, 1] < distance_lidar,
                front_lidar_data[:, 1] > -1 * distance_lidar,
            )
        ]

        lidar_manual = image_lidar_data

        # Save lidar distances
        self._lidar_points_distances = lidar_manual[:, 0]

        # Rotate lidar to camera plane
        # x_plane = y_lidar
        # y_plane = z_lidar
        # z_plane = x_lidar
        camera_plane_points = np.zeros((lidar_manual.shape[0], 3))
        camera_plane_points[:, 0] = lidar_manual[:, 1]
        camera_plane_points[:, 1] = lidar_manual[:, 2]
        camera_plane_points[:, 2] = lidar_manual[:, 0]

        # Normalize camera plane points
        camera_plane_points[:, 0] /= camera_plane_points[:, 2]
        camera_plane_points[:, 1] /= camera_plane_points[:, 2]
        camera_plane_points[:, 2] /= camera_plane_points[:, 2]

        # Project the points to the camera image
        self._points_on_2d = np.dot(self._camera_intrinsics, camera_plane_points.T).T

        # Lidar points in the camera image
        # camera_img = self.get_camera_image().copy()
        # for point in self._points_on_2d[:]:
        #     # Avoid overflow errors
        #     try:
        #         camera_img = cv2.circle(
        #             camera_img,
        #             (int(point[0]), int(point[1])),
        #             1,
        #             (255, 0, 0),
        #             -1,
        #         )
        #     except Exception as e:
        #         print(e)
        #         pass
        # self._image_lidar_points_v2 = camera_img

    def get_camera_image(self):
        """Get the camera image."""
        if self._camera_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self._camera_image

    def get_image_detection(self):
        """Get the image detection."""
        image = self.get_camera_image()

        red_objects: List[ObjectParameters] = self._analyzer.detect_red(image)
        red_count, img_red = self.get_fruit_localization(red_objects, image)
        self._fruits_red_count += red_count

        yellow_objects: List[ObjectParameters] = self._analyzer.detect_yellow(image)
        yellow_count, img_yellow = self.get_fruit_localization(yellow_objects, img_red)
        self._fruits_yellow_count += yellow_count

        return img_yellow

    def get_fruit_localization(
        self, image_objects: List[ObjectParameters], image: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """Update the internal fruits' 3D positions.

        Args:
            image_objects (List[ObjectParameters]): The detected objects in the image.
            image (np.ndarray): The camera image.

        Returns:
            Tuple[int, np.ndarray]: The number of new detections and the fruits marked \
            in the image.
        """
        n_new_detections = 0
        # Check if necessary data is available
        if self._imu_rpy_current is not None and self._gps_position is not None:
            obj: ObjectParameters
            for obj in image_objects:
                fruit_x = float(obj.x)
                fruit_y = float(obj.y)

                # From camera image to camera plane
                camera_plane_point = np.dot(
                    np.linalg.inv(self._camera_intrinsics),
                    np.array([fruit_x, fruit_y, 1.0]),
                )

                # Get closest lidar points to the fruits
                points_on_2d = np.delete(self._points_on_2d, 2, 1)
                distances = np.linalg.norm(
                    points_on_2d - np.array([fruit_x, fruit_y]), axis=1
                )
                closest_indices = list(np.argsort(distances)[:N_CLOSEST_POINTS])

                # Use distance from lidar to scale the points
                avg_distance = np.mean(self._lidar_points_distances[closest_indices])
                camera_plane_point *= avg_distance

                # Rotate points to the camera frame
                # x_frame = z_plane
                # y_frame = -x_plane
                # z_frame = -y_plane
                camera_frame_point = np.zeros(camera_plane_point.shape)
                camera_frame_point[0] = camera_plane_point[2]
                camera_frame_point[1] = -camera_plane_point[0]
                camera_frame_point[2] = -camera_plane_point[1]

                # Transform the points to the IMU frame
                imu_origin_point = camera_frame_point + self._camera_in_imu_coords

                # Rotate the points to the global frame
                rot_matrix = euler_matrix(
                    self._imu_rpy_current[0],
                    self._imu_rpy_current[1],
                    self._imu_rpy_current[2],
                )[:3, :3]
                fruit_in_3d = np.dot(rot_matrix, imu_origin_point) + self._gps_position

                # Update fruit xyz
                if self._fruits_xyz is None:
                    self._fruits_xyz = fruit_in_3d.reshape(1, -1)
                    n_new_detections += 1
                    cv2.rectangle(
                        image,
                        (obj.bbox[0], obj.bbox[1]),
                        (obj.bbox[0] + obj.bbox[2], obj.bbox[1] + obj.bbox[3]),
                        (0, 255, 0),
                        2,
                    )
                    rospy.loginfo(
                        f"First fruit detected at {fruit_in_3d} "
                        f"({len(self._fruits_xyz)})"
                    )
                else:
                    # Check if the fruit is close to an existing one
                    distances = np.linalg.norm(self._fruits_xyz - fruit_in_3d, axis=1)
                    if np.min(distances) > NEW_FRUIT_PROXIMITY_THRESHOLD:
                        self._fruits_xyz = np.vstack((self._fruits_xyz, fruit_in_3d))
                        n_new_detections += 1
                        cv2.rectangle(
                            image,
                            (obj.bbox[0], obj.bbox[1]),
                            (obj.bbox[0] + obj.bbox[2], obj.bbox[1] + obj.bbox[3]),
                            (0, 255, 0),
                            2,
                        )
                        rospy.loginfo(
                            f"New fruit detected at {fruit_in_3d} "
                            f"({len(self._fruits_xyz)})"
                        )

        return n_new_detections, image

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
        # camera_image = self.get_camera_image().copy()
        image_detection = self.get_image_detection()
        msg = self._cv_bridge.cv2_to_imgmsg(image_detection, "bgr8")
        self._pub_camera.publish(msg)

    def publish_fruit_red_count(self):
        """Publish the red fruit count."""
        msg = Int32()
        msg.data = self._fruits_red_count
        self._pub_red_fruit_count.publish(msg)
        rospy.loginfo(f"Red fruit count: {self._fruits_red_count}")

    def publish_fruit_yellow_count(self):
        """Publish the yellow fruit count."""
        msg = Int32()
        msg.data = self._fruits_yellow_count
        self._pub_yellow_fruit_count.publish(msg)
        rospy.loginfo(f"Yellow fruit count: {self._fruits_yellow_count}")

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

    def publish_image_lidar_points_v2(self):
        """Publish the Image LiDAR data."""
        if self._image_lidar_points_v2 is not None:
            msg = self._cv_bridge.cv2_to_imgmsg(self._image_lidar_points_v2, "bgr8")
            self._pub_image_lidar_points_v2.publish(msg)

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

    # def publish_rotated_lidar(self):
    #     """Publish the rotated LiDAR data."""
    #     if self._lidar_points is not None:
    #         header = Header()
    #         header.stamp = rospy.Time.now()
    #         header.frame_id = "hawkblue/imu_link"
    #         header.seq = self._lidar_rotated_id
    #         lidar_cloud = point_cloud2.create_cloud_xyz32(
    #             header,
    #             self._lidar_points,
    #         )
    #         self._pub_rotated_lidar.publish(lidar_cloud)
    #         self._lidar_rotated_id += 1

    # def publish_image_lidar_map(self):
    #     """Publish the Image LiDAR data."""
    #     if self._lidar_image_points is not None:
    #         lidar_cloud = point_cloud2.create_cloud_xyz32(
    #             self._lidar_header,
    #             self._lidar_image_points,
    #         )
    #         self._pub_image_lidar_map.publish(lidar_cloud)

    # def publish_image_lidar(self):
    #     """Publish the Image LiDAR data."""
    #     if self._lidar_image_points_norm is not None:
    #         lidar_cloud = point_cloud2.create_cloud_xyz32(
    #             self._lidar_header,
    #             self._lidar_image_points_norm,
    #         )
    #         self._pub_image_lidar.publish(lidar_cloud)

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
            self.publish_global_map()
            self.publish_gps_position()
            # self.get_fruit_localization()
            # self.publish_image_lidar_points()
            # self.publish_image_lidar_points_v2()


if __name__ == "__main__":
    rospy.init_node("main_node", log_level=rospy.INFO)
    main_node = MainNode(frequency=50.0)
    main_node.run_processing()
