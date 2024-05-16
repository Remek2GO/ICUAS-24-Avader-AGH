import rosbag
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import cv2
import time


# Create an instance of CvBridge
bridge = CvBridge()

path = "/root/sim_ws/src/icuas24_competition/bags/"
# path = "/home/vision/Documents/Repositorium/icuas24_avader/bags/"
video_no = 1

### VIDEO BAG
video_name = f"ICUAS_bag_{video_no}.bag"

### CALIBRATION BAG
video_name = f"ICUAS_calib_bag_{video_no}.bag"


bag = rosbag.Bag(path + video_name)

types, topics = bag.get_type_and_topic_info()

# Print the names of all topics
for topic_name in topics.keys():
    print(topic_name)


lidar_topic = "/velodyne_points"
image_topic = "/camera/color/image_raw/compressed"

it = 0
pcd = o3d.geometry.PointCloud()


height, width, channels = 480, 640, 3

from matplotlib import cm
from scipy.spatial.transform import Rotation as R

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
cv_img = np.zeros((height, width, 3), dtype=np.uint8)
for topic, msg, t in bag.read_messages(topics=[lidar_topic, image_topic]):

    it += 1
    if it < 1500:
        continue

    # ax0.clear()
    # ax1.clear()
    # ax2.clear()
    
    if topic == lidar_topic:
        lidar_data = pc2.read_points(
            msg, field_names=("x", "y", "z", "intensity", "ring"), skip_nans=True
        )

        # Convert the data to a format that can be used with open3d
        lidar_data = np.array(list(lidar_data))

        rotation_matrix = R.from_rotvec([np.pi, 0, np.pi/18]).as_matrix()
        translation_matrix = np.array([0,0,0])#np.array([-0.083, 0.0, -0.126])

        lidar_data[:, :3] = np.dot(rotation_matrix, lidar_data[:, :3].T).T + translation_matrix


        # lidar_data[:, :2] = -1*lidar_data[:, :2]
        # lidar_data = lidar_data[lidar_data[:,2] > 0.5]




        ax0 = plt.subplot(3, 2, 1)
        ax0.scatter(
            lidar_data[:, 0], lidar_data[:, 1], c=lidar_data[:, 3], cmap="viridis", s=1
        )
        ax0.set_aspect("equal")

        front_lidar_data = lidar_data[
            np.logical_and(lidar_data[:, 0] > 0, lidar_data[:, 0] < 20)
        ]
        front_lidar_data = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -15, front_lidar_data[:, 1] < 15)
        ]

        # distance at point y = 0 and z = 0 (in the lidar frame)

        distance_x = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -0.1, front_lidar_data[:, 1] < 0.1)
        ][:, 0]
        distance_lidar = np.mean(distance_x)/2
        image_lidar_data = front_lidar_data[
            np.logical_and(
                front_lidar_data[:, 1] < distance_lidar,
                front_lidar_data[:, 1] > -1 * distance_lidar,
            )
        ]

        ax1 = plt.subplot(3, 2, 2)
        ax1.scatter(
            front_lidar_data[:, 1], front_lidar_data[:, 2], c=front_lidar_data[:, 3], cmap="viridis", s=1
        )
        ax1.set_aspect("equal")

        ax1 = plt.subplot(3, 2, 3)
        ax1.scatter(
            image_lidar_data[:, 1], image_lidar_data[:, 2], c=image_lidar_data[:, 3], cmap="viridis", s=1
        )
        ax1.set_aspect("equal")

        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # only above the ground
        image_lidar_data = image_lidar_data[image_lidar_data[:, 2] > 0]


        lidar_width_scale = (width - 1) / (
            np.max(image_lidar_data[:, 1]) - np.min(image_lidar_data[:, 1])
        )
        lidar_height_scale = (height - 1) / (
            np.max(image_lidar_data[:, 2]) - np.min(image_lidar_data[:, 2])
        )

        image_lidar_data_norm = np.array(
            [
                (image_lidar_data[:, 1] - np.min(image_lidar_data[:, 1]))
                * lidar_width_scale,
                (image_lidar_data[:, 2] - np.min(image_lidar_data[:, 2]))
                * lidar_height_scale,
            ]
        ).T

        value = np.array(
            image_lidar_data[:, 3] / np.max(image_lidar_data[:, 3])  * 255,
            dtype=np.int_,
        )
        # for x, y, data in zip(
        #     image_lidar_data_norm[:, 0],
        #     image_lidar_data_norm[:, 1],
        #     value,
        # ):
        #     # print(y, x, data)
        #     image[int(y), int(x), :] = [128, data, 128]

        ax1 = plt.subplot(3, 2, 4)
        ax1.scatter(
            image_lidar_data_norm[:, 0], image_lidar_data_norm[:, 1], c=value, cmap="viridis", s=1
        )

        ax1 = plt.subplot(3, 2, 5)
        ax1.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        ax1.scatter(
            image_lidar_data_norm[:, 0], max(image_lidar_data_norm[:, 1]) - image_lidar_data_norm[:, 1], c=value, cmap="viridis", s=1
        )

        


    elif topic == image_topic:
        # Convert the ROS Image message to an OpenCV image
        # cv_img = bridge.imgmsg_to_cv2(msg)
        cv_img = bridge.compressed_imgmsg_to_cv2(msg)

        # Now cv_img is an OpenCV image, you can do any processing you need here.
        # For example, display the image
        cv2.imshow("image", cv_img)
        cv2.waitKey(1)

    # plt.show()
    plt.draw()
    plt.pause(0.1)

    print(it)


bag.close()
cv2.destroyAllWindows()
