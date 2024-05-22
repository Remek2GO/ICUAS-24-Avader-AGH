import rosbag
import sensor_msgs.point_cloud2 as pc2

# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import cv2
import time
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
import tf
import tf.transformations


# Create an instance of CvBridge
bridge = CvBridge()

path = "/root/sim_ws/src/icuas24_competition/bags/"
# path = "/home/vision/Documents/Repositorium/icuas24_avader/bags/"
video_no = 1

### VIDEO BAG
video_name = f"ICUAS_bag_{video_no}.bag"

### CALIBRATION BAG
video_name = f"ICUAS_calib_bag_{video_no}.bag"


# Read the bag
bag = rosbag.Bag(path + video_name)
types, topics = bag.get_type_and_topic_info()

# Print the names of all topics
for topic_name in topics.keys():
    print(topic_name)


# Topic names
lidar_topic = "/velodyne_points"
image_topic = "/camera/color/image_raw/compressed"
imu_topic = "/hawkblue/mavros/imu/data"

it = 0

#  initialize the figure for 3D plot
# pcd = o3d.geometry.PointCloud()

# Size of the image
height, width, channels = 480, 640, 3


fig2, ax_lidar = plt.subplots(3, 2)


imu_offset = np.array(
    [-0.009820524603128667, -0.002914145588874866, 1.141822294392858]
)  # dla calib1, jesli nie startujemy od pierwszej ramki i dajemy ponizej True
first_frame_imu = True
acc_angles = list()
for topic, msg, t in bag.read_messages(topics=[lidar_topic, image_topic, imu_topic]):

    # how synchronize topics
    # while msg.header.stamp.to_sec() < t.to_sec():
    #     msg = next(bag.read_messages(topics=[lidar_topic, image_topic, imu_topic]))[1]

    it += 1
    # skip the first X messages
    if it < 1800:
        continue

    if topic == imu_topic:
        # print(msg.orientation)
        orientation = msg.orientation

        # from quatertion to euler
        angles = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        # to degrees
        angles_deg = [angle * 180 / np.pi for angle in angles]  # roll, pitch, yaw

        if not first_frame_imu:
            imu_offset = angles
            first_frame_imu = True
            print("First frame imu")
            print(imu_offset)

        acc_angles.append(angles_deg)
        acc_angles_arr = np.array(acc_angles)

        # display angles

        # fig1, ax11 = plt.subplots(3, 1)
        # ax11[0].plot(acc_angles_arr[:,0])
        # ax11[0].set_ylabel('Angle')
        # ax11[1].plot(acc_angles_arr[:,1])
        # ax11[2].plot(acc_angles_arr[:,2])
        # ax11[2].set_xlabel('Time')
        # ax11[2].set_title('Orientation Angles')

        # plt.show()
        # print(rotation_matrix_imu)
        # print(angles)

    elif topic == lidar_topic:
        lidar_data = pc2.read_points(
            msg, field_names=("x", "y", "z", "intensity", "ring"), skip_nans=True
        )

        # Convert the data to a format that can be used with open3d
        lidar_data = np.array(list(lidar_data))

        # Create Rotation matrix and translation matrix to align the lidar data with the camera
        rotation_matrix = R.from_rotvec([np.pi, 0, np.pi / 18]).as_matrix()
        translation_matrix = np.array([-0.083, 0.0, -0.126])

        # Rotate and translate the lidar data
        lidar_data[:, :3] = (
            np.dot(rotation_matrix, lidar_data[:, :3].T).T + translation_matrix
        )

        # Obrót lidaru względem początkowego położenia wyrażone przez obrot imu
        delta_angles = np.array(angles) - imu_offset

        # print(delta_angles)

        # Create Rotation matrix and translation matrix to align the lidar data with the camera
        rotation_matrix_imu = tf.transformations.euler_matrix(
            delta_angles[0], delta_angles[1], delta_angles[2]
        )

        # Rotate and translate the lidar data - using inverse rotation matrix
        lidar_data[:, :3] = np.dot(rotation_matrix_imu[:3, :3].T, lidar_data[:, :3].T).T

        # clear the plot
        ax_lidar[0, 0].clear()
        # ax_lidar[0,1].clear()
        # ax_lidar[1,0].clear()
        # ax_lidar[1,1].clear()
        # ax_lidar[2,0].clear()

        # Plot the lidar data - bird's eye view
        N = -1
        ax_lidar[0, 0].scatter(
            lidar_data[:N, 0],
            lidar_data[:N, 1],
            c=lidar_data[:N, 3],
            cmap="viridis",
            s=1,
        )
        ax_lidar[0, 0].set_aspect("equal")

        # Filter the lidar data
        #
        front_lidar_data = lidar_data[
            np.logical_and(lidar_data[:, 0] > 0, lidar_data[:, 0] < np.inf)
        ]
        front_lidar_data = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -15, front_lidar_data[:, 1] < 15)
        ]

        # distance at point y = 0 and z = 0 (in the lidar frame)

        distance_x = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -0.1, front_lidar_data[:, 1] < 0.1)
        ][:, 0]
        distance_lidar = np.mean(distance_x) / 2
        image_lidar_data = front_lidar_data[
            np.logical_and(
                front_lidar_data[:, 1] < distance_lidar,
                front_lidar_data[:, 1] > -1 * distance_lidar,
            )
        ]

        ax_lidar[1, 0].scatter(
            front_lidar_data[:, 1],
            front_lidar_data[:, 2],
            c=front_lidar_data[:, 3],
            cmap="viridis",
            s=1,
        )
        ax_lidar[1, 0].set_aspect("equal")

        ax_lidar[2, 0].scatter(
            image_lidar_data[:, 1],
            image_lidar_data[:, 2],
            c=image_lidar_data[:, 3],
            cmap="viridis",
            s=1,
        )
        ax_lidar[2, 0].set_aspect("equal")

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
            image_lidar_data[:, 3] / np.max(image_lidar_data[:, 3]) * 255,
            dtype=np.int_,
        )

        ax_lidar[0, 1].scatter(
            image_lidar_data_norm[:, 0],
            image_lidar_data_norm[:, 1],
            c=value,
            cmap="viridis",
            s=1,
        )

        ax_lidar[1, 1].imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        ax_lidar[1, 1].scatter(
            image_lidar_data_norm[:, 0],
            max(image_lidar_data_norm[:, 1]) - image_lidar_data_norm[:, 1],
            c=value,
            cmap="viridis",
            s=1,
        )

        # plt.show()

    elif topic == image_topic:
        # Convert the ROS Image message to an OpenCV image
        # cv_img = bridge.imgmsg_to_cv2(msg)
        cv_img = bridge.compressed_imgmsg_to_cv2(msg)

        # Now cv_img is an OpenCV image, you can do any processing you need here.
        # For example, display the image
        cv2.imshow("image", cv_img)
        cv2.waitKey(1)

    plt.draw()
    plt.pause(0.000000001)

    # print(it)


bag.close()
cv2.destroyAllWindows()
