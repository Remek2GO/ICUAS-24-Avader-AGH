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

for topic, msg, t in bag.read_messages(topics=[lidar_topic, image_topic]):

    it += 1
    if it < 600:
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
        # lidar_data[:, :2] = -1*lidar_data[:, :2]

        # lidar_data = lidar_data[lidar_data[:,2] > 0.5]

        # ax0.scatter(
        #     lidar_data[:, 0], lidar_data[:, 1], c=lidar_data[:, 2], cmap="viridis", s=1
        # )
        # ax0.set_aspect("equal")

        front_lidar_data = lidar_data[
            np.logical_and(lidar_data[:, 0] > 0, lidar_data[:, 0] < 10)
        ]
        front_lidar_data = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -5, front_lidar_data[:, 1] < 5)
        ]

        # distance at point y = 0 and z = 0 (in the lidar frame)

        distance_x = front_lidar_data[
            np.logical_and(front_lidar_data[:, 1] > -0.1, front_lidar_data[:, 1] < 0.1)
        ][:, 0]
        distance_lidar = np.mean(distance_x)
        image_lidar_data = front_lidar_data[
            np.logical_and(
                front_lidar_data[:, 1] < distance_lidar,
                front_lidar_data[:, 1] > -1 * distance_lidar,
            )
        ]

        image = np.ones((height, width, 3), dtype=np.uint8) * 255

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
            image_lidar_data[:, 0] / np.max(image_lidar_data[:, 0])  * 255,
            dtype=np.int_,
        )
        for x, y, data in zip(
            image_lidar_data_norm[:, 0],
            image_lidar_data_norm[:, 1],
            value,
        ):
            # print(y, x, data)
            image[int(y), int(x), :] = [128, data, 128]

        cv2.imshow("image_lidar_data", image)

        # ax2.scatter(
        #     image_lidar_data[:, 1], image_lidar_data[:, 2], c=image_lidar_data[:, 0], cmap="viridis", s=1
        # )
        # ax2.set_aspect("equal")

        # Create a point cloud
        # pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])
        # Display the point cloud

        # o3d.visualization.draw_geometries([pcd])
        # 3: add or update geometry objects
        # if not create_o3d_obj:
        #     vis.add_geometry(pcd)  # add point cloud
        #     create_o3d_obj = True  # change flag
        # else:
        #     vis.update_geometry(pcd)  # update point cloud

        # # 4 update o3d window
        # if not vis.poll_events():
        #     break
        # vis.update_renderer()

    elif topic == image_topic:
        # Convert the ROS Image message to an OpenCV image
        # cv_img = bridge.imgmsg_to_cv2(msg)
        cv_img = bridge.compressed_imgmsg_to_cv2(msg)

        # Now cv_img is an OpenCV image, you can do any processing you need here.
        # For example, display the image
        cv2.imshow("image", cv_img)
        cv2.waitKey(0)

    # plt.show()
    # plt.draw()
    # plt.pause(1)

    print(it)


bag.close()
cv2.destroyAllWindows()
