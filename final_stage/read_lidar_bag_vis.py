import rosbag
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import cv2
import time
from numpy.linalg import inv
from matplotlib import cm

# Create an instance of CvBridge
bridge = CvBridge()

path = "/root/sim_ws/src/icuas24_competition/bags/"
# path = "/home/vision/Documents/Repositorium/icuas24_avader/bags/"
video_no = 3

### VIDEO BAG
video_name = f"ICUAS_bag_{video_no}.bag"

### CALIBRATION BAG
# video_name = f"ICUAS_calib_bag_{video_no}.bag"


## Read the bag
bag = rosbag.Bag(path + video_name)
types, topics = bag.get_type_and_topic_info()

# Print the names of all topics
for topic_name in topics.keys():
    print(topic_name)


# Topic names
lidar_topic = "/velodyne_points"
image_topic = "/camera/color/image_raw/compressed"

it = 0

#initialize the figure for 3D plot
pcd = o3d.geometry.PointCloud()

# Build the K projection matrix:
# K = [[Fx,  0, image_w/2],
#      [ 0, Fy, image_h/2],
#      [ 0,  0,         1]]
image_w = 640
image_h = 480
fov = 90.0
focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

# In this case Fx and Fy are the same since the pixel aspect
# ratio is 1
K = np.identity(3)
# K[0, 0] = K[1, 1] = focal
# K[0, 2] = image_w / 2.0
# K[1, 2] = image_h / 2.0
dot_extent = -1


D = [ 0.06787524, -0.16097136, -0.00879257, -0.00882174]
Projection = [556.45144839, 555.40489096, 319.2977065, 223.04611185]
reprojection_error = [-0.000007, -0.000001] 
K[0, 0] = 556.45144839
K[1, 1] = 555.40489096
K[0, 2] = 319.2977065
K[1, 2] = 223.04611185



from matplotlib import cm
from scipy.spatial.transform import Rotation as R

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


for topic, msg, t in bag.read_messages(topics=[lidar_topic, image_topic]):

    it += 1
    if it < 600:
        continue

    # ax0.clear()
    # ax1.clear()
    # ax2.clear()

    if topic == lidar_topic:
        lidar_data = pc2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        )

        # Convert the data to a format that can be used with open3d
        lidar_data = np.array(list(lidar_data))
        # lidar_data[:, :2] = -1*lidar_data[:, :2]

        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = 640
        image_h = 480
        # fov = camera_bp.get_attribute("fov").as_float()
        # focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        # K = np.identity(3)
        # K[0, 0] = K[1, 1] = 
        # K[0, 2] = image_w / 2.0
        # K[1, 2] = image_h / 2.0

        K = np.array([[416.54542686, 0.0, 347.32699418], [0.0, 416.87498718 ,238.18622161], [0.0,0.0, 1.0]])

        # Get the lidar data and convert it to a numpy array.
        p_cloud_size = len(lidar_data)
        p_cloud = lidar_data

        # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
        # focus on the 3D points.
        intensity = np.array(p_cloud[:, 3])

        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(p_cloud[:, :3]).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

        # This (4, 4) matrix transforms the points from lidar space to world space.
        # lidar_2_world = lidar.get_transform().get_matrix()
        rot = R.from_rotvec([np.pi, np.pi/18,0])
        lidar_transform = np.eye(4)
        lidar_transform[:3, :3] = rot.as_matrix()
        lidar_transform[3,:3] = [0.15, 0, -0.12]
        lidar_2_world = lidar_transform

        # Transform the points from lidar space to world space.
        world_points = np.dot(lidar_2_world, local_lidar_points)

        camera_transform = np.eye(4)
        camera_transform[:3, :3] = R.from_rotvec([0, 0, 0]).as_matrix()
        camera_transform[3,:3] = [6.7, 0, -24.6]

        # This (4, 4) matrix transforms the points from world to sensor coordinates.
        world_2_camera = np.linalg.inv(camera_transform)

        # Transform the points from world space to camera space.
        sensor_points = np.dot(world_2_camera, world_points)

        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # This can be achieved by multiplying by the following matrix:
        # [[ 0,  1,  0 ],
        #  [ 0,  0, -1 ],
        #  [ 1,  0,  0 ]]

        # Or, in this case, is the same as swapping:
        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(K, point_in_camera_coords)

        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])

        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(np.int_)
        v_coord = points_2d[:, 1].astype(np.int_)

        # Since at the time of the creation of this script, the intensity function
        # is returning high values, these are adjusted to be nicely visualized.
        intensity = 4 * intensity - 3
        color_map = np.array([
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int_).T

        dot_extent = -1
        if dot_extent <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
            image_camera[v_coord, u_coord] = color_map
        else:
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                image_camera[
                    v_coord[i]-dot_extent : v_coord[i]+dot_extent,
                    u_coord[i]-dot_extent : u_coord[i]+dot_extent] = color_map[i]
                

        cv2.imshow("image2", image_camera)
        cv2.waitKey(0)


    elif topic == image_topic:
        # Convert the ROS Image message to an OpenCV image
        # cv_img = bridge.imgmsg_to_cv2(msg)
        image_camera = bridge.compressed_imgmsg_to_cv2(msg)

        # Now cv_img is an OpenCV image, you can do any processing you need here.
        # For example, display the image
        cv2.imshow("image", image_camera)
        cv2.waitKey(1)

    # plt.show()
    # plt.draw()
    # plt.pause(1)

    print(it)


bag.close()
cv2.destroyAllWindows()







