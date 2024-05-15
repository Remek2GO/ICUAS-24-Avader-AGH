import numpy  as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

path = "/home/vision/Documents/Repositorium/icuas24_avader/bags/"
lidar_no = 1
lidar_folder_name = f"ICUAS_bag_{lidar_no}"


# read files in the folder

lidar_folder_path = os.path.join(path, lidar_folder_name)
files = [f for f in os.listdir(lidar_folder_path) if os.path.isfile(os.path.join(lidar_folder_path, f))]

for file in files:

    file_path = os.path.join(lidar_folder_path, file)
    print(f"Processing file ... {file}")

    # read lidar data
    lidar_data = np.fromfile(file_path, dtype=np.float32)
    lidar_data = lidar_data.reshape(-1, 4)

    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, 0:3])

    # display point cloud
    o3d.visualization.draw_geometries([pcd])

    