All bags have the following topics (calib bags do not have the GPS topic tho):

topics:      /camera/color/image_raw/compressed        : sensor_msgs/CompressedImage
             /hawkblue/mavros/global_position/global   : sensor_msgs/NavSatFix      
             /hawkblue/mavros/imu/data                 : sensor_msgs/Imu            
             /tf_static                                : tf2_msgs/TFMessage         
             /velodyne_points                          : sensor_msgs/PointCloud2
             
Topic description:
/camera/color/image_raw/compressed video stream from a front facing camera of the UAV at 30 fps
/hawkblue/mavros/global_position/global gps data from the gps receiver of the UAV at 50Hz
/hawkblue/mavros/imu/data imu data at 50Hz
/velodyne_points velodyne 3d lidar (VLP-16) at 10 Hz

Camera position (CAD, not calibrated) in IMU frame: (15, 0, -12) cm, roll = 0, pitch = 0, yaw = 0
Lidar position (CAD, not calibrated) in IMU frame: (6.7, 0, -24.6) cm, roll = 0, pitch = 10 degrees, yaw = 0

