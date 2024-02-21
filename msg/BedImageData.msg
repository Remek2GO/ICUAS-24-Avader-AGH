# Message to transfer the data of the bed image taken by the UAV

uint8 bed_id                        # 1-27
uint8 bed_side                      # 0-1 (0: left (closer to origin), 1: right)
uint8 img_seq                       # Sequence number of the image

bool enaugh_data                    # 0: not enough images for the (bed_id, bed_side), 1: enough images for the (bed_id, bed_side)

sensor_msgs/Image img_color         # RGB image
sensor_msgs/Image img_depth         # Depth image
UavSetpoint odom_data               # Pose of the UAV when the image was taken
