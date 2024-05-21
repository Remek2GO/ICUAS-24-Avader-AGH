import numpy as np
import rosbag
from scripts.main_node import MainNode

def get_fruit_pos(bag_name, x_center, y_center):

    frame = MainNode(50)
    frame._clb_imu()
    

    root_path = "/root/sim_ws/src/icuas24_competition/"
    bag_path = root_path + "bags/" + bag_name + ".bag"


    bag = rosbag.Bag(bag_path, "r")

    recorded_data = {
        "stamp": [],
        "frame_id": [],
        "altitude": [],
        "latitude": [],
        "longitude": [],
    }

    topic: str
    for topic, msg, t in bag.read_messages():
        if topic == "/hawkblue/mavros/global_position/global":
            recorded_data["stamp"].append(msg.header.stamp.to_sec())
            recorded_data["frame_id"].append(msg.header.frame_id)
            recorded_data["altitude"].append(msg.altitude)
            recorded_data["latitude"].append(msg.latitude)
            recorded_data["longitude"].append(msg.longitude)

    