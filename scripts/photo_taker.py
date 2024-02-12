#!/usr/bin/env python
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
import numpy as np
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scripts.utils import positions
from tf.transformations import euler_from_quaternion
from typing import List, Tuple
from icuas24_competition.msg import ImageForAnalysis


bridge = CvBridge()

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images"
PROXIMITY_THRESHOLD = 0.5
YAW_THRESHOLD = np.pi/8
MAX_IMAGES = 5
FRAMES_TO_SKIP = 10

class PhotoTaker:
    def __init__(self):
        self.image_topic_depth = "/red/camera/depth/image_raw"
        self.image_topic_color = "/red/camera/color/image_raw"
        self.odom_topic = "/red/odometry"
        self.plant_beds_topic = "/red/plants_beds"
        self.image_for_analysis_topic = "/image_for_analysis"
        self.take_photo_topic = "/take_photo"

        self.current_odom: Odometry = None
        self.current_color_path: str = None
        self.current_depth_path: str = None
        self.current_color_msg: Image = None
        self.current_depth_msg: Image = None
        self.current_position_id = (-1, -1)
        self.last_position_id = (-1, -1)

        self.color_counter = 0
        self.depth_counter = 0

        rospy.Subscriber(self.plant_beds_topic, String, self.plants_beds_callback)
        rospy.Subscriber(self.take_photo_topic, Bool, self.take_photo_callback)
        self.pub_image_taken = rospy.Publisher(self.image_for_analysis_topic, ImageForAnalysis, queue_size=10)
        
    def take_photo_callback(self, msg: Bool):
        pos_id = self.current_position_id
        if msg.data and not -1 in pos_id:
            img_color = bridge.imgmsg_to_cv2(self.current_color_msg, "bgr8")
            img_depth = bridge.imgmsg_to_cv2(self.current_depth_msg, "8UC1")
            
            unique_id = f"{self.last_position_id[0]}{self.last_position_id[1]}_manual"
            path = f'{IMAGES_FOLDER_PATH}/{unique_id}'
            
            cv2.imwrite(f"{path}_color.png", img_color)
            cv2.imwrite(f"{path}_depth.png", img_depth)
            
            img_msg = ImageForAnalysis()
            img_msg.img_path_color = f"{path}_color.png"
            img_msg.img_path_depth = f"{path}_depth.png"
            img_msg.bed_id = np.uint8(self.last_position_id[0])
            img_msg.bed_side = np.uint8(self.last_position_id[1])
            img_msg.img_id = 10
            img_msg.pose = self.current_odom.pose.pose
            self.pub_image_taken.publish(img_msg)
        

    def plants_beds_callback(self, msg: String):
        rospy.Subscriber(self.image_topic_depth, Image, self.image_callback_depth)
        rospy.Subscriber(self.image_topic_color, Image, self.image_callback_color)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        bed_ids = [int(bed_id) for bed_id in msg.data.split(" ")[1:]]
        self.set_bed_ids(bed_ids)

    def set_bed_ids(self, bed_ids: List[int]):
        self.bed_ids = bed_ids
        self.bed_id_to_id = {bed_id: i for i, bed_id in enumerate(bed_ids)}
        self.successful_shots = np.zeros((len(bed_ids), 2, 2), dtype=np.uint8)

    def image_callback_color(self, msg):
        pos_id = self.current_position_id
        self.current_color_msg = msg
        if not -1 in pos_id and self.successful_shots[self.bed_id_to_id[pos_id[0]], pos_id[1], 0] < MAX_IMAGES:
            self.color_counter += 1
            if self.color_counter%FRAMES_TO_SKIP == 1:
                try:
                    cv2_img_color = bridge.imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError:
                    rospy.logwarn('Error converting color image')
                else:
                    unique_id = f"{pos_id[0]}{pos_id[1]}{self.successful_shots[self.bed_id_to_id[pos_id[0]], pos_id[1], 0]}"
                    path = f'{IMAGES_FOLDER_PATH}/{unique_id}'

                    cv2.imwrite(f"{path}_color.png", cv2_img_color)
                    self.successful_shots[self.bed_id_to_id[pos_id[0]], pos_id[1], 0] += 1
                    
                    self.current_color_path = f"{path}_color.png"
                    # rospy.loginfo(f'Image saved as {path}_color.png')
        else:
            self.color_counter = 0

    def image_callback_depth(self, msg):
        pos_id = self.current_position_id
        self.current_depth_msg = msg
        if not -1 in pos_id and self.successful_shots[self.bed_id_to_id[pos_id[0]], pos_id[1], 1] < MAX_IMAGES:
            self.depth_counter += 1
            if self.depth_counter%FRAMES_TO_SKIP == 1:
                try:
                    cv2_img_depth = bridge.imgmsg_to_cv2(msg, "8UC1")
                except CvBridgeError:
                    rospy.logwarn('Error converting depth image')
                else:
                    path = f'{IMAGES_FOLDER_PATH}/{pos_id[0]}{pos_id[1]}{self.successful_shots[self.bed_id_to_id[pos_id[0]], pos_id[1], 1]}'
                    cv2.imwrite(f"{path}_depth.png", cv2_img_depth)
                    self.successful_shots[self.bed_id_to_id[pos_id[0]], pos_id[1], 1] += 1

                    self.current_depth_path = f"{path}_depth.png"
                    # rospy.loginfo(f'Image saved as {path}_depth.png')
        else:
            self.depth_counter = 0

    def odom_callback(self, msg):
        self.current_odom = msg
        for bed_id in self.bed_ids:
            for position_id in range(2):
                if self.is_close_to_position(msg, (bed_id, position_id)):
                    self.current_position_id = (bed_id, position_id)
                    self.last_position_id = self.current_position_id
                    return
        if self.current_position_id != (-1, -1):
            rospy.loginfo(f"Took {self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 0]} images of bed {self.current_position_id[0]} side {self.current_position_id[1]}")
        self.current_position_id = (-1, -1)
            
    def is_close_to_position(self, odom_msg: Odometry, position_ids: Tuple[int, int]) -> bool:
        odom_position = [
            odom_msg.pose.pose.position.x, 
            odom_msg.pose.pose.position.y, 
            odom_msg.pose.pose.position.z, 
            *euler_from_quaternion([odom_msg.pose.pose.orientation.x, 
                                    odom_msg.pose.pose.orientation.y, 
                                    odom_msg.pose.pose.orientation.z, 
                                    odom_msg.pose.pose.orientation.w])]
        poi_position = positions.POINTS_OF_INTEREST[position_ids[0]][position_ids[1]]
        
        return np.linalg.norm(np.array(odom_position[:3]) - np.array(poi_position[:3])) < PROXIMITY_THRESHOLD and \
                (np.abs(odom_position[-1]-poi_position[-1]) < YAW_THRESHOLD or np.abs(odom_position[-1]-poi_position[-1]) > 2*np.pi-YAW_THRESHOLD)
    
    def run(self):
        while not rospy.is_shutdown():
            if self.current_color_path is not None and self.current_depth_path is not None:
                msg = ImageForAnalysis()
                msg.img_path_color = self.current_color_path
                msg.img_path_depth = self.current_depth_path
                msg.bed_id = np.uint8(self.last_position_id[0])
                msg.bed_side = np.uint8(self.last_position_id[1])
                msg.img_id = int(self.successful_shots[self.bed_id_to_id[self.last_position_id[0]], self.last_position_id[1], 0])
                msg.pose = self.current_odom.pose.pose

                self.pub_image_taken.publish(msg)
                self.current_color_path = None
                self.current_depth_path = None
            rospy.sleep(0.01)
    

if __name__ == "__main__":
    rospy.init_node('photo_taker')
    rospy.loginfo('photo_taker node started')

    photo_taker = PhotoTaker()
    photo_taker.run()







