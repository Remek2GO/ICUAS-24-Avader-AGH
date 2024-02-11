#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scripts import positions
from tf.transformations import euler_from_quaternion
from typing import List, Tuple

bridge = CvBridge()

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images"
PROXIMITY_THRESHOLD = 0.5
YAW_THRESHOLD = np.pi/4
MAX_IMAGES = 5
FRAMES_TO_SKIP = 10

class PhotoTaker:
    def __init__(self, bed_ids: List[int]) -> None:
        self.image_topic_depth = "/red/camera/depth/image_raw"
        self.image_topic_color = "/red/camera/color/image_raw"
        self.odom_topic = "/red/odometry"

        self.bed_ids = bed_ids
        self.bed_id_to_id = {bed_id: i for i, bed_id in enumerate(bed_ids)}
        self.successful_shots = np.zeros((len(bed_ids), 2, 2))
        self.current_position_id = (-1, -1)

        self.color_counter = 0
        self.depth_counter = 0

        rospy.Subscriber(self.image_topic_depth, Image, self.image_callback_depth)
        rospy.Subscriber(self.image_topic_color, Image, self.image_callback_color)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

    def image_callback_color(self, msg):
        if self.current_position_id != (-1, -1) and self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 0] < MAX_IMAGES:
            self.color_counter += 1
            if self.color_counter%FRAMES_TO_SKIP == 1:
                try:
                    cv2_img_color = bridge.imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError:
                    rospy.logwarn('Error converting color image')
                else:
                    cv2.imwrite(f'{IMAGES_FOLDER_PATH}/{self.current_position_id[0]}{self.current_position_id[1]}{self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 0]}_color.png', cv2_img_color)
                    self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 0] += 1
                    rospy.loginfo("Image saved as "+f'{IMAGES_FOLDER_PATH}/{self.current_position_id[0]}{self.current_position_id[1]}{self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 0]}_color.png')
        else:
            self.color_counter = 0

    def image_callback_depth(self, msg):
        if self.current_position_id != (-1, -1) and self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 1] < MAX_IMAGES:
            self.depth_counter += 1
            if self.depth_counter%FRAMES_TO_SKIP == 1:
                try:
                    cv2_img_depth = bridge.imgmsg_to_cv2(msg, "8UC1")
                except CvBridgeError:
                    rospy.logwarn('Error converting depth image')
                else:
                    cv2.imwrite(f'{IMAGES_FOLDER_PATH}/{self.current_position_id[0]}{self.current_position_id[1]}{self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 1]}_depth.png', cv2_img_depth)
                    self.successful_shots[self.bed_id_to_id[self.current_position_id[0]], self.current_position_id[1], 1] += 1
                    rospy.loginfo("Image saved as "+f'{IMAGES_FOLDER_PATH}/{self.current_position_id[0]}{self.current_position_id[1]}_depth.png')
        else:
            self.depth_counter = 0

    def odom_callback(self, msg):
        for bed_id in self.bed_ids:
            for position_id in range(2):
                if self.is_close_to_position(msg, (bed_id, position_id)):
                    self.current_position_id = (bed_id, position_id)
                    return
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







