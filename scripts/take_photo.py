#!/usr/bin/env python

import rospy
import os
import numpy as np
# ROS Image message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()

class Photo:
    def __init__(self):
        rospy.init_node('take_photo')
        self.color_taken = False
        self.depth_taken = False
        self.image_topic_depth = "/red/camera/depth/image_raw"
        self.image_topic_color = "/red/camera/color/image_raw"
        rospy.Subscriber(self.image_topic_depth, Image, self.image_callback_depth)
        rospy.Subscriber(self.image_topic_color, Image, self.image_callback_color)
        rospy.spin()

    def image_callback_color(self, msg):
        if not self.color_taken:
            print("Received an image!") #  os.getcwd()
            try:
                # Convert your ROS Image message to OpenCV2
                cv2_img_color = bridge.imgmsg_to_cv2(msg, "bgr8") 
                self.color_taken = True
            except CvBridgeError:
                print('error')
            else:
                # Save your OpenCV2 image as a jpeg 
                time = msg.header.stamp
                cv2.imwrite(''+str(time)+'_color.png', cv2_img_color)
                self.color_taken = True
                rospy.loginfo("Image saved as "+str(time)+"_color.png")
                #rospy.signal_shutdown("Image saved")
                #rospy.sleep(1)

    def image_callback_depth(self, msg):
        if not self.depth_taken:
            print("Received an image!") #  os.getcwd()
            #rospy.loginfo(np.dtype(msg.data))
            #rospy.sleep(1)
            #rospy.signal_shutdown("Image saved")
            try:
                # Convert your ROS Image message to OpenCV2
                cv2_img_depth = bridge.imgmsg_to_cv2(msg, "8UC1")
                self.depth_taken = True
            except CvBridgeError:
                print('error')
            else:
                # Save your OpenCV2 image as a jpeg 
                time = msg.header.stamp
                cv2.imwrite(''+str(time)+'_depth.png', cv2_img_depth)
                self.depth_taken = True
                rospy.loginfo("Image saved as "+str(time)+"_depth.png")
                #rospy.signal_shutdown("Image saved")
                #rospy.sleep(1)

if __name__ == '__main__':
    take_photo = Photo()
    while not take_photo.color_taken and not take_photo.depth_taken:
        rospy.sleep(1)
    rospy.signal_shutdown("Images saved")
    