#!/usr/bin/env python
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from scripts.ImageProcessing.Tracking import AnalyzeFrame 

def publish_fruit_count(pub_y, pub_r, video_reader):
    prev_red = None
    prev_yellow = None

    rate = rospy.Rate(10)
    msg = Int32()
    if prev_red != video_reader.red_count:
        prev_red = video_reader.red_count
        msg.data = video_reader.red_count
        pub_y.publish(video_reader.red_count)
    if prev_yellow != video_reader.yellow_count:
        prev_yellow = video_reader.yellow_count
        msg.data = video_reader.yellow_count
        pub_r.publish(msg)

def publish_image(pub, bridge, video_reader, msg):
    cv_image = bridge.compressed_imgmsg_to_cv2(msg)
    marked_image = video_reader.analizer(cv_image)
    msg = bridge.cv2_to_imgmsg(marked_image, encoding="bgr8")
    pub.publish(msg)
    
def read_bag(bag):
    pub = rospy.Publisher('/fruit_detection', Image, queue_size=1)
    pub_yellow_count = rospy.Publisher('/yellow_fruit_count', Int32, queue_size=10)
    pub_red_count = rospy.Publisher('/red_fruit_count', Int32, queue_size=10)
    rospy.init_node('image', anonymous=False)

    rate = rospy.Rate(10)
    video_reader = AnalyzeFrame()
    bridge = CvBridge()

    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw/compressed']):
        if rospy.is_shutdown():
            break
        
        publish_fruit_count(pub_yellow_count,pub_red_count, video_reader)
        publish_image(pub, bridge, video_reader,msg)

        rospy.sleep(0.05)
    bag.close()

if __name__ == '__main__':
    bag_path = '/root/sim_ws/src/icuas24_competition/bags/ICUAS_bag_1.bag'
    bag = rosbag.Bag(bag_path, 'r')

    read_bag(bag)