#!/usr/bin/env python

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
from icuas24_competition.msg import ImageForAnalysis
from std_msgs.msg import Int32
from typing import List, Dict, Tuple
from scripts.utils.plant_bed import PlantBed, Plant, PlantSide
from scripts.utils.types import PlantType
from scripts.utils import detect_fruits
import cv2


class ImageAnalyzer:
    def __init__(self):
        self.image_topic = "/image_for_analysis"

        self.analysis_queue: List[ImageForAnalysis] = []
        self.plant_beds: Dict[int, PlantBed] = {}

        self.image_sub = rospy.Subscriber(self.image_topic, ImageForAnalysis, self.image_callback)
        self.current_fruit_count_pub = rospy.Publisher("/current_fruit_count", Int32, queue_size=10)

    def image_callback(self, msg: ImageForAnalysis):
        rospy.loginfo(f"Received image for analysis: {msg.img_path_color}")
        self.analysis_queue.append(msg)

    def analyze_image(self, image_for_analysis: ImageForAnalysis):
        if image_for_analysis.img_id == 1:
            I = cv2.imread(image_for_analysis.img_path_color)
            D = cv2.imread(image_for_analysis.img_path_depth)
            
            plant_sides, type = detect_fruits.process_frame(I, D)
            plant_type = None
            if type == 0:
                plant_type = PlantType.TOMATO
            elif type == 1:
                plant_type = PlantType.EGGPLANT
            elif type == 2:
                plant_type = PlantType.PEPPER
                
            if not image_for_analysis.bed_id in self.plant_beds:
                self.plant_beds[image_for_analysis.bed_id] = PlantBed()
                
            for i, plant_side in enumerate(plant_sides):
                self.plant_beds[image_for_analysis.bed_id].set_plant(i, image_for_analysis.bed_side, plant_side.fruit_count, plant_side.fruit_position, plant_type)
                
            rospy.loginfo(f"Plant bed {image_for_analysis.bed_id} updated")
            
    def get_fruit_count(self) -> int:
        return sum(self.plant_beds[bed_id].get_bed_fruit_count() for bed_id in self.plant_beds.keys())  
    
    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if len(self.analysis_queue) > 0:
                image_for_analysis = self.analysis_queue.pop(0)
                rospy.loginfo(f"Analyzing image: {image_for_analysis.img_path_color}")

                self.analyze_image(image_for_analysis)

                current_fruit_count = self.get_fruit_count()
                self.current_fruit_count_pub.publish(current_fruit_count)
                rospy.loginfo(f"Current fruit count: {self.get_fruit_count()}")
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('image_analyzer', anonymous=True)
    rospy.loginfo('ImageAnalyzer node started')
    image_analyzer = ImageAnalyzer()
    image_analyzer.run()