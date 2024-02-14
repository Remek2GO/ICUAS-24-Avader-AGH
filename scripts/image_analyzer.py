#!/usr/bin/env python

import os
import sys
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
from icuas24_competition.msg import ImageForAnalysis
from std_msgs.msg import Int32, String
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
        self.fruit_type: PlantType = None

        self.image_sub = rospy.Subscriber(
            self.image_topic, ImageForAnalysis, self.image_callback
        )
        self.current_fruit_count_pub = rospy.Publisher(
            "/current_fruit_count", Int32, queue_size=10
        )

        self.sub_plants_beds = rospy.Subscriber(
            "/red/plants_beds", String, self.set_fruit_type
        )

        self.f_beds = open('/root/sim_ws/src/icuas24_competition/images/beds_out.csv', 'w')
        self.w_beds = csv.writer(self.f_beds)

    def __del__(self):

        #TODO Tu wypisanie informacji o tych danych ?    
        for bed_id in self.plant_beds.keys():
            no_pepper = self.plant_beds[bed_id].get_bed_fruit_count(PlantType.PEPPER)
            no_pepper_l = self.plant_beds[bed_id].get_bed_fruit_count_left(PlantType.PEPPER)
            no_pepper_r = self.plant_beds[bed_id].get_bed_fruit_count_right(PlantType.PEPPER)
            
            no_tomato = self.plant_beds[bed_id].get_bed_fruit_count(PlantType.TOMATO)
            no_tomato_l = self.plant_beds[bed_id].get_bed_fruit_count_left(PlantType.TOMATO)
            no_tomato_r = self.plant_beds[bed_id].get_bed_fruit_count_right(PlantType.TOMATO)
            
            no_eggplant = self.plant_beds[bed_id].get_bed_fruit_count(PlantType.EGGPLANT)
            no_eggplant_l = self.plant_beds[bed_id].get_bed_fruit_count_left(PlantType.EGGPLANT)
            no_eggplant_r = self.plant_beds[bed_id].get_bed_fruit_count_right(PlantType.EGGPLANT)
            
            row = [bed_id,"peppers",no_pepper,no_pepper_l,no_pepper_r,"tomatos",no_tomato,no_tomato_l,no_tomato_r,"eggplants",no_eggplant,no_eggplant_l,no_eggplant_r]
            self.w_beds.writerow(row)
        
            

        self.f_beds.close()

    def set_fruit_type(self, data: String):
        type = data.data.split(" ")[0]

        self.fruit_type = PlantType(type.upper())

    def image_callback(self, msg: ImageForAnalysis):
        # rospy.loginfo(f"Received image for analysis: {msg.img_path_color}")
        self.analysis_queue.append(msg)

    def analyze_image(self, image_for_analysis: ImageForAnalysis):
        #self.w_beds.writerow(["test_2,test_2,test_2,test_2"])
               
        if image_for_analysis.img_id == 10:
            #self.w_beds.writerow("TEST")
            I = cv2.imread(image_for_analysis.img_path_color)
            D = cv2.imread(image_for_analysis.img_path_depth)
            
            plant_sides, type = detect_fruits.process_frame(I, D)
            #row = []
            #row.append(image_for_analysis.bed_id)
            
            #for p in plant_sides:
            #    row.append(p.fruit_type)
            #    row.append(p.fruit_count)
           
            # Jesli nie przeanalizowano ??    
            # Dodawanie pustego plant bed ??
            if not image_for_analysis.bed_id in self.plant_beds:
                self.plant_beds[image_for_analysis.bed_id] = PlantBed()

            # Obliczenia dla ew. drugiej strony (korekta)
                
            for i, plant_side in enumerate(plant_sides):
                idx = i if image_for_analysis.bed_side == 0 else len(plant_sides) - i - 1
                self.plant_beds[image_for_analysis.bed_id].set_plant(idx, image_for_analysis.bed_side, plant_side.fruit_count, plant_side.fruit_position.copy(), plant_side.fruit_type)


           

            #, self.plant_beds[image_for_analysis.bed_id]., sum([side.fruit_count for side in plant_sides])]
            #        self.plant_beds[image_for_analysis.bed_id].left.fruit_count, self.plant_beds[image_for_analysis.bed_id].right.fruit_count ]

            #f_name = f"{image_for_analysis.bed_id}_log.txt"
            #f_name = "xxx.txt"
            #f_beds = open("/root/sim_ws/src/icuas24_competition/images/"+f_name, 'w')
            #f_beds.write(row)
            #f_beds.close()


            #print("[!!!] " + row)
            #self.w_beds.writerow(row)
            
            
                
            # rospy.loginfo(f"Plant bed {image_for_analysis.bed_id} updated")
            rospy.loginfo(
                f"Bed #{image_for_analysis.bed_id} side {image_for_analysis.bed_side} found {sum([side.fruit_count for side in plant_sides])} fruits"
            )
            rospy.loginfo(f"Current fruit count: {self.get_fruit_count(True)}")

        #self.w_beds.writerow(["test_3,test_3,test_3,test_3"])

    def get_fruit_count(self, debug=False) -> int:
        sums = 0
        for bed_id in self.plant_beds.keys():
            no_fruits = self.plant_beds[bed_id].get_bed_fruit_count(self.fruit_type)
            if debug:
                rospy.loginfo(f"Bed #{bed_id} total fruit count: {no_fruits}")
            sums += no_fruits

        return sums


    def run(self):
        rate = rospy.Rate(100)
        
        #self.w_beds.writerow(["test","test","test"])
        while not rospy.is_shutdown():
            if len(self.analysis_queue) > 0:
                image_for_analysis = self.analysis_queue.pop(0)
                rospy.loginfo(f"Analyzing image: {image_for_analysis.img_path_color}")
                self.analyze_image(image_for_analysis)

                current_fruit_count = self.get_fruit_count()
                self.current_fruit_count_pub.publish(current_fruit_count)
            rate.sleep()
        



if __name__ == "__main__":
    rospy.init_node("image_analyzer", anonymous=True)
    rospy.loginfo("ImageAnalyzer node started")
    image_analyzer = ImageAnalyzer()
    image_analyzer.run()
