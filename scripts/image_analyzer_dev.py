#!/usr/bin/env python

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from scripts.utils import detect_fruits

import cv2 

if __name__ == "__main__":
    img_path = "./images"
    img_list = os.listdir(img_path)
    
    # Get unique cases
    cases = [img_name.split("_")[0] for img_name in img_list if img_name.endswith(".png")]
    cases = list(set(cases))

    #cases = ["21"]
     
    for c in cases:
        print("Case:" + c)

        I = cv2.imread(os.path.join(img_path, c + "_manual_color.png"))
        D = cv2.imread(os.path.join(img_path, c + "_manual_depth.png"))

        plant_sides, type = detect_fruits.process_frame(I, D, debug=True)
    cv2.destroyAllWindows() 