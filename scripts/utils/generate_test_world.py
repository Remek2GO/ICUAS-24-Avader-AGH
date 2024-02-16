#!/usr/bin/env python

import random
import csv


# Wspolrzedna X - ktory rzad regalow
x_shelf = [4.0, 10.0, 16.0]

# Wspolrzedna Y - ktora pozycja w rzedzie + offset rzedu
y_shelf_start = [4.5, 6.0, 7.5]
y_shelf_offset = [0, 7.5, 15]

# Wsplorzedna Z - wysokosc
z_shelf = [1.1, 3.9, 6.699999999999999]

#
yaw = 1.5707


number_of_rows = 3
number_of_columns = 3
number_of_shelfs = 3
number_of_plants = 3

# roslinka suma warzyw, lewa prawa (ile jest widocznych)
vege = [
    ["plant", 0, 0, 0],
    ["eggplant_1", 1, 1, 0],
    ["eggplant_2", 2, 2, 1],  # TODO Sprawdzic
    ["eggplant_3", 3, 2, 1],
    ["eggplant_4", 3, 3, 2],  # TODO Sprawdzic
    ["pepper_1", 1, 1, 1],
    ["pepper_2", 4, 4, 4],
    ["pepper_3", 3, 2, 1],
    ["pepper_4", 4, 2, 2],
    ["tomato_1", 1, 1, 1],
    ["tomato_2", 2, 2, 1],
    ["tomato_3", 3, 3, 0],
]


# roslaunch icuas24_competition spawn_plant.launch name:=plant1 model_name:=pepper_1 x:=4.0 y:=4.5 z:=6.699999999999999 yaw:=1.5707

#
# Orignialny przelot
# rostopic pub --latch /$UAV_NAMESPACE/plants_beds std_msgs/String "Pepper 10 11 14 21 23 24"
# 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
#  rostopic pub --latch /$UAV_NAMESPACE/plants_beds std_msgs/String "Tomato 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"


template = "roslaunch icuas24_competition spawn_plant.launch name:="

plant_number = 1
plant_bed_number = 1


f = open("plants.txt", "w", encoding="utf-8")

f_plants = open("plants.csv", "w")
w_platns = csv.writer(f_plants)
f_beds = open("beds.csv", "w")
w_beds = csv.writer(f_beds)


plant_names = ["pepper", "tomato", "eggplant"]
# Zliczenia warzyw: suma, lewa strona, prawa strona.


for rows in range(0, number_of_rows):  # x
    for columns in range(0, number_of_columns):  # y
        for shelfs in range(0, number_of_shelfs):  # z
            print(plant_bed_number)
            plant_counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

            # Ustalenie liczy rośline na półce (na pierwszej po jednej roślince, na drugiej po dwie)
            numbers = [0, 1, 2]
            if rows == 0 and columns == 0:
                plant_locations = random.sample(numbers, 1)
            elif rows == 0 and columns == 1:
                plant_locations = random.sample(numbers, 2)
            else:
                plant_locations = numbers

            for plants in plant_locations:  # roslinki

                line = template + "plant_" + str(plant_number)
                line = line + " model_name:="

                # Losowanie roslinki
                fruit_type = random.randrange(len(vege))

                fruite_name = vege[fruit_type][0]

                # Wyznacznie orientacji krzaczka (losowa)
                fruit_orientation = random.randrange(2)

                # Dekodowanie id warzywa
                if fruite_name[0 : len(fruite_name) - 2] == "pepper":
                    vid = 0
                elif fruite_name[0 : len(fruite_name) - 2] == "eggplant":
                    vid = 2
                elif fruite_name[0 : len(fruite_name) - 2] == "tomato":
                    vid = 1

                # Zapisaywanie ile jest warzyw danego typu
                plant_counts[vid][0] += vege[fruit_type][1]
                if fruit_orientation == 0:
                    plant_counts[vid][1] += vege[fruit_type][2]
                    plant_counts[vid][2] += vege[fruit_type][3]
                else:
                    plant_counts[vid][1] += vege[fruit_type][3]
                    plant_counts[vid][2] += vege[fruit_type][2]

                row = [
                    plant_bed_number,
                    plant_bed_number,
                    plant_names[vid],
                    vege[fruit_type][1],
                    vege[fruit_type][2],
                    vege[fruit_type][3],
                ]
                w_platns.writerow(row)

                # Utworzenie linijki konfiguracyjnej.

                line = line + fruite_name

                line = line + " x:=" + str(x_shelf[rows])
                line = (
                    line + " y:=" + str(y_shelf_offset[columns] + y_shelf_start[plants])
                )
                line = line + " z:=" + str(z_shelf[shelfs])
                if fruit_orientation == 0:
                    line = line + " yaw:=" + str(yaw)
                else:
                    line = line + " yaw:=" + str(-yaw)
                line = line + "\n"

                plant_number = plant_number + 1
                # print(line)
                f.write(line)

            # Zapis informacji o polce

            # print(peppers)
            # print(tomatos)
            # print(eggplants)

            row = [
                plant_bed_number,
                "peppers",
                plant_counts[0][0],
                plant_counts[0][1],
                plant_counts[0][2],
                "tomatos",
                plant_counts[1][0],
                plant_counts[1][1],
                plant_counts[1][2],
                "eggplants",
                plant_counts[2][0],
                plant_counts[2][1],
                plant_counts[2][2],
            ]

            w_beds.writerow(row)
            plant_bed_number = plant_bed_number + 1


f.close()
f_beds.close()
f_plants.close()
