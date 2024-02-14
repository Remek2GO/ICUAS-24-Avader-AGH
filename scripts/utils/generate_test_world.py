import random
import csv


#Wspolrzedna X - ktory rzad regalow
x_shelf = [4.0,10.0,16.0]

#Wspolrzedna Y - ktora pozycja w rzedzie + offset rzedu
y_shelf_start = [4.5,6.0,7.5]
y_shelf_offset = [0,7.5,15]

#Wsplorzedna Z - wysokosc
z_shelf = [1.1,3.9,6.699999999999999]

#
yaw = 1.5707




number_of_rows = 3
number_of_columns = 3
number_of_shelfs = 3
number_of_plants = 3

#roslinka suma warzyw, lewa prawa (ile jest widocznych)
vege = [["plant",0,0,0],
        ["eggplant_1",1,1,0],
        ["eggplant_2",2,2,1],    #TODO Sprawdzic
        ["eggplant_3",3,2,1],    
        ["eggplant_4",3,3,2],    #TODO Sprawdzic
        ["pepper_1",1,1,1],
        ["pepper_2",4,4,4],
        ["pepper_3",3,2,1],
        ["pepper_4",4,2,2],
        ["tomato_1",1,1,1],
        ["tomato_2",2,2,1],
        ["tomato_3",3,3,0]              
        ]


#roslaunch icuas24_competition spawn_plant.launch name:=plant1 model_name:=pepper_1 x:=4.0 y:=4.5 z:=6.699999999999999 yaw:=1.5707

#
# Orignialny przelot
#rostopic pub --latch /$UAV_NAMESPACE/plants_beds std_msgs/String "Pepper 10 11 14 21 23 24"

template = "roslaunch icuas24_competition spawn_plant.launch name:="

plant_number = 1
plant_bed_number = 1


f = open('plants.txt', 'w', encoding="utf-8")

f_plants = open('plants.csv', 'w')
w_platns = csv.writer(f_plants)
f_beds = open('beds.csv', 'w')
w_beds = csv.writer(f_beds)


for rows in range (0,number_of_rows):   #x
    for columns in range (0,number_of_columns): #y
        for shelfs in range (0,number_of_shelfs):      #z
            print(plant_bed_number)
           
            peppers = [0,0,0]           
            eggplants = [0,0,0]           
            tomatos = [0,0,0]
           

            for plants in range(0,number_of_plants):          # roslinki

                line = template + "plant_" + str(plant_number)
                line = line + " model_name:="

                #TODO Byśmy tutaj losowali/wyznaczali roślinkę ?
                fruit_type = random.randrange(len(vege))

                fruite_name = vege[fruit_type][0]

   
                if fruite_name[0:len(fruite_name)-2] == "pepper":
                    peppers[0] += vege[fruit_type][1]
                    peppers[1] += vege[fruit_type][2]
                    peppers[2] += vege[fruit_type][3]
                    row = [plant_bed_number,plant_bed_number, "pepper", vege[fruit_type][1], vege[fruit_type][2], vege[fruit_type][3]  ]
                    w_platns.writerow(row)
                        
                if fruite_name[0:len(fruite_name)-2] == "tomato":
                    tomatos[0] += vege[fruit_type][1]
                    tomatos[1] += vege[fruit_type][2]
                    tomatos[2] += vege[fruit_type][3]
                    row = [plant_bed_number,plant_bed_number, "tomato", vege[fruit_type][1], vege[fruit_type][2], vege[fruit_type][3]  ]
                    w_platns.writerow(row)

                if fruite_name[0:len(fruite_name)-2] == "eggplant":
                     eggplants[0] += vege[fruit_type][1]
                     eggplants[1] += vege[fruit_type][2]
                     eggplants[2] += vege[fruit_type][3]
                     row = [plant_bed_number,plant_bed_number, "eggplant", vege[fruit_type][1], vege[fruit_type][2], vege[fruit_type][3]  ]
                     w_platns.writerow(row)
                    
                    #print(fruite_name[0:len(fruite_name)-2] )

                #print(fruit_type) 

                line = line + fruite_name


                line = line + " x:=" + str(x_shelf[rows]) 
                line = line + " y:=" + str(y_shelf_offset[columns]+ y_shelf_start[plants]) 
                line = line + " z:=" + str(z_shelf[shelfs]) 
                line = line + " yaw:=" + str(yaw)
                line = line + "\n"

                plant_number = plant_number +1
                #print(line)
                f.write(line)

            # Zapis informacji o polce
            
            print(peppers)
            print(tomatos)
            print(eggplants)

            row = [plant_bed_number, "peppers", peppers[0],peppers[1],peppers[2], "tomatos", tomatos[0],tomatos[1],tomatos[2], "eggplants", eggplants[0],eggplants[1],eggplants[2] ]
            
            w_beds.writerow(row)
            plant_bed_number = plant_bed_number +1


f.close()
f_beds.close()
f_plants.close()
