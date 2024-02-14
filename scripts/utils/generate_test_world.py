import random

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
        ["eggplant_4",3,2,3],    #TODO Sprawdzic
        ["pepper_1",1,1,1],
        ["pepper_2",4,4,4],
        ["pepper_3",3,2,1],
        ["pepper_4",4,2,2],
        ["tomato_1",1,1,1],
        ["tomato_2",3,0,3],
        ["tomato_3",3,3,0]              
        ]


#roslaunch icuas24_competition spawn_plant.launch name:=plant1 model_name:=pepper_1 x:=4.0 y:=4.5 z:=6.699999999999999 yaw:=1.5707

template = "roslaunch icuas24_competition spawn_plant.launch name:="

plant_number = 1
plant_bed_number = 1


f = open('plants.txt', 'w', encoding="utf-8")

for rows in range (0,number_of_rows):   #x
    for columns in range (0,number_of_columns): #y
        for shelfs in range (0,number_of_shelfs):      #z
            print(plant_bed_number)
            plant_bed_number = plant_bed_number +1
            for plants in range(0,number_of_plants):          # roslinki

                line = template + "plant_" + str(plant_number)
                line = line + " model_name:="

                #TODO Byśmy tutaj losowali/wyznaczali roślinkę ?
                fruit_type = random.randrange(len(vege))
                #print(fruit_type) 

                line = line + vege[fruit_type][0]


                line = line + " x:=" + str(x_shelf[rows]) 
                line = line + " y:=" + str(y_shelf_offset[columns]+ y_shelf_start[plants]) 
                line = line + " z:=" + str(z_shelf[shelfs]) 
                line = line + " yaw:=" + str(yaw)
                line = line + "\n"

                plant_number = plant_number +1
                #print(line)
                f.write(line)




f.close()




