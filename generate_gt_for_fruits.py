import csv



#roslaunch icuas24_competition spawn_plant.launch name:=plant1 model_name:=pepper_3 x:=4.0 y:=6.0 z:=6.699999999999999 yaw:=1.5707


#TODO moze byc ew. wspolne z generate_test_world.py

# Wspolrzedna X - ktory rzad regalow
x_shelf = [4.0, 10.0, 16.0]

# Wspolrzedna Y - ktora pozycja w rzedzie + offset rzedu
y_shelf_start = [4.5, 6.0, 7.5]
y_shelf_offset = [0, 7.5, 15]

# Wsplorzedna Z - wysokosc
z_shelf = [1.1, 3.9, 6.699999999999999]


number_of_rows = 3
number_of_columns = 3
number_of_shelfs = 3
number_of_plants = 3


beds = []
bed_id =1
for rows in range(0, number_of_rows):  # x
    for columns in range(0, number_of_columns):  # y
        for shelfs in range(0, number_of_shelfs):  # z
                beds.append([x_shelf[rows], y_shelf_offset[columns] + y_shelf_start[0],y_shelf_offset[columns] + y_shelf_start[2] ,z_shelf[shelfs]])

#print(beds)

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

# Specify the path to your text file
file_path = 'default_world.txt'

beds_count = []
for i in range(0,27):
    beds_count.append([[0, 0, 0],[0, 0, 0],[0, 0, 0]])

#print(beds_count)


f_beds = open("beds_default.csv", "w")
w_beds = csv.writer(f_beds)

# Open the file and read it line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        
        # Decode fruite type
        model_start = line.find("model_name")
        model_end = line.find(" ",model_start)

        model = line[model_start+12:model_end]
        #print(model)

        # X
        x_start = line.find("x:=")
        x_end = line.find(" ",x_start)

        x = line[x_start+3:x_end]    
        #print(x)

        # Y
        y_start = line.find("y:=")
        y_end = line.find(" ",y_start)

        y = line[y_start+3:y_end]    
        #print(y)

         # Z
        z_start = line.find("z:=")
        z_end = line.find(" ",z_start)

        z = line[z_start+3:z_end]    

         # Z
        yaw_start = line.find("yaw:=")
        yaw_end = line.find(" ",yaw_start)

        yaw = line[yaw_start+5:yaw_end]  
        #print(z)

        # Szukamy

        #print(model)
        for i in range(0,len(beds)):
            b = beds[i]
            
            if float(x) == b[0] and float(y) >= b[1] and float(y) <= b[2] and float(z) == b[3] and model != "plant" :
                #print("Bed" + str(i+1))

                if model[0 : len(model) - 2] == "pepper":
                    vid = 0
                elif model[0 : len(model) - 2] == "eggplant":
                    vid = 2
                elif model[0 : len(model) - 2]== "tomato":
                    vid = 1

                idx =-1
                xx = 0
                for v in vege:
                    if v[0] == model :
                        idx = xx
                    xx += 1
                
                if (idx != -1):
                    beds_count[i][vid][0] += vege[idx][1]
                
                    if (float(yaw) == 1.5707):
                        beds_count[i][vid][1] += vege[idx][2]
                        beds_count[i][vid][2] += vege[idx][3]
                    else: 
                        beds_count[i][vid][1] += vege[idx][3]
                        beds_count[i][vid][2] += vege[idx][2]





            #print(beds[i])


for i, b in enumerate(beds_count):
  
  row = [
                i+1,
                "peppers",
                b[0][0],
                b[0][1],
                b[0][2],
                "tomatos",
                b[1][0],
                b[1][1],
                b[1][2],
                "eggplants",
                b[2][0],
                b[2][1],
                b[2][2],
            ]

  w_beds.writerow(row)

f_beds.close()


# Wczytywanie i tokenizacja

