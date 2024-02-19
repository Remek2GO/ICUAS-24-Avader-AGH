import cv2
import numpy as np


# TODO 
# 1. Detekcja śmigła... 






# start_image

IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images_eval"
 
id = 2440


h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255
light = 100



def mask_plants(hsv):
    # TODO Przenieść progi do gory
    # Wykrywanie bialych obszarow
    mask_white = cv2.inRange(hsv, (0, 0, 250), (5, 3, 255))

    # TODO Przenieść progi do gory
    # Wykrywanie zielonych obszarow
    mask_green = cv2.inRange(hsv, (50, 0, 45), (60, 180, 255))

    mask_wg = cv2.bitwise_or(mask_white, mask_green)

    #   cv2.imshow('Mask white and green',mask_wg)

    # Eksperyment
    kernel_1_25 = np.ones((1, 25), np.uint8)
    kernel_25_1 = np.ones((25, 1), np.uint8)

    th_RGB_F = cv2.dilate(mask_wg, kernel_1_25)
    th_RGB_F = cv2.dilate(th_RGB_F, kernel_25_1)

    th_RGB_F = cv2.medianBlur(th_RGB_F, 7)
    th_RGB_F = np.uint8(th_RGB_F / 255)

    #cv2.imshow("Mask after median blur",mask_wg)
    return th_RGB_F


def mask_rotors(hsv):

    #TODO do ew. korekty
    
    mask_blue = cv2.inRange(hsv, (113, 75, 0), (122, 255, 255))

    mask_red = cv2.inRange(hsv, (0, 230, 0), (2, 255, 255))

    mask_rotors = cv2.bitwise_or(mask_blue, mask_red)

    return mask_rotors

for id in range(2440,2640):


    unique_id = f"{id}_eval"
    path = f"{IMAGES_FOLDER_PATH}/{unique_id}"
    I = cv2.imread(f"{path}_color.png")
    D = cv2.imread(f"{path}_depth.png")
    f = open(f"{path}_odom.txt","r")
    t =f.readline()
    f.close()


    #print(t)
    cv2.imshow("I",I)



    # Obrót zdjęcia względem roll
    ts = t.split()
    roll = float(ts[3])

    roll_d = -1 * roll/ np.pi * 180
    dimensions = I.shape
    cols = int(dimensions[1]*1.2)
    rows = int(dimensions[0]*1.2)
    M = cv2.getRotationMatrix2D((float(dimensions[0]/2),float(dimensions[1]/2)),roll_d,1) 
    IR = cv2.warpAffine(I,M,(cols,rows)) 
    DR = cv2.warpAffine(D,M,(cols,rows)) 
    #DR = cv2.cvtColor(DR, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("IR",IR)
    cv2.imshow("DR",DR*10  )


    # Segmentacja mapy głębi - poszukiwanie kwadratow (jak na tym obrazie)
    hsv = cv2.cvtColor(IR, cv2.COLOR_BGR2HSV)
    plants_mask = mask_plants(hsv)

    DR = cv2.cvtColor(DR, cv2.COLOR_BGR2GRAY)
    result = cv2.multiply(plants_mask, DR)
    result = cv2.medianBlur(result,5)
    cv2.imshow("B",plants_mask*10)

    B = np.uint8(result == 3) * 255
    cv2.imshow("B",B)

    R = mask_rotors(hsv)
    cv2.imshow("Rotors",R)

    # Analiza:
    print(f"=== NEW FRAME === ")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(B)
    for i in range(1, num_labels):
        left, top, width, height, area = stats[i]
        # Korekcja
        left = left + 12
        width = width - 24
        height = height - 12
        # TODO Dodatkowe warunki a ten sprawdzic
        # TODO Korekcja ramki
        sqare_ratio = width / height 
        print(f"SQUARE = {sqare_ratio} AREA = {area}")

        


        if area > 100 * 100 and sqare_ratio > 1 and sqare_ratio < 1.3:

            #TODO Nie powinno być wirników w kadrze !!!
            rotor_rotor = R[top: top + height, left: left + width]   
            cv2.imshow("RR",rotor_rotor) 
           
            rotors_in_fov = False
            num_labels_r, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(rotor_rotor)
            # Jesli jest jakikolwiek obiekt (0 to zawsze jest tło)
            if (num_labels_r > 1):
                for rr in range(1, num_labels_r):
                    left_r, top_r, width_r, height_r, area_r = stats_r[rr]  #TPDO a jak będzie jakiś szum ?
                    if area_r > 100:                
                        print(f"Rotor area {area_r}")
                        rotors_in_fov = True
            
            if not rotors_in_fov:
            
                # TODO Polaczyc z dylatacja
                # Reczna korekcja (wynika z rozmiaru dylatacji)
                cv2.rectangle(IR, (left, top), (left + width, top + height), 255, 2)
            #if DEBUG_MODE:
            #    print(
            #        f"Component {i}: Area={area}, Centroid={centroids[i].astype(int)}"
            #    )
            # Wycinek
            #patches.append((top, top + height, left, left + width))

            # if (count >0 ):
            #   text = f"{count} {fruite_type[type]}"

            #   cv2.putText(I, text, (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            #   print(count," ",fruite_type[type])

            # cv2.imshow("B", B)

    cv2.imshow("IR",IR)
    cv2.waitKey(0)
