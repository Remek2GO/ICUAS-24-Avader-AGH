import cv2
import numpy as np

#TODO - doczytac jak maja wygladac te owoce i czy sa jakies ograniczenia.
#TODO - inne przestrzenie barw
#TODO - maska na smigla
#TODO - 

kernel_3 = np.ones((5, 5), np.uint8) 


fruite_type = ["apple","eggplant","citron"]


#TODO Warto by na te progi jeszcze raz zerknać
th = [[0,10,5,255,140,255], 
      [90,150,180,255,50,255],
      [20,35,175,255,90,255]
        ]


h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255
light = 100

# Przetwarzanie fragmentu sceny z owocami
def process_patch(patch):
    
    count = -1
    type = -1

    

    for k in range(0,3):
      t = th[k]
      mask = cv2.inRange(patch, (t[0], t[2], t[4]), (t[1], t[3], t[5]))
      #maskStacked = np.stack([mask, mask, mask], axis=-1)
      #merg = cv2.hconcat([imageCopy, maskStacked, cv2.bitwise_and(imageCopy, imageCopy, mask=mask)])
    
      #Filtracja
      mask = cv2.medianBlur(mask,3)  
      #TODO Duzy workaround
      #TODO MINA to by trzeba inaczej zrobic
      if ( k == 0 or k == 2):
        mask = cv2.erode(mask,kernel_3)
        mask = cv2.erode(mask,kernel_3)
      
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

      # Odejmujemy obiekt typu tlo
      num_labels = num_labels - 1
      #TODO Dodać analizę jakąś i zabezpiecznie
      #TODO Np jak obiekty sa zlaczone - to na razie jest dość ordynardny workaround



      #cv2.imshow("Test", mask)
      #cv2.waitKey(0)

      # Tu jest założenie, że nie ma krzaków mulitruit :)
      if (num_labels>0):
        count = num_labels
        type = k

    return (count, type)




# Przetwarzanie pojedycznej ramki 
# obraz RGB i odpowiadajaca mapa glebi
def process_frame(I,D):

  # Maska smigla
  dim = I.shape
  I = I[200:dim[0],:,:]
  D = D[200:dim[0],:]

  #imageCopy = I.copy()

  #imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2HSV)
  #h, s, v = cv2.split(imageCopy)

  #v = (v * (light / 100)).astype(np.uint8)
  #v[v > 255] = 255
  #v[v < 0] = 0
    
  #final_hsv = cv2.merge((h, s, v))
  #imageCopy = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

  hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

 
  #TODO Przenieść progi do gory
  # Wykrywanie bialych obszarow
  mask_white = cv2.inRange(hsv, (0, 0, 250), (5, 3, 255))

  #TODO Przenieść progi do gory
  # Wykrywanie zielonych obszarow
  mask_green = cv2.inRange(hsv, (50, 0, 45), (60, 180, 255))

  mask_wg = cv2.bitwise_or(mask_white,mask_green)

  cv2.imshow('Mask white and green',mask_wg)

  #Eksperyment
  kernel_1_25 = np.ones((1, 25), np.uint8)
  kernel_25_1 = np.ones((25, 1), np.uint8)



  #th_RGB_F = cv2.erode(th_RGB,kernel_1_7)
  th_RGB_F = cv2.dilate(mask_wg,kernel_1_25)
  th_RGB_F = cv2.dilate(th_RGB_F,kernel_25_1)

  th_RGB_F = cv2.medianBlur(th_RGB_F,7)

  cv2.imshow("Mask after median blur",th_RGB_F)


  # Mapa glebi
  # Konwersja do skali szarosci
  D = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY)

  #cv2.imshow("3D", D*20)


  # Maskowanie mapy głębi ....
  # Przrobienie na 0-1 (z 0 255)
  th_RGB_F =  np.uint8(th_RGB_F/255)
  result = cv2.multiply(th_RGB_F, D)

  


  cv2.imshow("3D_masking", result*10)

  # Analiza obrazu -- wyszukanie obiektów na pierwszym planie

  #TODO - trzeba to sprytniej - pytanie ile może być poziomów przed pierwszym planem
  #Ew. można to inaczej jakoś 

  # 0 pomijamy, bo tam na pewno nic ciekawego nie będzie.
  #TODO czy 10 to dobra wartosc
  for i in range(1,10):
      B = np.uint8(result == i)*255
      
      #Analiza:
      num_labels, labels, stats, centroids  = cv2.connectedComponentsWithStats(B)
      for i in range(1, num_labels):
          left, top, width, height, area = stats[i]
          # TODO Dodatkowe warunki a ten sprawdzic
          # TODO Korekcja ramki
          if area > 100*100:
            #TODO Polaczyc z dylatacja
            # Reczna korekcja (wynika z rozmiaru dylatacji)
            left = left + 12
            width = width - 24
            height = height - 12
            #cv2.rectangle(I, (left, top), (left + width, top + height), 255, 2)
            print(f"Component {i}: Area={area}, Centroid={centroids[i].astype(int)}")
            # Wycinek
            patch = hsv[top:top+height,left:left+width,:]
            count, type = process_patch(patch)
            if (count >0 ):
              text = f"{count} {fruite_type[type]}"

              cv2.putText(I, text, (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
              print(count," ",fruite_type[type])

      
      #cv2.imshow("B", B)
  
  # Wizualizacja
  cv2.imshow("Detection results", I)
  cv2.waitKey(0)     
      



if __name__ == "__main__":
    cases = ["A",'B','C','D','E','F','G']

    for c in range(0,len(cases)):

        I = cv2.imread(cases[c] + "_color.png")
        D = cv2.imread(cases[c] + "_depth.png")

        #TODO Tu pewnie jakieś logowanie wynikow
        process_frame(I,D)
























#Filtration

#th_RGB_F = cv2.erode(th_sum,kernel_7)
#th_RGB_F = cv2.dilate(th_RGB_F,kernel_7)


#Close some gaps (vertical, horizontal)
#kernel_1_11 = np.ones((1, 11), np.uint8) 



#Edge detection 
#th_RGB_FE = cv2. erode(th_RGB_F, kernel)

#edges = th_RGB_F - th_RGB_FE


#cv2.imshow("SEG White",th_RGB)
#cv2.imshow("SEG Green",thG_RGB)
#cv2.imshow("OR White, Green",th_sum)

#Hough
#lines_list =[]
#lines = cv2.HoughLinesP(
#            edges, # Input edge image
#            1, # Distance resolution in pixels
#            np.pi/180, # Angle resolution in radians
#            threshold=80, # Min number of votes for valid line
#            minLineLength=40, # Min allowed length of line
#            maxLineGap=60 # Max allowed gap between line for joining them
#            )

#for points in lines:
      # Extracted points nested in the list
 #   x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
 #   cv2.line(I,(x1,y1),(x2,y2),(0,0,255),4)
    # Maintain a simples lookup list for points
 #   lines_list.append([(x1,y1),(x2,y2)])


#cv2.imshow("th_R",thG_R)
#cv2.imshow("th_G",thG_G)
#cv2.imshow("th_B",thG_B)
#cv2.imshow("th_RGB",th_RGB_F)

#cv2.imshow("edges",edges)


# Detect green areas


#cv2.imshow("RGB",I)
#cv2.imshow("D",D)

cv2.waitKey(0)
cv2.destroyAllWindows()