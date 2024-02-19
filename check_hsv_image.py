import cv2
import numpy as np

h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255
light = 100




def update_image():
    global h_min, h_max, s_min, s_max, v_min, v_max, light
    imageCopy = img.copy()

    imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imageCopy)

    v = (v * (light / 100)).astype(np.uint8)
    v[v > 255] = 255
    v[v < 0] = 0
    
    final_hsv = cv2.merge((h, s, v))
    imageCopy = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
 
    hsv = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2HSV)
    

    mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
    maskStacked = np.stack([mask, mask, mask], axis=-1)
    merg = cv2.hconcat([imageCopy, maskStacked, cv2.bitwise_and(imageCopy, imageCopy, mask=mask)])
    cv2.imshow(windowName, merg)

def on_change(value, name):
    global h_min, h_max, s_min, s_max, v_min, v_max, light
    if name == 'lighting':
        light = value
    elif name == 'h_min':
        h_min = value
    elif name == 'h_max':
        h_max = value
    elif name == 's_min':
        s_min = value
    elif name == 's_max':
        s_max = value
    elif name == 'v_min':
        v_min = value
    elif name == 'v_max':
        v_max = value
    update_image()

img = cv2.imread('images_eval/2440_eval_color.png') # Tutaj zmienić nazwę pliku
img = cv2.resize(img, (320, 240))
windowName = 'image'

update_image()

cv2.createTrackbar('Lighting', windowName, 0, 100, lambda v: on_change(v, 'lighting'))
cv2.createTrackbar('H_min', windowName, 0, 179, lambda v: on_change(v, 'h_min'))
cv2.createTrackbar('H_max', windowName, 0, 179, lambda v: on_change(v, 'h_max'))
cv2.createTrackbar('S_min', windowName, 0, 255, lambda v: on_change(v, 's_min'))
cv2.createTrackbar('S_max', windowName, 0, 255, lambda v: on_change(v, 's_max'))
cv2.createTrackbar('V_min', windowName, 0, 255, lambda v: on_change(v, 'v_min'))
cv2.createTrackbar('V_max', windowName, 0, 255, lambda v: on_change(v, 'v_max'))

cv2.waitKey(0)
cv2.destroyAllWindows()