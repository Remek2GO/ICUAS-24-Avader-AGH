import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # Read video frame

    path = "/home/vision/Documents/Repositorium/icuas24_avader/bags/"
    video_no = 1
    video_name = f"ICUAS_bag_{video_no}_camera_color_image_raw_compressed.mp4"

    video_path = path + video_name
    cap = cv2.VideoCapture(video_path)
    it = 0

    # video 1
    # 530
    # 850
    # 1650

    #video 2
    # 885
    # 1145
    # 1165
    while True:
        ret, frame = cap.read()
        print("number of frames: ", it)
        if ret:
            image = frame
            # change color space
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image3 = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            image4 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            image5 = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            image6 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            image7 = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

            # display all images using matplotlib
            # plt.figure(figsize=(20, 10))
            # plt.subplot(2, 4, 1)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.title('BGR')
            # plt.axis('off')

            # plt.subplot(2, 4, 2)
            # plt.imshow(image2)
            # plt.title('HSV') # Hue, Saturation, Value

            # plt.subplot(2, 4, 3)
            # plt.imshow(image3)
            # plt.title('HLS')
            # plt.axis('off')

            # plt.subplot(2, 4, 4)
            # plt.imshow(image4)
            # plt.title('LAB')
            # plt.axis('off')
            
            # plt.subplot(2, 4, 5)
            # plt.imshow(image5)
            # plt.title('LUV')
            # plt.axis('off')
            
            # plt.subplot(2, 4, 6)
            # plt.imshow(image6)
            # plt.title('YCrCb')
            # plt.axis('off')

            # plt.subplot(2, 4, 7)
            # plt.imshow(image7)
            # plt.title('XYZ')
            # plt.axis('off')

            # plt.show()

            cv2.imshow('frame', image)
        else:
            break

        key = cv2.waitKey(0)
        if key  & 0xFF == ord('q'):
            break

        if key & 0xFF == ord('r'):
            #save image

            #create folder
            
            if not os.path.exists(f"images/video_{video_no}"):
                os.makedirs(f"images/video_{video_no}")
            
            cv2.imwrite(f"images/video_{video_no}/image_{it}.png", image)
        it += 1

    cap.release()
    cv2.destroyAllWindows()
