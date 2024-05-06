import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # Read video frame

    path = "/home/vision/Documents/Repositorium/icuas24_avader/images/"


    # read only folders
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    print(folders)

    for folder in folders:
        #read images
        images = [f for f in os.listdir(os.path.join(path, folder)) if os.path.isfile(os.path.join(path, folder, f))]

        for image in images:
            image_path = os.path.join(path, folder, image)
            img = cv2.imread(image_path)

            image_name = image.split('.')[0]
            image_no = image_name.split('_')[1]

            if image_name.split('_')[0] != 'image':
                continue
            
            print(f"Image processing ... {image_name}")

            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            image3 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            image4 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            image5 = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            image6 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            image7 = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

            # display all images using matplotlib
            plt.figure(figsize=(20, 10))
            plt.subplot(2, 4, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('BGR')
            plt.axis('off')

            plt.subplot(2, 4, 2)
            plt.imshow(image2)
            plt.title('HSV') # Hue, Saturation, Value

            plt.subplot(2, 4, 3)
            plt.imshow(image3)
            plt.title('HLS')
            plt.axis('off')

            plt.subplot(2, 4, 4)
            plt.imshow(image4)
            plt.title('LAB')
            plt.axis('off')
            
            plt.subplot(2, 4, 5)
            plt.imshow(image5)
            plt.title('LUV')
            plt.axis('off')
            
            plt.subplot(2, 4, 6)
            plt.imshow(image6)
            plt.title('YCrCb')
            plt.axis('off')

            plt.subplot(2, 4, 7)
            plt.imshow(image7)
            plt.title('XYZ')
            plt.axis('off')

            # save figure
            plt.savefig(f"images/{folder}/figure_{image_no}.png")

            plt.show()

            


            cv2.imshow('image', img)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cv2.destroyAllWindows()


    

    # video 1
    # 530
    # 850
    # 1650

    #video 2
    # 885
    # 1145
    # 1165
   
    #     print("number of frames: ", it)
    #     if ret:
    #         image = frame
    #         # change color space
    #        



    #         cv2.imshow('frame', image)
    #     else:
    #         break

    #     key = cv2.waitKey(0)
    #     if key  & 0xFF == ord('q'):
    #         break

    #     if key & 0xFF == ord('r'):
    #         #save image

    #         #create folder
            
    #         if not os.path.exists(f"images/video_{video_no}"):
    #             os.makedirs(f"images/video_{video_no}")
            
    #         cv2.imwrite(f"images/video_{video_no}/image_{it}.png", image)
    #     it += 1

    # cap.release()
    # cv2.destroyAllWindows()
