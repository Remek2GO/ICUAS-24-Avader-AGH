#!/usr/bin/env python
import cv2
from dataclasses import dataclass


@dataclass
class ObjectParameters:
    id: int
    bbox: tuple
    x: float
    y: float
    area: int
    visible: bool
    invisibleCounter: int
    # visibleCounter: int

class AnalyzeFrame:
    def __init__(self):
        self.tempObjects = []
        self.objects = []
        self.ID = 1
        self.frame = None
        self.yellow_count = 0
        self.red_count = 0
    
    def iou(self, box_a, box_b):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        yb = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

        # Compute the area of intersection rectangle
        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        box_a_area = box_a[2] * box_a[3]
        box_b_area = box_b[2] * box_b[3]

        # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction +
        # ground-truth areas - the intersection area
        iou_val = inter_area / float(box_a_area + box_b_area - inter_area)

        # Return the intersection over union value
        return iou_val


    def distance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def analizer(self, frame):
        image_luv = cv2.cvtColor(frame, cv2.COLOR_BGR2Luv)
        image_luv_l, image_luv_u, image_luv_v = cv2.split(image_luv)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        image_tophat_luv_u = cv2.morphologyEx(image_luv_u, cv2.MORPH_TOPHAT, kernel)
        # image_tophat_luv_u = cv2.GaussianBlur(image_tophat_luv_u, (17, 17), 0)

        # Print max value of the image
        # print(image_tophat_luv_u.max())

        # Normalize the image
        image_tophat_luv_u_norm = cv2.normalize(image_tophat_luv_u, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Display the red apples
        # cv2.imshow('Tophat LUV U', image_tophat_luv_u_norm)

        # Binarize the image
        image_bin = image_tophat_luv_u > 10

        # Median filter
        image_bin = cv2.medianBlur(image_bin.astype('uint8'), 5)

        # Display the binarized image
        # cv2.imshow('Binary', image_bin.astype('uint8') * 255)

        # Perform CCL and filter small objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_bin.astype('uint8'), connectivity=8)

        # Intersection over Union tracking
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:
                continue

            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
                i, cv2.CC_STAT_HEIGHT]
            bbox = (x, y, w, h)
            x_c = centroids[i, 0]
            y_c = centroids[i, 1]
            area = stats[i, cv2.CC_STAT_AREA]

            # Initialize the object parameters
            obj = ObjectParameters(i, bbox, x_c, y_c, area, True, 0)

            self.tempObjects.append(obj)

        # Set all objects to invisible
        for obj in self.objects:
            obj.visible = False

        # Compare the objects
        for tempObj in self.tempObjects:
            bestIoU = 0
            bestDistance = 100
            bestIoUObj = None
            bestDistObj = None
            for obj in self.objects:
                iouVal = self.iou(tempObj.bbox, obj.bbox)
                if iouVal > bestIoU and iouVal > 0.5:
                    bestIoU = iouVal
                    bestIoUObj = obj
                dist = self.distance(tempObj.x, tempObj.y, obj.x, obj.y)
                if dist < bestDistance:
                    bestDistance = dist
                    bestDistObj = obj

            if bestIoUObj is not None:
                bestIoUObj.bbox = tempObj.bbox
                bestIoUObj.x = tempObj.x
                bestIoUObj.y = tempObj.y
                bestIoUObj.area = tempObj.area
                bestIoUObj.visible = True
                bestIoUObj.invisibleCounter = 0
            elif bestDistObj is not None:
                bestDistObj.bbox = tempObj.bbox
                bestDistObj.x = tempObj.x
                bestDistObj.y = tempObj.y
                bestDistObj.area = tempObj.area
                bestDistObj.visible = True
                bestDistObj.invisibleCounter = 0
            else:
                obj = ObjectParameters(self.ID, tempObj.bbox, tempObj.x, tempObj.y, tempObj.area, True, 0)
                self.objects.append(obj)
                self.ID += 1
                self.red_count += 1
                print('New object added with ID:', obj.id)

        self.tempObjects.clear()

        # Analyse non-visible objects
        for obj in self.objects:
            if not obj.visible:
                obj.invisibleCounter += 1
                if obj.invisibleCounter > 50:
                    self.objects.remove(obj)

        # Display the frame with bounding boxes and IDs
        for obj in self.objects:
            if not obj.visible:
                continue
            x, y, w, h = obj.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(obj.id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv2.imshow('Bounding Box', frame)
        self.frame = frame
        return frame