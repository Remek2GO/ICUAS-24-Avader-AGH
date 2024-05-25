#!/usr/bin/env python
import cv2
from dataclasses import dataclass
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from scripts.ImageProcessing.dcf import initialize, predict


@dataclass
class ObjectParameters:
    id: int
    bbox: tuple
    x: float
    y: float
    area: int
    visible: bool = True
    invisibleCounter: int = 0
    visibleCounter: int = 0
    tracker: cv2.TrackerCSRT = None


class AnalyzeFrame:
    def __init__(self):
        self.tempObjects_yellow = []
        self.objects_yellow = []
        self.trackedObjects_yellow = []
        self.tempObjects_red = []
        self.objects_red = []
        self.trackedObjects_red = []

        self.ID_red = 1
        self.ID_yellow = 1
        self.frame = None
        self.yellow_count = 0
        self.red_count = 0
        self._svm_classifier = cv2.ml.SVM_load(
            "/root/sim_ws/src/icuas24_competition/scripts/ImageProcessing/svm_model_cv3.xml"
        )
        self._pattern_red = cv2.imread(
            "/root/sim_ws/src/icuas24_competition/scripts/ImageProcessing/pattern_red.png",
            cv2.IMREAD_GRAYSCALE,
        )
        self._pattern_yellow = cv2.imread(
            "/root/sim_ws/src/icuas24_competition/scripts/ImageProcessing//pattern_yellow.png",
            cv2.IMREAD_GRAYSCALE,
        )

        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

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

    def initializecorrfilter(self, frame, x, y):
        tracker = cv2.TrackerCSRT.create()
        tracker.init(frame, (x - 10, y - 10, 20, 20))
        return tracker

    def distance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_features(self, image):

        position = [0, 0, 20, 20]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Ai, Bi, G = initialize(self._pattern_red, position)
        response_red = predict(gray, Ai / Bi)

        Ai, Bi, G = initialize(self._pattern_yellow, position)
        response_yellow = predict(gray, Ai / Bi)

        features = np.concatenate(
            (image.flatten(), response_red.flatten(), response_yellow.flatten())
        )

        return features

    def svm_predict(self, predict_data, obj, objects):
        
        y_pred = 0
        color = (0, 0, 0)
        if predict_data is not None:
            if predict_data.shape[0] > 0 and predict_data.shape[1] > 0:
                predict_data = cv2.resize(predict_data, (20, 20))
                features = self.get_features(predict_data).astype(np.float32)

                _, y_pred = self._svm_classifier.predict(features.reshape(1, -1))
                y_pred = y_pred[0][0]

        if y_pred == 1: # red
            color = (0, 255, 255)
            objects.remove(obj)
        elif y_pred == 2: # yellow
            color = (0, 0, 255)
        elif y_pred == 3: # backgrond
            color = (0, 0, 0)
            objects.remove(obj)

        return y_pred, color

    def detect_yellow(self, frame):

        image_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        image_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        image_hls_h, image_hls_l, image_hls_s = cv2.split(image_hls.astype(float))
        image_lab_l, image_lab_a, image_lab_b = cv2.split(image_lab.astype(float))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # Convert the frame from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for yellow color
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # Multiply the tophat images
        image_tophat_mult = cv2.multiply(image_hls_s, image_lab_b)
        image_tophat_mult = cv2.morphologyEx(
            image_tophat_mult, cv2.MORPH_TOPHAT, kernel
        )

        # Normalize the image
        # image_tophat_mult_norm = cv2.normalize(image_tophat_mult, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Display the red apples
        # cv2.imshow('Tophat multiplication', image_tophat_mult_norm)

        # Binarize the image
        image_bin = image_tophat_mult > 20000
        image_bin = np.logical_or(np.logical_and(image_bin, mask), mask)

        # Median filter
        image_bin = cv2.medianBlur(image_bin.astype("uint8"), 5)

        # Track objects using the CSRT tracker
        for obj in self.trackedObjects_yellow.copy():
            success, bbox = obj.tracker.update(frame)
            if success:
                obj.bbox = bbox
                obj.x = bbox[0] + bbox[2] / 2
                obj.y = bbox[1] + bbox[3] / 2
                obj.visible = True
                obj.invisibleCounter = 0
                obj.visibleCounter += 1
            else:
                obj.visible = False
                obj.invisibleCounter += 1
                if obj.invisibleCounter > 50:
                    self.trackedObjects_yellow.remove(obj)

        # Remove the tracked pixels from the image_bin
        for obj in self.trackedObjects_yellow:
            x, y, w, h = obj.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            image_bin[y : y + h, x : x + w] = 0

        # Display the binarized image
        # cv2.imshow('Tracked removed binary', image_bin.astype('uint8') * 255)

        # Perform CCL and filter small objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image_bin.astype("uint8"), connectivity=8
        )

        # Intersection over Union tracking
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:
                continue

            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            bbox = (x, y, w, h)
            x_c = centroids[i, 0]
            y_c = centroids[i, 1]
            area = stats[i, cv2.CC_STAT_AREA]

            # Initialize the object parameters
            obj = ObjectParameters(i, bbox, x_c, y_c, area)

            self.tempObjects_yellow.append(obj)

        # Set all objects to invisible
        for obj in self.objects_yellow:
            obj.visible = False

        # Compare the objects
        for tempObj in self.tempObjects_yellow:
            bestIoU = 0
            bestDist = 100
            bestDistTracked = 100
            bestIoUObj = None
            bestDistObj = None
            bestTrackedObj = None
            for obj in self.objects_yellow:
                iouVal = self.iou(tempObj.bbox, obj.bbox)
                if iouVal > bestIoU and iouVal > 0.5:
                    bestIoU = iouVal
                    bestIoUObj = obj
                dist = self.distance(tempObj.x, tempObj.y, obj.x, obj.y)
                if dist < bestDist:
                    bestDist = dist
                    bestDistObj = obj
            for obj in self.trackedObjects_yellow:
                if obj.visible:
                    continue
                dist = self.distance(tempObj.x, tempObj.y, obj.x, obj.y)
                if dist < bestDistTracked:
                    bestDistTracked = dist
                    bestTrackedObj = obj

            if bestIoUObj is not None:
                bestIoUObj.bbox = tempObj.bbox
                bestIoUObj.x = tempObj.x
                bestIoUObj.y = tempObj.y
                bestIoUObj.area = tempObj.area
                bestIoUObj.visible = True
                bestIoUObj.invisibleCounter = 0
                bestIoUObj.visibleCounter += 1
            elif bestDistObj is not None:
                bestDistObj.bbox = tempObj.bbox
                bestDistObj.x = tempObj.x
                bestDistObj.y = tempObj.y
                bestDistObj.area = tempObj.area
                bestDistObj.visible = True
                bestDistObj.invisibleCounter = 0
                bestDistObj.visibleCounter += 1
            elif bestTrackedObj is not None:
                # Additional check based on correlation should be added here
                bestTrackedObj.bbox = tempObj.bbox
                bestTrackedObj.x = tempObj.x
                bestTrackedObj.y = tempObj.y
                bestTrackedObj.area = tempObj.area
                bestTrackedObj.visible = True
                bestTrackedObj.invisibleCounter = 0
                bestTrackedObj.visibleCounter += 1
                bestTrackedObj.tracker = self.initializecorrfilter(
                    frame, int(bestTrackedObj.x), int(bestTrackedObj.y)
                )
            else:
                obj = ObjectParameters(
                    self.ID_yellow, tempObj.bbox, tempObj.x, tempObj.y, tempObj.area
                )
                self.objects_yellow.append(obj)
                self.ID_yellow += 1
                # print('New object added with ID:', obj.id)

        self.tempObjects_yellow.clear()

        # Analyse non-visible objects
        for obj in self.objects_yellow.copy():
            if not obj.visible:
                obj.invisibleCounter += 1
                if obj.invisibleCounter > 50:
                    self.objects_yellow.remove(obj)

        for obj in self.objects_yellow.copy():
            # print('Object with ID:', obj.id, 'is visible for', obj.visibleCounter, 'frames')
            if obj.visibleCounter > 10:
                obj.tracker = self.initializecorrfilter(frame, int(obj.x), int(obj.y))
                self.trackedObjects_yellow.append(obj)
                self.objects_yellow.remove(obj)
                # print('Object with ID:', obj.id, 'is now tracked')

        # Display the frame with potential objects
        for obj in self.objects_yellow:
            if not obj.visible:
                continue
            x, y, w, h = obj.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            predict_data = frame[y : y + h, x : x + w]
            y_pred, color = self.svm_predict(predict_data, obj, self.objects_yellow)

            if obj.visible and y_pred == 2:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    str(obj.id),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Display the frame with tracked objects
        for obj in self.trackedObjects_yellow:
            x, y, w, h = obj.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            predict_data = frame[y : y + h, x : x + w]
            y_pred, color = self.svm_predict(predict_data, obj, self.trackedObjects_yellow)

            if obj.visible and y_pred == 2:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                cv2.putText(
                    frame,
                    str(obj.id),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )


        return frame

    def detect_red(self, frame):

        image_luv = cv2.cvtColor(frame, cv2.COLOR_BGR2Luv)
        image_luv_l, image_luv_u, image_luv_v = cv2.split(image_luv)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        image_tophat_luv_u = cv2.morphologyEx(image_luv_u, cv2.MORPH_TOPHAT, kernel)

        # Normalize the image
        image_tophat_luv_u_norm = cv2.normalize(
            image_tophat_luv_u, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )

        # Binarize the image
        image_bin = image_tophat_luv_u > 10

        # Median filter
        image_bin = cv2.medianBlur(image_bin.astype("uint8"), 5)

        # Track objects using the CSRT tracker
        for obj in self.trackedObjects_red.copy():
            success, bbox = obj.tracker.update(frame)
            if success:
                obj.bbox = bbox
                obj.x = bbox[0] + bbox[2] / 2
                obj.y = bbox[1] + bbox[3] / 2
                obj.visible = True
                obj.invisibleCounter = 0
                obj.visibleCounter += 1
            else:
                obj.visible = False
                obj.invisibleCounter += 1
                if obj.invisibleCounter > 50:
                    self.trackedObjects_red.remove(obj)

        # Remove the tracked pixels from the image_bin
        for obj in self.trackedObjects_red:
            x, y, w, h = obj.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            image_bin[y : y + h, x : x + w] = 0

        # Perform CCL and filter small objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image_bin.astype("uint8"), connectivity=8
        )

        # Intersection over Union tracking
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:
                continue

            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            bbox = (x, y, w, h)
            x_c = centroids[i, 0]
            y_c = centroids[i, 1]
            area = stats[i, cv2.CC_STAT_AREA]

            # Initialize the object parameters
            obj = ObjectParameters(i, bbox, x_c, y_c, area)

            self.tempObjects_red.append(obj)

        # Set all objects to invisible
        for obj in self.objects_red:
            obj.visible = False

        # Compare the objects
        for tempObj in self.tempObjects_red:
            bestIoU = 0
            bestDist = 100
            bestDistTracked = 100
            bestIoUObj = None
            bestDistObj = None
            bestTrackedObj = None
            for obj in self.objects_red:
                iouVal = self.iou(tempObj.bbox, obj.bbox)
                if iouVal > bestIoU and iouVal > 0.5:
                    bestIoU = iouVal
                    bestIoUObj = obj
                dist = self.distance(tempObj.x, tempObj.y, obj.x, obj.y)
                if dist < bestDist:
                    bestDist = dist
                    bestDistObj = obj
            for obj in self.trackedObjects_red:
                if obj.visible:
                    continue
                dist = self.distance(tempObj.x, tempObj.y, obj.x, obj.y)
                if dist < bestDistTracked:
                    bestDistTracked = dist
                    bestTrackedObj = obj

            if bestIoUObj is not None:
                bestIoUObj.bbox = tempObj.bbox
                bestIoUObj.x = tempObj.x
                bestIoUObj.y = tempObj.y
                bestIoUObj.area = tempObj.area
                bestIoUObj.visible = True
                bestIoUObj.invisibleCounter = 0
                bestIoUObj.visibleCounter += 1
            elif bestDistObj is not None:
                bestDistObj.bbox = tempObj.bbox
                bestDistObj.x = tempObj.x
                bestDistObj.y = tempObj.y
                bestDistObj.area = tempObj.area
                bestDistObj.visible = True
                bestDistObj.invisibleCounter = 0
                bestDistObj.visibleCounter += 1
            elif bestTrackedObj is not None:
                # Additional check based on correlation should be added here
                bestTrackedObj.bbox = tempObj.bbox
                bestTrackedObj.x = tempObj.x
                bestTrackedObj.y = tempObj.y
                bestTrackedObj.area = tempObj.area
                bestTrackedObj.visible = True
                bestTrackedObj.invisibleCounter = 0
                bestTrackedObj.visibleCounter += 1
                bestTrackedObj.tracker = self.initializecorrfilter(
                    frame, int(bestTrackedObj.x), int(bestTrackedObj.y)
                )

            else:
                obj = ObjectParameters(
                    self.ID_red, tempObj.bbox, tempObj.x, tempObj.y, tempObj.area
                )
                self.objects_red.append(obj)
                self.ID_red += 1
                # print("New object added with ID:", obj.id)

        self.tempObjects_red.clear()

        # Analyse non-visible objects
        for obj in self.objects_red.copy():
            if not obj.visible:
                obj.invisibleCounter += 1
                if obj.invisibleCounter > 50:
                    self.objects_red.remove(obj)

        for obj in self.objects_red.copy():
            if obj.visibleCounter > 10:
                obj.tracker = self.initializecorrfilter(frame, int(obj.x), int(obj.y))
                self.trackedObjects_red.append(obj)
                self.objects_red.remove(obj)

        # Display the frame with potential objects
        for obj in self.objects_red:
            if not obj.visible:
                continue
            x, y, w, h = obj.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            predict_data = frame[y : y + h, x : x + w]
            y_pred, color = self.svm_predict(predict_data, obj, self.objects_red)

            if obj.visible and y_pred == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    str(obj.id),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Display the frame with tracked objects
        for obj in self.trackedObjects_red:
            x, y, w, h = obj.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            predict_data = frame[y : y + h, x : x + w]
            y_pred, color = self.svm_predict(predict_data, obj, self.trackedObjects_red)

            if obj.visible and y_pred == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(
                    frame,
                    str(obj.id),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        return frame
