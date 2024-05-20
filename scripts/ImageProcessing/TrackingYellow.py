import cv2
from dataclasses import dataclass


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


def iou(box_a, box_b):
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


def initializecorrfilter(frame, x, y):
    tracker = cv2.TrackerCSRT.create()
    tracker.init(frame, (x-8, y-8, 16, 16))
    return tracker


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


tempObjects = []
objects = []
trackedObjects = []

ID = 1

# Read the video stream
video = cv2.VideoCapture('real_video/ICUAS_bag_1_camera_color_image_raw_compressed.mp4')

# Read the first frame
ret, frame = video.read()

# Display consecutive frames
while ret:
    # Display the frame
    cv2.imshow('Original', frame)

    image_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    image_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    image_hls_h, image_hls_l, image_hls_s = cv2.split(image_hls.astype(float))
    image_lab_l, image_lab_a, image_lab_b = cv2.split(image_lab.astype(float))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # image_tophat_hls_s = cv2.morphologyEx(image_hls_s, cv2.MORPH_TOPHAT, kernel)
    # image_tophat_lab_b = cv2.morphologyEx(image_lab_b, cv2.MORPH_TOPHAT, kernel)

    # Display the tophat images
    cv2.imshow('Tophat HLS S', image_hls_s.astype('uint8'))
    cv2.imshow('Tophat LAB B', image_lab_b.astype('uint8'))

    # Multiply the tophat images
    image_tophat_mult = cv2.multiply(image_hls_s, image_lab_b)
    image_tophat_mult = cv2.morphologyEx(image_tophat_mult, cv2.MORPH_TOPHAT, kernel)

    # Normalize the image
    image_tophat_mult_norm = cv2.normalize(image_tophat_mult, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display the red apples
    cv2.imshow('Tophat multiplication', image_tophat_mult_norm)

    # Binarize the image
    image_bin = image_tophat_mult > 20000

    # Median filter
    image_bin = cv2.medianBlur(image_bin.astype('uint8'), 5)

    # Display the binarized image
    cv2.imshow('Binary', image_bin.astype('uint8') * 255)

    # Track objects using the CSRT tracker
    for obj in trackedObjects.copy():
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
                trackedObjects.remove(obj)

    # Remove the tracked pixels from the image_bin
    for obj in trackedObjects:
        x, y, w, h = obj.bbox
        image_bin[y:y+h, x:x+w] = 0

    # Display the binarized image
    cv2.imshow('Tracked removed binary', image_bin.astype('uint8') * 255)

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
        obj = ObjectParameters(i, bbox, x_c, y_c, area)

        tempObjects.append(obj)

    # Set all objects to invisible
    for obj in objects:
        obj.visible = False

    # Compare the objects
    for tempObj in tempObjects:
        bestIoU = 0
        bestDist = 100
        bestDistTracked = 100
        bestIoUObj = None
        bestDistObj = None
        bestTrackedObj = None
        for obj in objects:
            iouVal = iou(tempObj.bbox, obj.bbox)
            if iouVal > bestIoU and iouVal > 0.5:
                bestIoU = iouVal
                bestIoUObj = obj
            dist = distance(tempObj.x, tempObj.y, obj.x, obj.y)
            if dist < bestDist:
                bestDist = dist
                bestDistObj = obj
        for obj in trackedObjects:
            if obj.visible:
                continue
            dist = distance(tempObj.x, tempObj.y, obj.x, obj.y)
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
            bestTrackedObj.tracker = initializecorrfilter(frame, int(bestTrackedObj.x), int(bestTrackedObj.y))
        else:
            obj = ObjectParameters(ID, tempObj.bbox, tempObj.x, tempObj.y, tempObj.area)
            objects.append(obj)
            ID += 1
            print('New object added with ID:', obj.id)

    tempObjects.clear()

    # Analyse non-visible objects
    for obj in objects.copy():
        if not obj.visible:
            obj.invisibleCounter += 1
            if obj.invisibleCounter > 50:
                objects.remove(obj)

    for obj in objects.copy():
        print('Object with ID:', obj.id, 'is visible for', obj.visibleCounter, 'frames')
        if obj.visibleCounter > 10:
            obj.tracker = initializecorrfilter(frame, int(obj.x), int(obj.y))
            trackedObjects.append(obj)
            objects.remove(obj)
            print('Object with ID:', obj.id, 'is now tracked')

    # Display the frame with potential objects
    for obj in objects:
        if not obj.visible:
            continue
        x, y, w, h = obj.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, str(obj.id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame with tracked objects
    for obj in trackedObjects:
        x, y, w, h = obj.bbox
        if obj.visible:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, str(obj.id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # else:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        #     cv2.putText(frame, str(obj.id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Objects', frame)

    # Mark bounding boxes
    # for i in range(1, num_labels):
    #     x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
    #         i, cv2.CC_STAT_HEIGHT]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Display the connected components
    # cv2.imshow('Connected Components', labels.astype('uint8') * 255)

    # contours, _ = cv2.findContours(image_bin.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Display the frame with bounding boxes
    # cv2.imshow('Bounding Box', frame)

    # Read the next frame
    ret, frame = video.read()

    # Break the loop when the video ends
    if not ret:
        break

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
