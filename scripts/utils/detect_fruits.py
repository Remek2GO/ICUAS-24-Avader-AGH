import cv2
import numpy as np
from .plant_bed import PlantSide
from .types import PlantType
from typing import List, Tuple

# TODO - doczytac jak maja wygladac te owoce i czy sa jakies ograniczenia.
# TODO - inne przestrzenie barw
# TODO - maska na smigla
# TODO -

kernel_3 = np.ones((5, 5), np.uint8)


fruite_type = ["apple", "eggplant", "citron"]


# TODO Warto by na te progi jeszcze raz zerknać
th = [
    [0, 10, 5, 255, 140, 255],
    [90, 150, 180, 255, 50, 255],
    [20, 35, 175, 255, 90, 255],
]


h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255
light = 100


def process_patch(patch):
    count = -1
    type = -1
    centers = []
    for k in range(0, 3):
        t = th[k]
        mask = cv2.inRange(patch, (t[0], t[2], t[4]), (t[1], t[3], t[5]))
        # maskStacked = np.stack([mask, mask, mask], axis=-1)
        # merg = cv2.hconcat([imageCopy, maskStacked, cv2.bitwise_and(imageCopy, imageCopy, mask=mask)])

        # Filtracja
        mask = cv2.medianBlur(mask, 3)
        # TODO Duzy workaround
        # TODO MINA to by trzeba inaczej zrobic
        # if k == 0 or k == 2:
        #    mask = cv2.erode(mask, kernel_3)
        #    mask = cv2.erode(mask, kernel_3)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        fruit_count = 0
        for i in range(1, num_labels):
            left, top, width, height, area = stats[i]
            bbox_area = width * height
            abbox = area / bbox_area
            print(bbox_area, "|", area, "|", area / bbox_area)
            if bbox_area > 200:
                mask_small = mask[top : top + height, left : left + width]
                dist = cv2.distanceTransform(mask_small, cv2.DIST_L2, 3)
                cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

                # Threshold
                ret, dist_th = cv2.threshold(dist, 0.70, 255, cv2.THRESH_BINARY)
                dist_th = np.uint8(dist_th)

                #   cv2.imshow("Dist T", dist_th)
                #   cv2.waitKey(0)

                num_labels_small, labels_small, stats_small, centroids_small = (
                    cv2.connectedComponentsWithStats(dist_th)
                )
                fruit_count = fruit_count + num_labels_small - 1

                for ii in range(1, num_labels_small):
                    # TODO Do sprawdzenia
                    centers.append(
                        (
                            (centroids_small[ii][0] + height) / patch.shape[0],
                            (centroids_small[ii][1] + width) / patch.shape[1],
                        )
                    )
                    # Tu wyliczamy dwa centoridy - robimy erozję dopóki nam się to nie rodzieli

        # cv2.imshow("Test", mask)
        # cv2.waitKey(0)

        # Tu jest założenie, że nie ma krzaków mulitruit :)
        if fruit_count > 0:
            count = fruit_count
            type = k

    centers = np.array(centers)

    return (count, type, centers)


def preprocess_color_image(I):
    dim = I.shape
    I = I[200 : dim[0], :, :]
    return I


def preprocess_depth_image(D):
    dim = D.shape
    D = D[200 : dim[0], :]
    return D


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

    #   cv2.imshow("Mask after median blur",th_RGB_F)
    return th_RGB_F


def get_patches(masked_result):
    patches = []
    for i in range(1, 10):
        B = np.uint8(masked_result == i) * 255

        # Analiza:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(B)
        for i in range(1, num_labels):
            left, top, width, height, area = stats[i]
            # TODO Dodatkowe warunki a ten sprawdzic
            # TODO Korekcja ramki
            if area > 100 * 100:
                # TODO Polaczyc z dylatacja
                # Reczna korekcja (wynika z rozmiaru dylatacji)
                left = left + 12
                width = width - 24
                height = height - 12
                # cv2.rectangle(I, (left, top), (left + width, top + height), 255, 2)
                print(
                    f"Component {i}: Area={area}, Centroid={centroids[i].astype(int)}"
                )
                # Wycinek
                patches.append((top, top + height, left, left + width))

                # if (count >0 ):
                #   text = f"{count} {fruite_type[type]}"

                #   cv2.putText(I, text, (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                #   print(count," ",fruite_type[type])

        # cv2.imshow("B", B)
    return patches


# Przetwarzanie pojedycznej ramki
# obraz RGB i odpowiadajaca mapa glebi
def process_frame(I, D, debug=False) -> Tuple[List[PlantSide], int]:

    # Maska smigla
    I = preprocess_color_image(I)
    D = preprocess_depth_image(D)

    hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    plants_mask = mask_plants(hsv)

    D = cv2.cvtColor(D, cv2.COLOR_BGR2GRAY)
    result = cv2.multiply(plants_mask, D)

    #   cv2.imshow("3D_masking", result*10)

    # Analiza obrazu -- wyszukanie obiektów na pierwszym planie

    # TODO - trzeba to sprytniej - pytanie ile może być poziomów przed pierwszym planem
    # Ew. można to inaczej jakoś

    # 0 pomijamy, bo tam na pewno nic ciekawego nie będzie.
    # TODO czy 10 to dobra wartosc
    patches = get_patches(result)

    patches = sorted(patches, key=lambda x: x[2])

    plant_sides = []
    type = -1

    for p in patches:
        patch = hsv[p[0] : p[1], p[2] : p[3], :]
        # cv2.imshow("Patch", patch)
        # cv2.waitKey(0)
        count, type, centers = process_patch(patch)
        plant_type = None
        if type == 0:
            plant_type = PlantType.TOMATO
        elif type == 1:
            plant_type = PlantType.EGGPLANT
        elif type == 2:
            plant_type = PlantType.PEPPER
        plant_side = PlantSide(
            fruit_count=count, fruit_position=centers, fruit_type=plant_type
        )

        if count > 0:
            text = f"{count} {fruite_type[type]}"
            cv2.putText(
                I,
                text,
                (p[2], p[0] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        cv2.rectangle(I, (p[2], p[0]), (p[3], p[1]), 255, 2)

        plant_sides.append(plant_side)
        print(count, " ", fruite_type[type], " ")

    if debug:
        cv2.imshow("Detection results", I)
        cv2.startWindowThread()
        cv2.waitKey(1)

    return plant_sides, type


if __name__ == "__main__":
    cases = ["A", "B", "C", "D", "E", "F", "G"]

    for c in range(0, len(cases)):

        I = cv2.imread(cases[c] + "_color.png")
        D = cv2.imread(cases[c] + "_depth.png")

        # TODO Tu pewnie jakieś logowanie wynikow
        process_frame(I, D)
