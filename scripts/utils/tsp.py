#!/usr/bin/env python
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import copy
import fast_tsp
from .positions import (
    PointOfInterest,
    get_distance,
    POINTS_OF_INTEREST,
    create_distance_matrix,
)
from typing import List

NUM_CHEBYSHEV_BETWEEN_BEDS = 15
NUM_TIMES_REPEAT_POINT = 5
NUM_CHEBYSHEV_CHANGE_BEDS = 5
NUM_CHEBYSHEV_ALONG_BEDS_TO_START = 10
NUM_CHEBYSHEV_BACK_TO_START = 10

START_POINT = [1, 1, 1, 0, 0, 0]
END_POINT = [1, 1, 2, 0, 0, 0]


class TSP:
    def __init__(self, matrix_of_distances):
        self.matrix_of_distances = matrix_of_distances

    def solve(self):
        tour = fast_tsp.find_tour(self.matrix_of_distances)
        return tour


def intermediate_points(points_to_visit):
    new_path = []
    previous_point = None

    # Przejscie na wysokosc pierwszego regalu - przemieszczenie w osi X

    start_point = START_POINT

    current_point = [
        points_to_visit[0][0],
        2,
        points_to_visit[0][2],
        0,
        0,
        points_to_visit[0][5],
    ]
    new_points = generate_intermediate_points(
        start_point, current_point, NUM_CHEBYSHEV_BETWEEN_BEDS
    )
    # new_path += new_points
    # new_path.append(current_point)

    points_to_visit.insert(0, current_point)

    for i, point in enumerate(points_to_visit):
        if not previous_point:
            current_point = copy.deepcopy(point)
            previous_point = current_point
            # new_path.append(current_point)
        else:
            if (
                point[0] == previous_point[0]
            ):  # jezeli punkty maja taka sama wspolrzedna x
                current_point = copy.deepcopy(point)

                # chebyshev nodes - interpolacja
                new_points = generate_intermediate_points(
                    previous_point,
                    current_point,
                    NUM_CHEBYSHEV_BETWEEN_BEDS,
                    False,
                )
                new_path += new_points

                # powtorzenie punktu docelowego n razy
                for i in range(NUM_TIMES_REPEAT_POINT):
                    new_path.append(current_point)

                previous_point = current_point

            else:  # jezeli punkty maja rozne wspolrzedne x - zmiana regalu

                ## 1. wyznaczenie punktow poza regal dla x rownego odwiedzonego punktu
                edge_previous_point = copy.copy(previous_point)
                current_point = copy.deepcopy(point)

                edge_previous_point[5] = current_point[
                    5
                ]  # obrot taki jak nastepny punkt
                if abs(previous_point[1] - 25) > abs(previous_point[1] - 2):
                    edge_previous_point[1] = 2
                else:
                    edge_previous_point[1] = 25

                # chebyshev nodes - interpolacja
                new_points = generate_intermediate_points(
                    previous_point,
                    edge_previous_point,
                    NUM_CHEBYSHEV_CHANGE_BEDS,
                    False,
                )
                new_path += new_points

                new_path.append(edge_previous_point)

                ## 2. wyznaczenie punktow wzdluz krotszego boku regalu
                edge_current_point = copy.copy(edge_previous_point)
                edge_current_point[2] = copy.copy(current_point[2])
                edge_current_point[0] = copy.copy(current_point[0])

                # chebyshev nodes - interpolacja
                new_points = generate_intermediate_points(
                    edge_previous_point,
                    edge_current_point,
                    NUM_CHEBYSHEV_CHANGE_BEDS,
                )
                new_path += new_points

                new_path.append(edge_current_point)

                ## 3. wyznaczenie punktow poza regal dla x rownego docelowej pozycji

                # chebyshev nodes - interpolacja
                new_points = generate_intermediate_points(
                    edge_current_point,
                    current_point,
                    NUM_CHEBYSHEV_CHANGE_BEDS,
                )
                new_path += new_points

                # powtorzenie punktu docelowego n razy
                for i in range(NUM_TIMES_REPEAT_POINT):
                    new_path.append(current_point)

                # Aktualizacja poprzedniego punktu
                previous_point = current_point

    # Powrot do punktu startowego - przemieszczenie w osi Y poza regal
    current_point = copy.copy(point)
    current_point[1] = 2

    # chebyshev nodes - interpolacja
    new_points = generate_intermediate_points(
        point, current_point, NUM_CHEBYSHEV_ALONG_BEDS_TO_START
    )
    new_path += new_points

    new_path.append(current_point)

    # Powrot do punktu startowego - przemieszczenie w osi X do punktu startowego
    end_point = END_POINT
    new_points = generate_intermediate_points(
        current_point, end_point, NUM_CHEBYSHEV_BACK_TO_START
    )
    new_path += new_points
    new_path.append(end_point)

    # if DEBUG_MODE:
    # print("Punkty z pośrednimi: ")
    # for item in new_path:
    #     print(item)

    return new_path


def chebyshev_nodes(n, a, b):
    xk = [np.cos((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)][::-1]
    # od -1 do 1

    # przeksztalcenie afiniczne do zadanego przedzialu (a, b)
    xk_norm = np.zeros((n, len(a)))
    for i in range(n):
        xk_norm[i] = xk[i] * (b - a) / 2 + (a + b) / 2

    return xk_norm


def generate_intermediate_points(previous_point, point, n, change_bed=True):
    a = np.array(previous_point)
    b = np.array(point)

    if np.all(a[:3] == b[:3]):
        n = n // 2

    xk_norm = chebyshev_nodes(n, a, b)
    # xk_norm_angle = chebyshev_nodes(n, a, b)

    if not change_bed:
        mid_point = len(xk_norm) // 3
        for i in range(mid_point):
            xk_norm[i][3:] = previous_point[3:]

    new_points = []
    for i in range(len(xk_norm)):
        interpoint = copy.deepcopy(previous_point)
        interpoint = xk_norm[i]
        # interpoint[:3] = xk_norm[i]
        # interpoint[3:] = xk_norm_angle[i]
        new_points.append(interpoint)

    return new_points


def prepare_points(points_from_drone):
    example_of_points = []
    for x in points_from_drone:
        example_of_points.append(x * 2)
        example_of_points.append(x * 2 + 1)
    return example_of_points


def get_photo_poses(tour, points_indexes):
    setpoints = []
    for i in range(len(tour)):
        setpoints.append(
            POINTS_OF_INTEREST[points_indexes[tour[i]] // 2][
                points_indexes[tour[i]] % 2
            ]
        )
    return setpoints[1:]


def start(AREAS_FROM_DRONE):
    points_indexes = prepare_points(AREAS_FROM_DRONE)
    print(f"Points indexes: {points_indexes}")
    distance_matrix = create_distance_matrix(points_indexes)
    distance_matrix_int = (distance_matrix * 100.0).astype(int)
    tsp = TSP(distance_matrix_int)
    tour = tsp.solve()
    print(f"Tour: {tour}")

    points_indexes = [0, *points_indexes]
    sorted_points = [points_indexes[i] for i in tour]
    print(f"Sorted points: {sorted_points}")

    target_points = get_photo_poses(tour, points_indexes)
    print(f"Target points: {target_points}")

    setpoints = intermediate_points(target_points)

    length = fast_tsp.compute_cost(tour, distance_matrix_int)
    print(f"Tour length: {length/100.0}")

    return setpoints, [points_indexes[i] for i in tour]


if __name__ == "__main__":
    AREAS_FROM_DRONE = [13, 15, 17, 19, 25]
    setpoints, photo_poses = start(AREAS_FROM_DRONE)

    print(photo_poses)
