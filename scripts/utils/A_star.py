#!/usr/bin/env python
"""Implementation of the A* algorithm to find the path to follow."""

import copy
import numpy as np

# from dataclasses import dataclass
from scripts.utils import positions
from scripts.utils.types import Setpoint

DEBUG_MODE = False


class A_star:
    def __init__(self, start_location, end_location, heuristic) -> None:
        self.setpoints = {}
        self.start_location = start_location
        self.end_location = end_location
        self.heuristic_distances = heuristic
        self.open_list = []
        self.close_list = []
        self.path = []

        self.num_chebyshev_between_beds = 10
        self.num_chebyshev_back_to_start = 5
        self.num_chebyshev_along_beds_to_start = 5
        self.num_chebyshev_change_beds = 3

    def add_setpoint(self, number, setpoint: Setpoint):
        self.setpoints.update({number: setpoint})

    def MST(self, setpoint):

        # print("MST")
        open_list_copy = copy.copy(self.open_list)

        # open_list_copy.remove(setpoint) # nie usuwanie wybranego punktu
        N = open_list_copy
        # print("Open list copy (update): ", open_list_copy)
        selected_node = np.zeros(len(N))
        no_edge = 0
        selected_node[0] = True
        sum_weights = 0
        a = 0
        b = 0
        while no_edge < len(N) - 1:

            minimum = np.inf
            a = 0
            b = 0
            for m in N:
                # print("M: ", m)
                # print("Selected node: ", selected_node[N.index(m)], "N index: ", N.index(m))
                if selected_node[N.index(m)]:
                    for n in N:
                        if not selected_node[N.index(n)]:
                            if minimum > self.heuristic_distances[m][n]:
                                minimum = self.heuristic_distances[m][n]
                                a = m
                                b = n
            # print("A: ", a, "B: ", b)

            sum_weights += self.heuristic_distances[a][b]
            selected_node[N.index(b)] = True
            no_edge += 1
            # print("Sum weights: ", sum_weights)

        # Dodanie dystansu od ostatniego punktu do punktu poczatkowego/docelowego
        sum_weights += self.heuristic_distances[b][0]

        return sum_weights

    def search_path(self, LOCALIZATION):
        end_found = False
        current_node = 0
        g_cost = 0

        for setpoint in self.setpoints.keys():
            self.open_list.append(setpoint)
        # print("Open list: ", self.open_list)

        self.open_list.remove(current_node)
        self.open_list.remove(current_node + 1)

        while not end_found:
            f_cost = {}
            # print(self.heuristic_distances)
            if DEBUG_MODE:
                print("\n Aktualna pozycja: ", current_node / 2, " \n")
            if len(self.open_list) != 1:
                # print("Open list: ", self.open_list)
                for setpoint in self.open_list:

                    h_cost = self.MST(setpoint)

                    f_cost.update(
                        {
                            setpoint: g_cost
                            + self.heuristic_distances[current_node][setpoint]
                            + h_cost
                        }
                    )
                    if DEBUG_MODE:
                        print(
                            setpoint / 2,
                            " - ",
                            g_cost,
                            self.heuristic_distances[current_node][setpoint],
                            h_cost,
                        )

                next_node = min(f_cost, key=f_cost.get)
                self.path.append(next_node)
                self.open_list.remove(next_node)
                self.close_list.append(next_node)
                g_cost += self.heuristic_distances[current_node][next_node]
                # print("Next node: ", next_node)
                # print("g_cost: ", g_cost)
                # print("Heuristic: ",self.heuristic_distances[current_node][next_node])
                current_node = next_node
            else:
                next_node = self.open_list[0]
                self.path.append(next_node)
                self.close_list.append(next_node)
                self.open_list.remove(next_node)
                current_node = next_node

            if self.open_list == []:
                end_found = True
                path = self.intermediate_points(self.path, LOCALIZATION)
                return path

    def intermediate_points(self, areas_to_visit, LOCALIZATION):
        new_path = []
        previous_point = None

        points_to_visit = []
        for elem in areas_to_visit:
            points_to_visit.append(LOCALIZATION[np.floor(elem / 2)][elem % 2])

        if DEBUG_MODE:
            print("Punkty bez pośrednich :", points_to_visit)

        # new_path.append([points_to_visit[0][0], 2, points_to_visit[0][2], 0, 0, 0])
        for i, point in enumerate(points_to_visit):
            if not previous_point:
                current_point = copy.deepcopy(point)
                previous_point = current_point
                new_path.append(current_point)
            else:
                if (
                    point[0] == previous_point[0]
                ):  # jezeli punkty maja taka sama wspolrzedna x
                    current_point = copy.deepcopy(point)

                    # chebyshev nodes - interpolacja
                    new_points = generate_intermediate_points(
                        previous_point, current_point, self.num_chebyshev_between_beds
                    )
                    new_path += new_points

                    new_path.append(current_point)
                    previous_point = current_point

                else:  # jezeli punkty maja rozne wspolrzedne x - zmiana regalu

                    ## 1. wyznaczenie punktow poza regal dla x rownego odwiedzonego punktu
                    edge_previous_point = copy.copy(previous_point)
                    current_point = copy.deepcopy(point)

                    if i + 1 < len(points_to_visit):
                        # new_previous_point[5] =  np.pi / 2
                        edge_previous_point[5] = points_to_visit[i + 1][
                            5
                        ]  # obrot taki jak nastepny punkt
                    else:
                        edge_previous_point[5] = 0  # poczatkowa wartosc obrotu yaw

                    if abs(previous_point[1] - 25) > abs(previous_point[1] - 2):
                        edge_previous_point[1] = 2
                    else:
                        edge_previous_point[1] = 25

                    # chebyshev nodes - interpolacja
                    new_points = generate_intermediate_points(
                        previous_point,
                        edge_previous_point,
                        self.num_chebyshev_change_beds,
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
                        self.num_chebyshev_change_beds,
                    )
                    new_path += new_points

                    new_path.append(edge_current_point)

                    ## 3. wyznaczenie punktow poza regal dla x rownego docelowej pozycji

                    # chebyshev nodes - interpolacja
                    new_points = generate_intermediate_points(
                        edge_current_point,
                        current_point,
                        self.num_chebyshev_change_beds,
                    )
                    new_path += new_points

                    new_path.append(current_point)

                    # Aktualizacja poprzedniego punktu
                    previous_point = current_point

        # Powrot do punktu startowego - przemieszczenie w osi Y poza regal
        current_point = copy.copy(point)
        current_point[1] = 2

        # chebyshev nodes - interpolacja
        new_points = generate_intermediate_points(
            point, current_point, self.num_chebyshev_along_beds_to_start
        )
        new_path += new_points

        new_path.append(current_point)

        # Powrot do punktu startowego - przemieszczenie w osi X do punktu startowego
        end_point = [
            self.end_location.x,
            self.end_location.y,
            self.end_location.z,
            self.end_location.roll,
            self.end_location.pitch,
            self.end_location.yaw,
        ]
        new_points = generate_intermediate_points(
            current_point, end_point, self.num_chebyshev_back_to_start
        )
        new_path += new_points

        new_path.append(end_point)
        return new_path

    def prepare_points(self, points_from_drone):
        example_of_points = []
        example_of_points.append(0)
        example_of_points.append(1)
        for x in points_from_drone:
            example_of_points.append(x * 2)
            example_of_points.append(x * 2 + 1)
        return example_of_points


def chebyshev_nodes(n, a, b):
    xk = [np.cos((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)][::-1]
    # od -1 do 1

    # przeksztalcenie afiniczne do zadanego przedzialu (a, b)
    xk_norm = np.zeros((n, 3))
    for i in range(n):
        xk_norm[i] = xk[i] * (b - a) / 2 + (a + b) / 2

    return xk_norm


def generate_intermediate_points(previous_point, point, n):
    a = np.array(previous_point[:3])
    b = np.array(point[:3])
    xk_norm = chebyshev_nodes(n, a, b)

    new_points = []
    for i in range(len(xk_norm)):
        interpoint = copy.deepcopy(previous_point)
        interpoint[:3] = xk_norm[i]
        new_points.append(interpoint)

    return new_points


def start(AREAS_FROM_DRONE):

    LOCALIZATION = positions.POINTS_OF_INTEREST
    HEURISTIC = positions.matrix_of_distances()
    Astar_fly = A_star(
        start_location=Setpoint(*LOCALIZATION[0][0]),
        end_location=Setpoint(0, 0, 1, 0, 0, 0),
        heuristic=HEURISTIC,
    )
    POINTS_TO_VISIT = Astar_fly.prepare_points(AREAS_FROM_DRONE)

    for setpoint in POINTS_TO_VISIT:
        Astar_fly.add_setpoint(
            setpoint, Setpoint(*LOCALIZATION[np.floor(setpoint / 2)][setpoint % 2])
        )

    final_path = Astar_fly.search_path(LOCALIZATION)

    return final_path
