#!/usr/bin/env python

import numpy as np
from dataclasses import dataclass
from scripts import positions
import copy

@dataclass
class Setpoint:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

class A_star:
    def __init__(self, start_location, heuristic) -> None:
        self.setpoints = {}
        self.start_location = start_location
        self.end_location = start_location
        self.heuristic_distances = heuristic
        self.open_list = []
        self.close_list = []
        self.path = []

    def add_setpoint(self, number, setpoint: Setpoint):
        self.setpoints.update({number : setpoint})


    def MST(self):

        N = self.open_list
        selected_node = np.zeros(len(N))

        no_edge = 0
        selected_node[0] = True
        sum_weights = 0

        while (no_edge < len(N) - 1):
            
            minimum = np.inf
            a = 0
            b = 0
            for m in N:
                if selected_node[N.index(m)]:
                    for n in N:
                        if ((not selected_node[N.index(n)]) and self.heuristic_distances[m][n]):  
                            if minimum > self.heuristic_distances[m][n]:
                                minimum = self.heuristic_distances[m][n]
                                a = m
                                b = n
            sum_weights += self.heuristic_distances[a][b]
            selected_node[N.index(b)] = True
            no_edge += 1
        return sum_weights




    def search_path(self, LOCALIZATION):
        end_found = False
        current_node = 0
        g_cost = 0

        for setpoint in self.setpoints.keys():
            self.open_list.append(setpoint)
        self.open_list.remove(current_node)
        self.open_list.remove(current_node+1)

        while not end_found:
            f_cost = {}
            h_cost = self.MST()
            
            h_cost += min(self.heuristic_distances[0]) + min(self.heuristic_distances[current_node])

            for setpoint in self.open_list:
                f_cost.update({setpoint : g_cost + self.heuristic_distances[current_node][setpoint] + h_cost})
            
            next_node = min(f_cost, key=f_cost.get)
            self.path.append(next_node)
            self.open_list.remove(next_node)
            self.close_list.append(next_node)
            g_cost += self.heuristic_distances[current_node][next_node]
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
            points_to_visit.append(LOCALIZATION[np.floor(elem/2)][elem%2])
        
        print("Punkty bez pośrednich :", points_to_visit)
        new_path.append([points_to_visit[0][0],2,points_to_visit[0][2],0,0,0])
        for point in points_to_visit:
            if not previous_point:
                point_copy = copy.copy(point)
                previous_point = point_copy
                new_path.append(point_copy)
            else:
                if point[0] == previous_point[0]:
                    point_copy = copy.deepcopy(point)
                    new_path.append(point_copy)
                    previous_point = point_copy
                else:
                    previous_point_copy = copy.deepcopy(previous_point)
                    point_copy = copy.deepcopy(point)
                    previous_point_copy[5] = np.pi/2
                    if abs(previous_point[1] - 25) > abs(previous_point[1] - 2):
                        previous_point_copy[1] = 2
                    else:
                        previous_point_copy[1] = 25
                    new_path.append(previous_point_copy)
                    previous_point_copy_copy = copy.copy(previous_point_copy)
                    previous_point_copy_copy[2] = copy.copy(point_copy[2])
                    previous_point_copy_copy[0] = copy.copy(point_copy[0])
                    new_path.append(previous_point_copy_copy)
                    new_path.append(point_copy)
                    previous_point = point_copy

        point_copy = copy.copy(point)
        point_copy[1] = 2
        new_path.append(point_copy)
        new_path.append([0,0,1,0,0,0])
        return new_path


    def prepare_points(self, points_from_drone):
        example_of_points = []
        example_of_points.append(0)
        example_of_points.append(1)
        for x in points_from_drone:
            example_of_points.append(x*2)
            example_of_points.append(x*2+1)
        return example_of_points

def start(AREAS_FROM_DRONE):
    LOCALIZATION = positions.POINTS_OF_INTEREST

    HEURISTIC = positions.matrix_of_distances()

    #AREAS_FROM_DRONE = [5,15,18,20]

    Astar_fly = A_star(start_location=Setpoint(*LOCALIZATION[0][0]), heuristic=HEURISTIC)

    POINTS_TO_VISIT = Astar_fly.prepare_points(AREAS_FROM_DRONE)


    for setpoint in POINTS_TO_VISIT:    
        Astar_fly.add_setpoint(setpoint, Setpoint(*LOCALIZATION[np.floor(setpoint/2)][setpoint%2]))

    final_path = Astar_fly.search_path(LOCALIZATION)
    
    return final_path

#print("Z Pośrednimi :", final_path)


