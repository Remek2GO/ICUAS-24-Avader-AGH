"""Helper classes to work with plant beds."""

import numpy as np

from copy import deepcopy
from dataclasses import dataclass

from .types import PlantType

SAME_PLANT_THRESHOLD = 0.15 #TODO Było 0.1

DEBUG_MODE = False

@dataclass
class PlantSideCount:
    """Class to store the count and position of fruits on one side of the plant."""

    fruit_count: int
    fruit_position: (
        np.ndarray
    )  # fruit_count X 2 array of x, y coordinates normalized to [0, 1]
    fruit_type: PlantType = None


class Plant:
    def __init__(self):
        self.plant_type: PlantType = None
        self.left: PlantSideCount = None
        self.right: PlantSideCount = None

    def set_plant_type(self, plant_type: PlantType):
        self.plant_type = plant_type

    def set_left(self, fruit_count: int, fruit_position: np.ndarray):
        self.left = PlantSideCount(fruit_count, deepcopy(fruit_position))

    def set_right(self, fruit_count: int, fruit_position: np.ndarray):
        self.right = PlantSideCount(fruit_count, deepcopy(fruit_position))

    def get_real_fruit_count(self) -> int:
        if self.left is None and self.right is None:
            return 0
        # TODO DO sprawdzenia - dodane jeśli zliczenie jest równe 0
        elif self.left is None or self.left.fruit_count == 0:
            return self.right.fruit_count
        elif self.right is None or self.right.fruit_count == 0:
            return self.left.fruit_count

        reversed_right_position = deepcopy(self.right.fruit_position)
        reversed_right_position[:, 1] = 1 - reversed_right_position[:, 1]

        duplicate_count = 0

        for left_fruit in self.left.fruit_position:
            for right_fruit in reversed_right_position:
                if np.linalg.norm(left_fruit - right_fruit) < SAME_PLANT_THRESHOLD:
                    duplicate_count += 1

        return self.left.fruit_count + self.right.fruit_count - duplicate_count


class PlantBed:
    def __init__(self):
        self.plants = [Plant() for _ in range(3)]

    # TODO Sprawdzić te modyfikacje !!!
    def set_plant(
        self,
        index: int,
        side: int,
        fruit_count: int,
        fruit_position: np.ndarray,
        plant_type: PlantType,
    ):
        if plant_type != PlantType.EMPTY:
            self.plants[index].set_plant_type(plant_type)
            if side == 0:
                self.plants[index].set_left(fruit_count, fruit_position)
            else:
                self.plants[index].set_right(fruit_count, fruit_position)
        else:
            if DEBUG_MODE:
                print("Dodanie pustego.....")
            if side == 0:
                self.plants[index].set_left(0, fruit_position)
            else:
                self.plants[index].set_right(0, fruit_position)

    def get_bed_fruit_count(self, type: PlantType) -> int:
        return sum(
            plant.get_real_fruit_count()
            for plant in self.plants
            if plant.plant_type == type
        )

    def get_bed_fruit_count_right(self, type: PlantType) -> int:
        return sum(
            plant.right.fruit_count for plant in self.plants if plant.plant_type == type
        )

    def get_bed_fruit_count_left(self, type: PlantType) -> int:
        return sum(
            plant.left.fruit_count for plant in self.plants if plant.plant_type == type
        )
