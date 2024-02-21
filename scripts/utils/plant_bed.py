"""Helper classes to work with plant beds."""

import numpy as np

from copy import deepcopy
from dataclasses import dataclass

from .types import PlantType

SAME_PLANT_THRESHOLD = 0.15  # TODO Było 0.1

DEBUG_MODE = False


@dataclass
class PlantSideCount:
    """Class to store the count and position of fruits on one side of the plant."""

    fruit_count: int
    # fruit_count X 2 array of x, y coordinates normalized to [0, 1]
    fruit_position: np.ndarray
    fruit_type: PlantType = None


class Plant:
    """Class to store the count and position of fruits on both sides of the plant."""

    def __init__(self):
        """Initialize the plant with no fruits."""
        self.plant_type: PlantType = None
        self.left: PlantSideCount = None
        self.right: PlantSideCount = None

    def set_plant_type(self, plant_type: PlantType):
        """Set the type of the plant.

        Args:
            plant_type (PlantType): Type of the plant.
        """
        self.plant_type = plant_type

    def set_left(self, fruit_count: int, fruit_position: np.ndarray):
        """Set the count and position of fruits on the left side of the plant.

        Args:
            fruit_count (int): Fruit count.
            fruit_position (np.ndarray): Fruit position.
        """
        self.left = PlantSideCount(
            fruit_count=fruit_count,
            fruit_position=deepcopy(fruit_position),
            fruit_type=self.plant_type,
        )

    def set_right(self, fruit_count: int, fruit_position: np.ndarray):
        """Set the count and position of fruits on the right side of the plant.

        Args:
            fruit_count (int): Fruit count.
            fruit_position (np.ndarray): Fruit position.
        """
        self.right = PlantSideCount(
            fruit_count=fruit_count,
            fruit_position=deepcopy(fruit_position),
            fruit_type=self.plant_type,
        )

    def get_real_fruit_count(self) -> int:
        """Get the real fruit count of the plant.

        Returns:
            int: Real fruit count.
        """
        # if self.left is None and self.right is None:
        #     return 0
        # elif self.left is None or (
        #     self.left is not None and self.left.fruit_count == 0
        # ):
        #     return self.get
        # elif self.right is None or self.right.fruit_count == 0:
        #     return self.left.fruit_count

        # Get number of fruits on each side
        left_count = self.get_fruit_count_left()
        right_count = self.get_fruit_count_right()

        # Search for duplicates only if there are fruits on both sides
        if left_count == 0 or right_count == 0:
            return left_count + right_count

        reversed_right_position = deepcopy(self.right.fruit_position)
        reversed_right_position[:, 1] = 1 - reversed_right_position[:, 1]

        duplicate_count = 0

        for left_fruit in self.left.fruit_position:
            for right_fruit in reversed_right_position:
                if np.linalg.norm(left_fruit - right_fruit) < SAME_PLANT_THRESHOLD:
                    duplicate_count += 1

        return self.left.fruit_count + self.right.fruit_count - duplicate_count

    def get_fruit_count_left(self) -> int:
        """Get the count of fruits on the left side of the plant.

        NOTE: If the left side is `None`, the function returns 0.

        Returns:
            int: Fruit count on the left side.
        """
        if self.left is None:
            return 0
        return self.left.fruit_count

    def get_fruit_count_right(self) -> int:
        """Get the count of fruits on the right side of the plant.

        NOTE: If the right side is `None`, the function returns 0.

        Returns:
            int: Fruit count on the right side.
        """
        if self.right is None:
            return 0
        return self.right.fruit_count


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

    def get_bed_fruit_count(self, fruit_type: PlantType) -> int:
        return sum(
            plant.get_real_fruit_count()
            for plant in self.plants
            if plant.plant_type == fruit_type
        )

    def get_bed_fruit_count_right(self, fruit_type: PlantType) -> int:
        return sum(
            plant.get_fruit_count_right()
            for plant in self.plants
            if plant.plant_type == fruit_type
        )

    def get_bed_fruit_count_left(self, fruit_type: PlantType) -> int:
        return sum(
            plant.get_fruit_count_left()
            for plant in self.plants
            if plant.plant_type == fruit_type
        )
