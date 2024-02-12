import numpy as np
from .types import PlantType
from dataclasses import dataclass

@dataclass
class PlantSide:
    fruit_count: int
    fruit_position: np.ndarray # fruit_count X 2 array of x, y coordinates normalized to [0, 1]

class Plant:
    def __init__(self):
        self.plant_type: PlantType = None
        self.left: PlantSide = None
        self.right: PlantSide = None

    def set_plant_type(self, plant_type: PlantType):
        self.plant_type = plant_type

    def set_left(self, fruit_count: int, fruit_position: np.ndarray):
        self.left = PlantSide(fruit_count, fruit_position)

    def set_right(self, fruit_count: int, fruit_position: np.ndarray):
        self.right = PlantSide(fruit_count, fruit_position)

    def get_real_fruit_count(self) -> int:
        if self.left is None and self.right is None:
            return 0
        elif self.left is None:
            return self.right.fruit_count
        elif self.right is None:
            return self.left.fruit_count
            
                
        reversed_right_position = self.right.fruit_position
        reversed_right_position[:, 0] = 1 - reversed_right_position[:, 0]

        duplicate_count = 0

        for left_fruit in self.left.fruit_position:
            for right_fruit in reversed_right_position:
                if np.linalg.norm(left_fruit - right_fruit) < 0.1:
                    duplicate_count += 1
        
        return self.left.fruit_count + self.right.fruit_count - duplicate_count


class PlantBed:
    def __init__(self):
        self.plants = [Plant() for _ in range(3)]

    def set_plant(self, index: int, side: int, fruit_count: int, fruit_position: np.ndarray, plant_type: PlantType):
        self.plants[index].set_plant_type(plant_type)
        if side == 0:
            self.plants[index].set_left(fruit_count, fruit_position)
        else:
            self.plants[index].set_right(fruit_count, fruit_position)

    def get_bed_fruit_count(self) -> int:
        return sum(plant.get_real_fruit_count() for plant in self.plants)


        
