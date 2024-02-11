from enum import Enum
from dataclasses import dataclass
from typing import List

class PlantType(Enum):
    PEPPER = "PEPPER"
    TOMATO = "TOMATO"
    EGGPLANT = "EGGPLANT"


class TrackerStatus(Enum):
    OFF = "OFF"
    ACTIVE = "ACTIVE"
    ACCEPT = "ACCEPT"


class PathStatus(Enum):
    REACHED = 0
    PROGRESS = 1
    COMPLETED = 2
    WAITING = 3


@dataclass
class Setpoint:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class PlantBed:
    plant_type: PlantType
    bed_ids: List[int]