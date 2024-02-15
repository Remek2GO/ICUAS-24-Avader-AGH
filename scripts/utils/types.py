"""Types definition for the project."""

from enum import Enum
from dataclasses import dataclass
from typing import List


class PlantType(Enum):
    """Enum class for plant types.

    Attributes:
        PEPPER: Pepper plant type.
        TOMATO: Tomato plant type.
        EGGPLANT: Eggplant plant type.
        EMPTY: Empty plant type.
    """

    PEPPER = "PEPPER"
    TOMATO = "TOMATO"
    EGGPLANT = "EGGPLANT"
    EMPTY = "EMPTY"

    def __str__(self) -> str:
        """Return the string representation of the plant type.

        Returns:
            str: The name of the plant type.
        """
        return self.value


class TrackerStatus(Enum):
    """Enum class for tracker status.

    Attributes:
        OFF: Tracker is off.
        ACTIVE: Tracker is active.
        ACCEPT: Tracker is accepting new commands.
    """

    OFF = "OFF"
    ACTIVE = "ACTIVE"
    ACCEPT = "ACCEPT"


class PathStatus(Enum):
    """Enum class for path status.

    Attributes:
        REACHED: The tracker has reached the setpoint.
        PROGRESS: The tracker is following the path.
        COMPLETED: The tracker has completed the path.
        WAITING: The tracker is waiting for the next command.
    """

    REACHED = 0
    PROGRESS = 1
    COMPLETED = 2
    WAITING = 3


@dataclass
class PlantBedsIds:
    """Class for storing the location of the beds of plants of the specific type to \
        visit during flight.

    Attributes:
        plant_type: `PlantType`: The type of the plant.
        bed_ids: `List[int]`: The list of bed ids.
    """

    plant_type: PlantType
    bed_ids: List[int]


@dataclass
class Setpoint:
    """Class for setpoint definition.

    Attributes:
        x: `float`: x coordinate.
        y: `float`: y coordinate.
        z: `float`: z coordinate.
        roll: `float`: roll angle.
        pitch: `float`: pitch angle.
        yaw: `float`: yaw angle.
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
