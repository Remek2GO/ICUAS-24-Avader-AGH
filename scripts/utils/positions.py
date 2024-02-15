"""Helper functions to work with beds positions."""

import numpy as np
import sys
from typing import NewType, Tuple

np.set_printoptions(threshold=sys.maxsize)

PointOfInterest = NewType(
    "PointOfInterest", Tuple[float, float, float, float, float, float]
)
POINTS_OF_INTEREST = {
    # [x, y, z, roll, pitch, yaw]
    # first row is point with smaller x, second row is point on the other side
    0: [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    1: [[1, 6, 1.8, 0, 0, 0], [7, 6, 1.8, 0, 0, np.pi]],
    2: [[1, 6, 4.6, 0, 0, 0], [7, 6, 4.6, 0, 0, np.pi]],
    3: [[1, 6, 7.4, 0, 0, 0], [7, 6, 7.4, 0, 0, np.pi]],
    4: [[1, 13.5, 1.8, 0, 0, 0], [7, 13.5, 1.8, 0, 0, np.pi]],
    5: [[1, 13.5, 4.6, 0, 0, 0], [7, 13.5, 4.6, 0, 0, np.pi]],
    6: [[1, 13.5, 7.4, 0, 0, 0], [7, 13.5, 7.4, 0, 0, np.pi]],
    7: [[1, 21, 1.8, 0, 0, 0], [7, 21, 1.8, 0, 0, np.pi]],
    8: [[1, 21, 4.6, 0, 0, 0], [7, 21, 4.6, 0, 0, np.pi]],
    9: [[1, 21, 7.4, 0, 0, 0], [7, 21, 7.4, 0, 0, np.pi]],
    10: [[7, 6, 1.8, 0, 0, 0], [13, 6, 1.8, 0, 0, np.pi]],
    11: [[7, 6, 4.6, 0, 0, 0], [13, 6, 4.6, 0, 0, np.pi]],
    12: [[7, 6, 7.4, 0, 0, 0], [13, 6, 7.4, 0, 0, np.pi]],
    13: [[7, 13.5, 1.8, 0, 0, 0], [13, 13.5, 1.8, 0, 0, np.pi]],
    14: [[7, 13.5, 4.6, 0, 0, 0], [13, 13.5, 4.6, 0, 0, np.pi]],
    15: [[7, 13.5, 7.4, 0, 0, 0], [13, 13.5, 7.4, 0, 0, np.pi]],
    16: [[7, 21, 1.8, 0, 0, 0], [13, 21, 1.8, 0, 0, np.pi]],
    17: [[7, 21, 4.6, 0, 0, 0], [13, 21, 4.6, 0, 0, np.pi]],
    18: [[7, 21, 7.4, 0, 0, 0], [13, 21, 7.4, 0, 0, np.pi]],
    19: [[13, 6, 1.8, 0, 0, 0], [19, 6, 1.8, 0, 0, np.pi]],
    20: [[13, 6, 4.6, 0, 0, 0], [19, 6, 4.6, 0, 0, np.pi]],
    21: [[13, 6, 7.4, 0, 0, 0], [19, 6, 7.4, 0, 0, np.pi]],
    22: [[13, 13.5, 1.8, 0, 0, 0], [19, 13.5, 1.8, 0, 0, np.pi]],
    23: [[13, 13.5, 4.6, 0, 0, 0], [19, 13.5, 4.6, 0, 0, np.pi]],
    24: [[13, 13.5, 7.4, 0, 0, 0], [19, 13.5, 7.4, 0, 0, np.pi]],
    25: [[13, 21, 1.8, 0, 0, 0], [19, 21, 1.8, 0, 0, np.pi]],
    26: [[13, 21, 4.6, 0, 0, 0], [19, 21, 4.6, 0, 0, np.pi]],
    27: [[13, 21, 7.4, 0, 0, 0], [19, 21, 7.4, 0, 0, np.pi]],
}


def get_distance(p1_indeces: Tuple[int, int], p2_indeces: Tuple[int, int]) -> float:
    """Calculate the distance between two points of interest.

    Args:
        p1_indeces (Tuple[int, int]): Indeces of the first point in POINTS_OF_INTEREST.
        p2_indeces (Tuple[int, int]): Indeces of the second point in POINTS_OF_INTEREST.

    Returns:
        float: The distance between the two points.
    """
    p1 = POINTS_OF_INTEREST[p1_indeces[0]][p1_indeces[1]]
    p2 = POINTS_OF_INTEREST[p2_indeces[0]][p2_indeces[1]]
    if p1[0] == p2[0]:
        # The same side of the beds, so the distance is just the euclidean distance
        return np.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )
    elif p1_indeces[0] == 0 or p2_indeces == 0:
        # One of the points is the start position, so the distance is the sum of the
        # euclidean distance and the distance to the start position
        distance = np.sqrt(abs(p2[0] - p1[0]) ** 2 + max(p1[2], p2[2]) ** 2)
        distance += max(abs(p1[1] - 2), abs(p2[1] - 2))
        return distance
    else:
        # Both points are on different sides of the beds, so the distance is the sum of
        # the distance in x, the shortest distances to the end of the bed and the
        # euclidean distance.
        distance = abs(p2[0] - p1[0])
        distance += min(
            abs(p1[1] - 2) + abs(p2[1] - 2), abs(p1[1] - 25) + abs(p2[1] - 25)
        )
        distance = np.sqrt(distance**2 + (p1[2] - p2[2]) ** 2)
        return distance


def matrix_of_distances() -> np.ndarray:
    """Create a matrix of distances between all points of interest.

    Returns:
        np.ndarray: The matrix of distances.
    """
    n = 2 * len(POINTS_OF_INTEREST)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = get_distance(
                (np.floor(i / 2), i % 2), (np.floor(j / 2), j % 2)
            )
    return distances
