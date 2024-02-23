#!/usr/bin/env python
"""Node to evaluate the model on the test track."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import csv
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, NewType, Tuple

import rospy

# from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Int32, String
from tf.transformations import quaternion_matrix

from icuas24_competition.msg import AnalyzerResult
from scripts.utils.types import PlantBedsIds, PlantType

Boundaries = NewType("Boundaries", List[Tuple[float, float]])

# Tuples with the minimum and maximum values for the arena boundaries
# in the x, y, z axes
ARENA_BOUNDARIES: Boundaries = [(0.0, 20.0), (0.0, 27.0), (0.0, 9.0)]

# Tuples with the minimum and maximum values for the plant bed boundaries
# in the x, y, z axes
BED_BOUNDARIES: List[Boundaries] = [
    [(2.95, 5.05), (2.95, 24.05), (0.0, 9.0)],
    [(8.95, 11.05), (2.95, 24.05), (0.0, 9.0)],
    [(14.95, 17.05), (2.95, 24.05), (0.0, 9.0)],
]

# BRIDGE = CvBridge()

# End (destination) position for the UAV
END_POSITION = np.array([1.0, 1.0])

# IMAGES_FOLDER_PATH = "/root/sim_ws/src/icuas24_competition/images"

# Boundary vectors for the UAV in the shape [8, 3] - vertices of the cube
UAV_BOUNDARY_VECTORS = np.array(
    [
        # [x, y, z]
        [0.525, 0.525, 0.1],
        [0.525, -0.525, 0.1],
        [-0.525, -0.525, 0.1],
        [-0.525, 0.525, 0.1],
        [0.525, 0.525, -0.4],
        [0.525, -0.525, -0.4],
        [-0.525, -0.525, -0.4],
        [-0.525, 0.525, -0.4],
    ]
)


@dataclass
class Plant:
    """Class to store the plant type with the number of fruits on each side."""

    plant_type: PlantType
    all_fruits: int
    left_fruits: int
    right_fruits: int

    def __str__(self) -> str:
        """Return the string representation of the plant.

        Returns:
            str: String representation of the plant.
        """
        return f"Type: {self.plant_type}\n\tAll: {self.all_fruits}\
            \n\tLeft: {self.left_fruits}\n\tRight: {self.right_fruits}"


@dataclass
class PlantBed:
    """Class to store the type of plant in the bed."""

    left: Plant
    centre: Plant
    right: Plant

    def __str__(self) -> str:
        """Return the string representation of the plant bed.

        Returns:
            str: String representation of the plant bed.
        """
        return f"Left: {self.left}\nCentre: {self.centre}\nRight: {self.right}"


# TODO: New mechanism for collision detection and area boundaries
class Evaluator:
    """Class to evaluate the model on the test track."""

    def __init__(self, path_to_beds_csv: str, path_to_output_json: str):
        """Initialize the evaluator.

        Args:
            path_to_beds_csv (str): Path to the CSV file containing the plant beds.
        """
        self.challenge_started_received: bool = False
        self.plants_beds_received: bool = False
        self.fruit_count_received: bool = False
        self.end_position_reached: bool = False
        self.start_time: float = None
        self.end_time: float = None
        self.plant_beds_ids: PlantBedsIds = None
        self.beds_gt: Dict[int, Plant] = {}
        self.beds_results: Dict[int, Plant] = {}
        self.beds_counted: Dict[int, List[bool]] = {}
        self.fruit_count_gt: int = None
        self.fruit_count_result: int = None
        self.red_prev_position: np.ndarray = None
        self.red_distance: float = 0.0
        self.final_points: float = 0.0
        self.uav_in_collision: bool = False
        self.collision_bed_id: int = -1
        self.collision_cnt: int = 0
        self.path_to_output_json: str = path_to_output_json

        # Load the plant beds from the CSV file to the dictionary
        # NOTE: We add a dummy bed at the beginning to match the bed_id
        self.plant_beds: List[PlantBed] = [PlantBed(None, None, None)]
        with open(path_to_beds_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # NOTE: We remove the last character from the plant type to remove the
                # trailing 's'
                left = Plant(
                    PlantType(row[1][:-1].upper()),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                )
                centre = Plant(
                    PlantType(row[5][:-1].upper()),
                    int(row[6]),
                    int(row[7]),
                    int(row[8]),
                )
                right = Plant(
                    PlantType(row[9][:-1].upper()),
                    int(row[10]),
                    int(row[11]),
                    int(row[12]),
                )
                new_plant_bed = PlantBed(left, centre, right)
                self.plant_beds.append(new_plant_bed)
                rospy.loginfo(
                    f"[Evaluator] Loaded plant bed {row[0]}:\n{new_plant_bed}"
                )
        rospy.loginfo(f"[Evaluator] Loaded {len(self.plant_beds) - 1} plant beds.")

        # ROS subscribers
        rospy.Subscriber(
            "/evaluator/analyzer_result", AnalyzerResult, self._analyzer_result_clb
        )
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_clb)
        rospy.Subscriber("/fruit_count", Int32, self._fruit_count_clb)
        rospy.Subscriber("/red/challenge_started", Bool, self._challenge_started_clb)
        rospy.Subscriber("/red/plants_beds", String, self._plants_beds_clb)

    def _analyzer_result_clb(self, msg: AnalyzerResult):
        # Check if UAV is searching the proper bed
        if msg.bed_id not in self.plant_beds_ids.bed_ids:
            rospy.logerr(f"[Evaluator] UAV is searching wrong bed {msg.bed_id}.")
            return

        # Get the number of fruits on the given side of the bed
        gt = None
        if msg.bed_side == 0:
            gt = self.beds_gt[msg.bed_id].left_fruits
            count = msg.fruit_left
        elif msg.bed_side == 1:
            gt = self.beds_gt[msg.bed_id].right_fruits
            count = msg.fruit_right
        else:
            rospy.logerr(f"[Evaluator] Invalid bed side {msg.bed_side}.")
            return

        # Check the number of fruits on the given side of the bed
        self.beds_counted[msg.bed_id][msg.bed_side] = True
        if count == gt:
            rospy.loginfo(f"[Evaluator] ({msg.bed_id}, {msg.bed_side}): Correct {gt}.")
        else:
            rospy.loginfo(
                f"\033[31m[Evaluator] ({msg.bed_id}, {msg.bed_side}): Incorrect "
                f"{count} [GT: {gt}].\033[0m"
            )

        # Check all fruits count when both sides were counted
        if msg.bed_id not in self.beds_results:
            self.beds_results[msg.bed_id] = Plant(
                self.plant_beds_ids.plant_type,
                all_fruits=msg.fruit_sum,
                left_fruits=msg.fruit_left,
                right_fruits=msg.fruit_right,
            )
        else:
            self.beds_results[msg.bed_id].all_fruits = msg.fruit_sum
            self.beds_results[msg.bed_id].left_fruits = msg.fruit_left
            self.beds_results[msg.bed_id].right_fruits = msg.fruit_right

    def _calculate_collision_points(self) -> float:
        return -25 * self.collision_cnt

    def _calculate_fruit_points(self, count_val: int) -> float:
        denominator = (
            self.fruit_count_gt
            if self.fruit_count_gt > 0
            else 2 * len(self.plant_beds_ids.bed_ids)
        )
        return 50 * (1 - 4 * abs(count_val - self.fruit_count_gt) / denominator)

    def _calculate_path_points(self, distance_val: float) -> float:
        path_base = 150
        return 25 * np.exp(2 * (1 - distance_val / path_base))

    def _calculate_time_points(self, time_val: float) -> float:
        time_base = 100
        return 25 * np.exp(1 - time_val / time_base)

    def _challenge_started_clb(self, msg: Bool):
        if not self.challenge_started_received and msg.data:
            self.challenge_started_received = True
            rospy.loginfo("[Evaluator] Received challenge started.")
        if self.plants_beds_received:
            self.start_time = rospy.get_time()
            rospy.loginfo("[Evaluator] Time start.")

    def _fruit_count_clb(self, msg: Int32):
        # Save fruit count result
        self.fruit_count_result = msg.data
        self.fruit_count_received = True

    def _inside_test(self, points: np.ndarray, bounds: Boundaries) -> int:
        """Check if the points are inside the given boundaries.

        Args:
            points (np.ndarray): Points to check in the shape [N, 3]
            bounds (Boundaries): Boundaries for the x, y, z axes

        Returns:
            int: Number of points inside the boundaries
        """
        # Get boundaries for the x, y, z axes
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        zmin, zmax = bounds[2]

        # Check if the points are inside the boundaries
        points_inside = []
        for i in range(len(points)):
            if (
                xmin < points[i, 0] < xmax
                and ymin < points[i, 1] < ymax
                and zmin < points[i, 2] < zmax
            ):
                points_inside.append(i)

        return len(points_inside)

    def _model_states_clb(self, msg: ModelStates):
        # No measurements after end
        if self.end_position_reached:
            return

        # Get the pose of the UAV
        try:
            red_idx = msg.name.index("red")
        except ValueError:
            return

        # No measurements before start
        if not (self.challenge_started_received and self.plants_beds_received):
            return

        red_pose: Pose = msg.pose[red_idx]
        red_position = np.array(
            [red_pose.position.x, red_pose.position.y, red_pose.position.z]
        )
        red_rotation = quaternion_matrix(
            [
                red_pose.orientation.x,
                red_pose.orientation.y,
                red_pose.orientation.z,
                red_pose.orientation.w,
            ]
        )[:3, :3]
        boundary_points = np.matrix(
            np.matmul(red_rotation, UAV_BOUNDARY_VECTORS.T)
            + np.matrix(red_position).getT()
        ).getT()

        # Check arena boundaries
        if not self.uav_in_collision and self._inside_test(
            boundary_points, ARENA_BOUNDARIES
        ) < len(UAV_BOUNDARY_VECTORS):
            self.uav_in_collision = True
            self.collision_cnt += 1
            rospy.loginfo("\033[31m[Evaluator] UAV left the arena.\033[0m")
        elif self.uav_in_collision and self._inside_test(
            boundary_points, ARENA_BOUNDARIES
        ) == len(UAV_BOUNDARY_VECTORS):
            self.uav_in_collision = False
            rospy.loginfo("[Evaluator] UAV re-entered the arena.")

        # Check plant beds boundaries
        for i, bounds in enumerate(BED_BOUNDARIES):
            if (
                self.collision_bed_id == -1
                and self._inside_test(boundary_points, bounds) > 0
            ):
                self.collision_bed_id = i
                self.collision_cnt += 1
                rospy.loginfo(
                    f"\033[31m[Evaluator] UAV collided with plant bed {i}.\033[0m"
                )
                break
            elif (
                self.collision_bed_id == i
                and self._inside_test(boundary_points, bounds) == 0
            ):
                self.uav_in_collision = False
                self.collision_bed_id = -1
                rospy.loginfo(f"[Evaluator] UAV left plant bed {i}.")

        # Distance + position check
        if self.red_prev_position is not None:
            # Accumulate the distance travelled by the UAV
            self.red_distance += np.linalg.norm(red_position - self.red_prev_position)

            # Check if the UAV reached the end position
            if (
                np.linalg.norm(red_position[:2] - END_POSITION) < 0.1
                and np.linalg.norm(self.red_prev_position[:2] - END_POSITION) < 0.1
                and self.fruit_count_received
            ):
                rospy.loginfo("[Evaluator] End position reached.")
                self.end_position_reached = True
        else:
            self.red_distance = 0.0
        self.red_prev_position = red_position

    def _plants_beds_clb(self, msg: String):
        self.plants_beds_received = True
        rospy.loginfo("[Evaluator] Received plant beds.")
        if self.challenge_started_received:
            self.start_time = rospy.get_time()
            rospy.loginfo("[Evaluator] Time start.")

        beds_data = msg.data.split(" ")
        self.plant_beds_ids = PlantBedsIds(
            PlantType(beds_data[0].upper()),
            [int(x) for x in beds_data[1:]],
        )
        rospy.loginfo(f"[Evaluator] Mission specification:\n{self.plant_beds_ids}")

        # Compute ground truth for each bed and side
        self.fruit_count_gt = 0
        for bed_id in self.plant_beds_ids.bed_ids:
            all_fruits = 0
            left_fruits = 0
            right_fruits = 0
            if (
                self.plant_beds_ids.plant_type
                == self.plant_beds[bed_id].left.plant_type
            ):
                all_fruits += self.plant_beds[bed_id].left.all_fruits
                left_fruits += self.plant_beds[bed_id].left.left_fruits
                right_fruits += self.plant_beds[bed_id].left.right_fruits

            if (
                self.plant_beds_ids.plant_type
                == self.plant_beds[bed_id].centre.plant_type
            ):
                all_fruits += self.plant_beds[bed_id].centre.all_fruits
                left_fruits += self.plant_beds[bed_id].centre.left_fruits
                right_fruits += self.plant_beds[bed_id].centre.right_fruits

            if (
                self.plant_beds_ids.plant_type
                == self.plant_beds[bed_id].right.plant_type
            ):
                all_fruits += self.plant_beds[bed_id].right.all_fruits
                left_fruits += self.plant_beds[bed_id].right.left_fruits
                right_fruits += self.plant_beds[bed_id].right.right_fruits
            new_gt = Plant(
                self.plant_beds_ids.plant_type, all_fruits, left_fruits, right_fruits
            )
            self.beds_gt[bed_id] = new_gt
            self.fruit_count_gt += all_fruits
            self.beds_counted[bed_id] = [False, False]
            rospy.loginfo(f"[Evaluator] Ground truth for bed {bed_id}:\n{new_gt}")
        rospy.loginfo(
            f"[Evaluator] Ground truth for fruit count: {self.fruit_count_gt}."
        )

    def run(self):
        """Run the evaluator."""
        wait_rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.end_position_reached and self.fruit_count_received:
                self.end_time = rospy.get_time()

                # Log beds not counted
                not_counted_beds = []
                for bed_id in self.beds_gt:
                    if not all(self.beds_counted[bed_id]):
                        not_counted_beds.append(bed_id)
                        rospy.loginfo(
                            f"\033[31m[Evaluator] Bed {bed_id} not fully "
                            f"counted.\033[m"
                        )

                # Log fruit count summary
                for bed_id in self.beds_results:
                    gt = self.beds_gt[bed_id].all_fruits
                    count = self.beds_results[bed_id].all_fruits
                    if count == gt:
                        rospy.loginfo(
                            f"\033[32m[Evaluator] ({bed_id}): Correct final sum "
                            f"{gt}.\033[0m"
                        )
                    else:
                        rospy.loginfo(
                            f"\033[31m[Evaluator] ({bed_id}): Incorrect final sum "
                            f"{count} [GT: {gt}].\033[0m"
                        )

                # Log fruit points
                fruit_points = self._calculate_fruit_points(self.fruit_count_result)
                self.final_points += fruit_points

                if self.fruit_count_result == self.fruit_count_gt:
                    rospy.loginfo(
                        f"\033[32m[Evaluator] Correct fruit count: "
                        f"{self.fruit_count_result}: {fruit_points:.2f} points\033[0m"
                    )
                else:
                    rospy.loginfo(
                        f"\033[31m[Evaluator] Incorrect fruit count: "
                        f"{self.fruit_count_result}: {fruit_points:.2f} points "
                        f"[GT: {self.fruit_count_gt}]\033[0m"
                    )

                # Log collision and fly-off points
                collision_points = self._calculate_collision_points()
                self.final_points += collision_points
                rospy.loginfo(
                    f"[Evaluator] Collision and fly-off counts "
                    f"{self.collision_cnt}: {collision_points:.2f} points"
                )

                # Log distance points
                distance_points = self._calculate_path_points(self.red_distance)
                self.final_points += distance_points
                rospy.loginfo(
                    f"[Evaluator] Distance {self.red_distance:.2f}: "
                    f"{distance_points:.2f} points"
                )

                # Log time points
                time_delta = self.end_time - self.start_time
                time_points = self._calculate_time_points(time_delta)
                self.final_points += time_points
                rospy.loginfo(
                    f"[Evaluator] Time delta {time_delta:.4f}: {time_points:.2f} points"
                )

                # Log final points
                rospy.loginfo(
                    f"\033[1;33m[Evaluator] Total points: {self.final_points:.2f}."
                    f"\033[0m"
                )

                # Save the results to the JSON file
                if self.path_to_output_json:
                    with open(self.path_to_output_json, "w") as f:
                        json.dump(
                            {
                                "beds_to_visit": self.plant_beds_ids.bed_ids,
                                "beds_not_counted": not_counted_beds,
                                "fruit_count_gt": self.fruit_count_gt,
                                "fruit_count_result": self.fruit_count_result,
                                "fruit_points": fruit_points,
                                "collision_cnt": self.collision_cnt,
                                "collision_points": collision_points,
                                "distance": self.red_distance,
                                "distance_points": distance_points,
                                "time_delta": time_delta,
                                "time_points": time_points,
                                "final_points": self.final_points,
                            },
                            f,
                        )

                # Shutdown the node
                return
            wait_rate.sleep()


if __name__ == "__main__":
    myargv = rospy.myargv(argv=sys.argv)
    if len(myargv) < 2:
        rospy.logerr("[Evaluator] Please provide the path to the plant beds CSV file.")
        sys.exit(1)
    beds_csv_path = myargv[1]

    output_json_path = None
    if len(myargv) > 2:
        output_json_path = myargv[2]
    else:
        rospy.logwarn(
            "[Evaluator] No output JSON file path provided. "
            "The results will not be saved."
        )

    rospy.init_node("evaluator")
    evaluator = Evaluator(beds_csv_path, output_json_path)
    evaluator.run()

    rospy.loginfo("[Evaluator] Shutting down.")
