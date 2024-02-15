#!/usr/bin/env python
"""ROS node to set the path for the tracker to follow."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PoseStamped, Transform
from tf.transformations import quaternion_from_euler
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from typing import List

from scripts.utils import A_star
from scripts.utils.types import (
    PlantType,
    TrackerStatus,
    PathStatus,
    Setpoint,
    PlantBedsIds,
)

from icuas24_competition.msg import UavSetpoint


class PathSetter:
    """Class to set the path for the tracker to follow."""

    def __init__(self) -> None:
        """Initialize the PathSetter class."""
        self.challenge_started = False
        self.plant_beds: PlantBedsIds = None
        self.tracker_status = TrackerStatus.OFF
        self.idx_setpoint = 0
        self.path_status = PathStatus.REACHED
        self.move_on: bool = True
        self.rate = rospy.Rate(20)
        self.current_fruit_count = 0
        self.setpoints: List[Setpoint] = []
        self.take_photo_msg: UavSetpoint = UavSetpoint()

        # ROS publishers
        self.pub_pose = rospy.Publisher(
            "/red/tracker/input_pose", PoseStamped, queue_size=10
        )
        self.pub_fruit_count = rospy.Publisher("/fruit_count", Int32, queue_size=10)
        self.pub_trajectory = rospy.Publisher(
            "/red/tracker/input_trajectory", MultiDOFJointTrajectory, queue_size=10
        )
        self.pub_take_photo = rospy.Publisher("/take_photo", UavSetpoint, queue_size=10)

        # ROS subscribers
        self.sub_challenge_started = rospy.Subscriber(
            "/red/challenge_started", Bool, self._set_challenge_started_clb
        )
        self.sub_plants_beds = rospy.Subscriber(
            "/red/plants_beds", String, self._set_plants_beds_clb
        )
        self.sub_tracker_status = rospy.Subscriber(
            "/red/tracker/status", String, self._set_tracker_status_clb
        )
        self.sub_current_fruit_count = rospy.Subscriber(
            "/current_fruit_count", Int32, self._set_current_fruit_count_clb
        )
        self.sub_move_on = rospy.Subscriber("/move_on", Bool, self._move_on_clb)

    def _move_on_clb(self, data: Bool):
        self.move_on = data.data

    def _set_challenge_started_clb(self, data: Bool):
        if data.data is True:
            self.challenge_started = True

    def _set_current_fruit_count_clb(self, data: Int32):
        self.current_fruit_count = data.data

    def _set_plants_beds_clb(self, data: String):
        # Input format: "plant_type bed_id1 bed_id2 ..."
        plant_beds = data.data.split(" ")

        self.plant_beds = PlantBedsIds(
            PlantType(plant_beds[0].upper()), [int(bed_id) for bed_id in plant_beds[1:]]
        )

    def _set_tracker_status_clb(self, data: String):
        self.tracker_status = TrackerStatus(data.data)

    def set_setpoint(self, setpoint: Setpoint):
        """Set the setpoint for the tracker to follow.

        Args:
            setpoint (Setpoint): The setpoint to publish on the tracker topic.
        """
        msg = PoseStamped()
        msg.pose.position.x = setpoint.x
        msg.pose.position.y = setpoint.y
        msg.pose.position.z = setpoint.z

        quaternion = quaternion_from_euler(setpoint.roll, setpoint.pitch, setpoint.yaw)
        # NOTE: The order of the quaternion is (x, y, z, w) - ROS tf convention
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.pub_pose.publish(msg)

    def add_setpoint(self, setpoint: Setpoint):
        """Add a new setpoint to the list of setpoints.

        Args:
            setpoint (Setpoint): The setpoint to add to the list.
        """
        self.setpoints.append(setpoint)

    def wait_for_challenge_start(self):
        """Wait for the challenge to start.

        This method waits for the 3 conditions to be met:
        - to receive the challenge_started message,
        - to receive the plant_beds message,
        - to receive the tracker status "OFF" message.
        """
        rospy.loginfo("Waiting for challenge to start")
        while (
            not self.challenge_started
            or self.plant_beds is None
            or self.tracker_status == TrackerStatus.OFF
        ):
            self.rate.sleep()
        rospy.loginfo("Challenge started")

    def run(self):
        """Run the path setter node.

        This method runs the path setter node. It waits for the tracker to reach the \
            setpoint, then it sets the next setpoint.
        """
        while not self.path_status == PathStatus.COMPLETED:
            # UAV reached the last setpoint
            if self.path_status == PathStatus.REACHED and self.idx_setpoint == len(
                self.setpoints
            ):
                self.path_status = PathStatus.COMPLETED
                self.handle_challenge_completed()
                return

            # UAV flying to the next setpoint
            if (
                self.tracker_status == TrackerStatus.ACTIVE
                and self.path_status == PathStatus.WAITING
            ):
                self.path_status = PathStatus.PROGRESS

            # UAV reached the setpoint
            if (
                self.path_status == PathStatus.PROGRESS
                and self.tracker_status == TrackerStatus.ACCEPT
            ):
                self.path_status = PathStatus.REACHED
                rospy.sleep(1)
                self.pub_take_photo.publish(self.take_photo_msg)
                self.move_on = False
                rospy.loginfo("[Path Setter] Take photo")
                # rospy.loginfo("Setpoint reached")
            elif self.path_status == PathStatus.REACHED and self.move_on:
                # Set the next setpoint
                rospy.loginfo(
                    f"Setting new setpoint {self.setpoints[self.idx_setpoint]}"
                )
                # Send new setpoint to the tracker
                self.set_setpoint(self.setpoints[self.idx_setpoint])

                # Update take photo message
                self.take_photo_msg.x = self.setpoints[self.idx_setpoint].x
                self.take_photo_msg.y = self.setpoints[self.idx_setpoint].y
                self.take_photo_msg.z = self.setpoints[self.idx_setpoint].z
                self.take_photo_msg.roll = self.setpoints[self.idx_setpoint].roll
                self.take_photo_msg.pitch = self.setpoints[self.idx_setpoint].pitch
                self.take_photo_msg.yaw = self.setpoints[self.idx_setpoint].yaw

                self.idx_setpoint += 1
                self.path_status = PathStatus.WAITING

            self.rate.sleep()

    def send_trajectory(self):
        """Send the entire trajectory to the tracker to follow.

        This is an alternative to the run method. It sends the entire trajectory to \
            the tracker to follow instead of sending each setpoint individually.
        """
        trajectory = MultiDOFJointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = "world"

        for setpoint in self.setpoints:
            point = MultiDOFJointTrajectoryPoint()
            point.transforms = []
            transform = Transform()
            transform.translation.x = setpoint.x
            transform.translation.y = setpoint.y
            transform.translation.z = setpoint.z
            x, y, z, w = quaternion_from_euler(
                setpoint.roll, setpoint.pitch, setpoint.yaw
            )
            transform.rotation.x = x
            transform.rotation.y = y
            transform.rotation.z = z
            transform.rotation.w = w

            point.transforms.append(transform)
            trajectory.points.append(point)

        self.pub_trajectory.publish(trajectory)

    def handle_challenge_completed(self):
        """Handle the challenge completion.

        This method publishes the final fruit count to the `/fruit_count` topic.
        """
        rospy.loginfo("Challenge completed")
        self.pub_fruit_count.publish(self.current_fruit_count)


if __name__ == "__main__":
    myargv = rospy.myargv(argv=sys.argv)
    arg_manual = "--manual" in myargv
    arg_use_points = "--use-points" in myargv

    rospy.init_node("path_setter", anonymous=True)
    rospy.loginfo("[Path Setter] Node started")

    path_setter = PathSetter()
    path_setter.wait_for_challenge_start()

    if arg_manual:
        # Manual setpoints
        x = input("X:")
        y = input("Y:")
        z = input("Z:")
        roll = input("Roll:")
        pitch = input("Pitch:")
        yaw = input("Yaw:")
        path_setter.add_setpoint(
            Setpoint(
                float(x), float(y), float(z), float(roll), float(pitch), float(yaw)
            )
        )
    else:
        # A* setpoints
        setpoints = A_star.start(path_setter.plant_beds.bed_ids)
        for setpoint in setpoints:
            path_setter.add_setpoint(Setpoint(*setpoint))

    if arg_use_points:
        # Fly poiny by point
        path_setter.run()
    else:
        # Send the entire trajectory
        path_setter.send_trajectory()

    rospy.spin()
