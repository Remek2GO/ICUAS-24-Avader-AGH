#!/usr/bin/env python

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PoseStamped, Transform
from tf.transformations import quaternion_from_euler
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List
from scripts.utils import A_star
from scripts.utils.types import PlantType, TrackerStatus, PathStatus, Setpoint, PlantBed


class PathSetter:
    def __init__(self) -> None:
        self.challenge_started = False
        self.plant_beds: PlantBed = None
        self.tracker_status = TrackerStatus.OFF
        self.idx_setpoint = 0
        self.path_status = PathStatus.REACHED
        self.rate = rospy.Rate(20)
        self.current_fruit_count = 0

        self.pub_pose = rospy.Publisher(
            "/red/tracker/input_pose", PoseStamped, queue_size=10
        )
        self.pub_fruit_count = rospy.Publisher("/fruit_count", Int32, queue_size=10)
        self.pub_trajectory = rospy.Publisher(
            "/red/tracker/input_trajectory", MultiDOFJointTrajectory, queue_size=10
        )

        self.sub_challenge_started = rospy.Subscriber(
            "/red/challenge_started", Bool, self.set_challenge_started
        )
        self.sub_plants_beds = rospy.Subscriber(
            "/red/plants_beds", String, self.set_plants_beds
        )
        self.sub_tracker_status = rospy.Subscriber(
            "/red/tracker/status", String, self.check_tracker_status
        )
        self.sub_current_fruit_count = rospy.Subscriber(
            "/current_fruit_count", Int32, self.set_current_fruit_count
        )

        self.setpoints: List[Setpoint] = []

    def set_challenge_started(self, data: Bool):
        if data.data == True:
            self.challenge_started = True

    def set_plants_beds(self, data: String):
        plant_beds = data.data.split(" ")

        self.plant_beds = PlantBed(
            PlantType(plant_beds[0].upper()), [int(bed_id) for bed_id in plant_beds[1:]]
        )

    def check_tracker_status(self, data: String):
        self.tracker_status = TrackerStatus(data.data)
        
    def set_current_fruit_count(self, data: Int32):
        self.current_fruit_count = data.data

    def set_setpoint(self, setpoint: Setpoint):
        msg = PoseStamped()
        msg.pose.position.x = setpoint.x
        msg.pose.position.y = setpoint.y
        msg.pose.position.z = setpoint.z

        euler_RPY = [setpoint.roll, setpoint.pitch, setpoint.yaw]
        quaternion = quaternion_from_euler(euler_RPY[0], euler_RPY[1], euler_RPY[2])
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.pub_pose.publish(msg)

    def add_setpoint(self, setpoint: Setpoint):
        self.setpoints.append(setpoint)

    def wait_for_challenge_start(self):
        rospy.loginfo("Waiting for challenge to start")
        while (
            not self.challenge_started
            or self.plant_beds == None
            or self.tracker_status == TrackerStatus.OFF
        ):
            self.rate.sleep()
        rospy.loginfo("Challenge started")

    def run(self):
        while not self.path_status == PathStatus.COMPLETED:
            if self.path_status == PathStatus.REACHED and self.idx_setpoint == len(
                self.setpoints
            ):
                self.path_status = PathStatus.COMPLETED
                self.handle_challenge_completed()
                return

            if (
                self.tracker_status == TrackerStatus.ACTIVE
                and self.path_status == PathStatus.WAITING
            ):
                self.path_status = PathStatus.PROGRESS

            if (
                self.path_status == PathStatus.PROGRESS
                and self.tracker_status == TrackerStatus.ACCEPT
            ):
                self.path_status = PathStatus.REACHED
                rospy.loginfo("Setpoint reached")
            elif self.path_status == PathStatus.REACHED:
                rospy.loginfo(
                    f"Setting new setpoint {self.setpoints[self.idx_setpoint]}"
                )
                self.set_setpoint(self.setpoints[self.idx_setpoint])
                self.idx_setpoint += 1
                self.path_status = PathStatus.WAITING

            self.rate.sleep()

    def send_trajectory(self):
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
            x, y, z, w = quaternion_from_euler(setpoint.roll, setpoint.pitch, setpoint.yaw)
            transform.rotation.x = x
            transform.rotation.y = y
            transform.rotation.z = z
            transform.rotation.w = w
            
            point.transforms.append(transform)
            trajectory.points.append(point)
        
        self.pub_trajectory.publish(trajectory)

    def handle_challenge_completed(self):
        rospy.loginfo("Challenge completed")
        self.pub_fruit_count.publish(self.current_fruit_count)

MANUAL = False
# MANUAL = True

USE_POINTS = True
# USE_POINTS = False

if __name__ == "__main__":
    rospy.init_node("path_setter", anonymous=True)
    rospy.loginfo("path_setter node started")

    path_setter = PathSetter()
    path_setter.wait_for_challenge_start()

    if MANUAL:
        x = input("X:")
        y = input("Y:")
        z = input("Z:")
        roll = input("Roll:")
        pitch = input("Pitch:")
        yaw = input("Yaw:")
        path_setter.add_setpoint(Setpoint(float(x), float(y), float(z), float(roll), float(pitch), float(yaw)))
    else:
        SETPOINTS = A_star.start(path_setter.plant_beds.bed_ids)      
    
        for setpoint in SETPOINTS:
            path_setter.add_setpoint(Setpoint(*setpoint))

    if USE_POINTS:
        path_setter.run()
    else:
        path_setter.send_trajectory()

    rospy.spin()
