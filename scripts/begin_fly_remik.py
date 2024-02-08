#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import numpy as np
from dataclasses import dataclass

@dataclass
class Setpoint:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

class StartProgram:
    def __init__(self):
        rospy.loginfo("Rozpoczeto program!")
        rospy.init_node('Poczatek')

        self.drone_is_ready = False

        self.defchallengestarted = rospy.Subscriber("/red/challenge_started",Bool, callback = self.drone_ready)
        self.publish_point = rospy.Publisher("/red/tracker/input_pose", PoseStamped, queue_size=10)

    def drone_ready(self,msg: Bool):
        if msg.data == True:
            self.drone_is_ready = True
            rospy.loginfo("Dron jest gotowy.")

    def addpoint(self, point):
        self.point = point

    def run(self):
        rate = rospy.Rate(100)
        while not self.drone_is_ready:
            rospy.loginfo('Czekam')
            rate.sleep()

        rospy.loginfo("FLY")
        msg = PoseStamped()
        msg.pose.position.x = self.point.x
        msg.pose.position.y = self.point.y
        msg.pose.position.z = self.point.z
        msg.pose.orientation.x = self.point.roll
        msg.pose.orientation.y = self.point.pitch
        msg.pose.orientation.z = self.point.yaw
        msg.pose.orientation.w = 0
        

        self.publish_point.publish(msg)



if __name__ == '__main__':
    first_point = [13, 13.5, 4.6, 0, 0, 0]

    begin_fly = StartProgram()
    begin_fly.addpoint(Setpoint(*first_point))
    begin_fly.run()

    rospy.spin()

