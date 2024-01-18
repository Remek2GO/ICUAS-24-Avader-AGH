#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import numpy as np

challenge_started = False
plant_beds = None
pose_publisher = None
fruit_count_publisher = None
STAGE = 0
PROGRESS = False

SETPOINTS = [
    [1, 5, 2, 0, 0, np.pi/2],
    [10, 6, 5, 0, 0, -np.pi/2],
    [10, 10, 3, 0, 0, 0],
    [0, 0, 1, 0, 0, np.pi]
]

COMPLETED = True

def enable(data: Bool):
    global challenge_started
    if data.data == True:
        challenge_started = True

def read_plant_beds(data: String):
    global plant_beds
    # format: PLANT_TYPE BED_ID1 BED_ID2 ....
    data_array = data.data.split(" ")

    plant_beds = data_array
    
def listener():
    rospy.init_node('avader_uav', anonymous=True)
    rospy.loginfo("avader_uav node started")

    rospy.Subscriber("/red/challenge_started", Bool, enable)
    rospy.Subscriber("/red/plants_beds", String, read_plant_beds)


def check_completion(data: String):
    global COMPLETED, STAGE, PROGRESS
    if data.data == "ACTIVE":
        PROGRESS = True
        return
    if data.data == "ACCEPT" and COMPLETED == False and PROGRESS == True:
        COMPLETED = True
        rospy.loginfo("Setpoint reached")

        STAGE += 1
        if STAGE == len(SETPOINTS):
            rospy.loginfo("Challenge completed")
            fruit_count_publisher.publish(42)
            return
        
        set_setpoint(SETPOINTS[STAGE])

        


def set_setpoint(data):
    global COMPLETED, PROGRESS
    rospy.loginfo("Setting new setpoint")
    msg = PoseStamped()
    msg.pose.position.x = data[0]
    msg.pose.position.y = data[1]
    msg.pose.position.z = data[2]

    euler_RPY = [data[3], data[4], data[5]]
    quaternion = quaternion_from_euler(euler_RPY[0], euler_RPY[1], euler_RPY[2])
    msg.pose.orientation.x = quaternion[0]
    msg.pose.orientation.y = quaternion[1]
    msg.pose.orientation.z = quaternion[2]
    msg.pose.orientation.w = quaternion[3]

    pose_publisher.publish(msg)
    COMPLETED = False
    PROGRESS = False


def talker():
    global pose_publisher, fruit_count_publisher
    pose_publisher = rospy.Publisher('/red/tracker/input_pose', PoseStamped, queue_size=10)
    fruit_count_publisher = rospy.Publisher('/fruit_count', Int32, queue_size=10)

    rate = rospy.Rate(100)

    rospy.loginfo("Waiting for challenge to start")

    while not challenge_started or plant_beds == None:
        rate.sleep()

    rospy.loginfo("Challenge started")
    step = 0

    while not rospy.is_shutdown():
        if step == 500:
            set_setpoint(SETPOINTS[STAGE])
            break
        rate.sleep()
        step += 1

    rospy.Subscriber("/red/tracker/status", String, check_completion)



    

if __name__ == '__main__':
    try:
        listener()
        talker()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass