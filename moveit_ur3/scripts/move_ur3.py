#!/usr/bin/env python3

from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import math
import tf.transformations as tfm
import geometry_msgs.msg
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import PointStamped, PoseStamped
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String, Bool
from moveit_commander.conversions import pose_to_list
from sensor_msgs.msg import JointState

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("move_group_ur3_interface", anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()

group_name = "manipulator"
move_group = moveit_commander.MoveGroupCommander(group_name)

display_trajectory_publisher = rospy.Publisher(
    "/move_group/display_planned_path",
    moveit_msgs.msg.DisplayTrajectory,
    queue_size=20,
)

planning_frame = move_group.get_planning_frame()
print("============ Planning frame: %s" % planning_frame)

# We can also print the name of the end-effector link for this group:
eef_link = move_group.get_end_effector_link()
print("============ End effector link: %s" % eef_link)

# We can get a list of all the groups in the robot:
group_names = robot.get_group_names()
print("============ Available Planning Groups:", robot.get_group_names())

# Sometimes for debugging it is useful to print the entire state of the
# robot:
print("============ Printing robot state")
print(robot.get_current_state())
print("")

laser_pub = rospy.Publisher("/set_laser_pointer", Bool, queue_size=10)

weed_positions = []
subscriber = None

joint_state_times = []
joint_state_positions = {}  

def move_robot_to_target(x, y, z):

    # Define the target pose
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z

    # Compute a quaternion for a 180 degree rotation about the Y-axis.
    # This will flip the end-effector so that its z-axis (the tool axis) points downward.
    q = tfm.quaternion_from_euler(0, math.pi, 0)
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]

    # Set pose target
    move_group.set_pose_target(target_pose)
    move_group.set_planning_time(50.0)

    # Plan and execute motion
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    if plan:
        rospy.loginfo(f"Robot moved to target (X={x}, Y={y}, Z={z}) successfully.")
    else:
        rospy.logwarn("Motion planning failed!")


def weed_callback(msg):
    global weed_positions, subscriber
    rospy.loginfo("Received weed coordinates: X = {:.3f}, Y = {:.3f}, Z = {:.3f}".format(msg.point.x, msg.point.y, msg.point.z))
    home_position = [-0.08987249676046005, -1.513799495494775, -1.563427913618015, -1.6978510331504433, 1.5968145769242623, -0.08069322207556606]
    move_group.go(home_position, wait="true")
    move_group.stop()
    if len(weed_positions) < 6:
        weed_positions.append([msg.point.x, msg.point.y, msg.point.z])

    if len(weed_positions) >= 6 and subscriber is not None:
        subscriber.unregister()
        rospy.loginfo("Collected 6 weed positions. Unsubscribing from /weed_coordinates.")
        rospy.loginfo("Weed positions: {}".format(weed_positions))

        for pos in weed_positions:
            move_robot_to_target(pos[0], pos[1], 1.02)
            #reorient_end_effector([pos[0], pos[1], pos[2]])
            laser_pub.publish(Bool(data=True))
            rospy.loginfo("Laser turned ON at weed position: {}".format(pos))
            rospy.sleep(2.0)
            laser_pub.publish(Bool(data=False))
            rospy.loginfo("Laser turned OFF")
            rospy.sleep(2.0) 

def joint_state_callback(msg):
    global joint_state_times, joint_state_positions
    t = msg.header.stamp.to_sec()
    joint_state_times.append(t)
    for i, name in enumerate(msg.name):
        if name not in joint_state_positions:
            joint_state_positions[name] = []
        joint_state_positions[name].append(msg.position[i])

def plot_joint_states():
    """
    Plot joint positions over time using the collected joint state data.
    """
    # Find the minimum number of data points collected
    min_length = len(joint_state_times)
    for positions in joint_state_positions.values():
        if len(positions) < min_length:
            min_length = len(positions)
    
    if min_length == 0:
        rospy.logwarn("No joint state data collected!")
        return

    # Normalize time data so that it starts at 0
    times = np.array(joint_state_times[:min_length]) - joint_state_times[0]
    
    plt.figure(figsize=(10, 6))
    for joint, positions in joint_state_positions.items():
        plt.plot(times, positions[:min_length], label=joint)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Position (rad)")
    plt.title("Joint States Over Time (Simulated Annealing TSP)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    moveit_commander.roscpp_initialize([])
    home_position = [0, 0, 0, 0, 0, 0]
    move_group.go(home_position, wait="true")
    move_group.stop()

    subscriber = rospy.Subscriber("/weed_coordinates", PointStamped, weed_callback)
    rospy.spin()