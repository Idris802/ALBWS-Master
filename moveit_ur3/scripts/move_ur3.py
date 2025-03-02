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
from geometry_msgs.msg import PointStamped, PoseStamped

from math import pi, tau, dist, fabs, cos

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


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


print("Moving to Home position!")


current_pose_stamped = move_group.get_current_pose()


home_position = [-0.08987249676046005, -1.513799495494775, -1.563427913618015, -1.6978510331504433, 1.5968145769242623, -0.08069322207556606]
move_group.go(home_position, wait="true")
move_group.stop()


import numpy as np
import tf.transformations as tfm
import math


def quaternion_from_vectors(v0, v1):
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    d = np.dot(v0, v1)
    if d >= 1.0:
        return np.array([0, 0, 0, 1])
    if d <= -0.999999:
        axis = np.cross(np.array([0, 0, 1]), v0)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(np.array([0, 1, 0]), v0)
        axis = axis / np.linalg.norm(axis)
        return tfm.quaternion_about_axis(math.pi, axis)
    s = math.sqrt((1+d)*2)
    invs = 1 / s
    cross = np.cross(v0, v1)
    q = np.array([cross[0]*invs, cross[1]*invs, cross[2]*invs, 0.5*s])
    return q

def reorient_end_effector(weed_coords):
    current_pose = move_group.get_current_pose().pose
    current_pos = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
    
    # Compute the direction vector from the current end-effector position to the weed
    target_pos = np.array(weed_coords)
    direction = target_pos - current_pos
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        rospy.logwarn("Target coincides with current position; using default orientation.")
        new_quat = np.array([0, 0, 0, 1])
    else:
        direction = direction / norm
        # Default forward axis of the tool is assumed to be [1, 0, 0]
        default_forward = np.array([0, 0, 1])
        new_quat = quaternion_from_vectors(default_forward, direction)
    
    # Construct new end effector pose: keep current position, update orientation
    new_pose = PoseStamped()
    new_pose.header.frame_id = move_group.get_planning_frame()
    new_pose.pose.position = current_pose.position
    new_pose.pose.orientation.x = new_quat[0]
    new_pose.pose.orientation.y = new_quat[1]
    new_pose.pose.orientation.z = new_quat[2]
    new_pose.pose.orientation.w = new_quat[3]

    move_group.set_pose_target(new_pose.pose)
    move_group.set_planning_time(10)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    
    if plan:
        rospy.loginfo("End effector reoriented successfully.")
    else:
        rospy.logwarn("End effector reorientation failed.")


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
    rospy.loginfo("Received weed coordinates: X = {:.3f}, Y = {:.3f}, Z = {:.3f}".format(msg.point.x, msg.point.y, msg.point.z))
    home_position = [-0.08987249676046005, -1.513799495494775, -1.563427913618015, -1.6978510331504433, 1.5968145769242623, -0.08069322207556606]
    move_group.go(home_position, wait="true")
    move_group.stop()
    move_robot_to_target(msg.point.x, msg.point.y, 1.04)


if __name__ == "__main__":
    moveit_commander.roscpp_initialize([])
    # Example weed coordinates in the base frame
    weed_coordinates = [-0.41, -0.11, 0.9]
    #reorient_end_effector(weed_coordinates)
    
    #rospy.Subscriber("/weed_coordinates", PointStamped, weed_callback)
    #rospy.spin()

    #move_robot_to_target(-0.51+0.07, 0.33, 1.04)