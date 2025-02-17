#!/usr/bin/env python3

from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

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

# This home position point in the middle of one of the weeds
#home_position = [-3.386220136689589, -0.791759142008023, 2.1463065543869195, -3.175737824943092, -1.6229534627056639, 5.955665051921544]


current_pose_stamped = move_group.get_current_pose()


home_position = [-0.08987249676046005, -1.513799495494775, -1.563427913618015, -1.6978510331504433, 1.5968145769242623, -0.08069322207556606]
move_group.go(home_position, wait="true")
move_group.stop()

print(current_pose_stamped)




def move_robot_to_target(x, y, z):

    # Define the target pose
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z

    target_pose.orientation.x = -0.6740364714501995
    target_pose.orientation.y = -0.7280108198417545
    target_pose.orientation.z = 0.03600485520975907
    target_pose.orientation.w = 0.11991134954467493

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

if __name__ == "__main__":
    rospy.sleep(2)  # Wait for MoveIt! to start
    move_robot_to_target( -0.198, -0.371, 0.99)  # Example target



"""
    position: 
    x: -0.36893114552717743
    y: 0.14882972121227517
    z: 0.997584799281745
  orientation: 
    x: -0.7148948362138683
    y: -0.6729972065244401
    z: 0.1780274555097695
    w: 0.06562284853435958


"""