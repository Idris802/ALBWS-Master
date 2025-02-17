#!/usr/bin/env python3

import sys
import rospy
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from moveit_commander import PlanningSceneInterface

rospy.init_node('add_collision_objects')

scene = PlanningSceneInterface(synchronous=True)

# Wait for the planning scene to initialize
rospy.sleep(2)

# Create a collision object for the table
table = CollisionObject()
table.header.frame_id = "world"  # adjust if your planning frame is different
table.id = "table"

# Define the primitive (e.g., a box) for the table dimensions
box = SolidPrimitive()
box.type = SolidPrimitive.BOX
box.dimensions = [0.8, 1.5, 0.03]  # length, width, height (in meters)

table.primitives = [box]

# Define the pose of the table (position and orientation)
table_pose = PoseStamped()
table_pose.header.frame_id = "world"
table_pose.pose.position.x = -0.485472 # adjust as needed
table_pose.pose.position.y = -0.000012
table_pose.pose.position.z = 1  # table top height (table thickness + support height)


table.primitive_poses = [table_pose.pose]

# Add the table into the planning scene
scene.add_object(table)
#scene.remove_world_object("table")

rospy.loginfo("Added table to the planning scene")
rospy.spin()
