cmake_minimum_required(VERSION 3.0.2)
project(laser_pointer_plugin)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs gazebo_ros)
find_package(gazebo REQUIRED)   # Find Gazebo for plugin development

catkin_package(
  LIBRARIES laser_pointer_plugin
  CATKIN_DEPENDS roscpp std_msgs gazebo_ros
)

# Include directories for ROS and Gazebo
include_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})


# Build the plugin library
add_library(laser_pointer_plugin src/laser_pointer_plugin.cpp)
target_link_libraries(laser_pointer_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

