cmake_minimum_required(VERSION 3.0.2)
project(laser_control_plugin)

find_package(catkin REQUIRED COMPONENTS roscpp gazebo_ros std_msgs)

find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES laser_toggle_plugin
  CATKIN_DEPENDS roscpp gazebo_ros std_msgs
  DEPENDS GAZEBO
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(laser_toggle_plugin src/laser_toggle_plugin.cpp)
target_link_libraries(laser_toggle_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
