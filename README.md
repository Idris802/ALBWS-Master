# Repo for Master Thesis Project 

This repository contains the source code for my Master’s thesis project at the Norwegian University of Life Sciences (NMBU). The thesis develops an autonomous laser-based weeding framework composed of four modules: machine vision, weed detection, motion planning, and laser actuation. A novel contribution is the fusion of YOLO with RANSAC to enhance weed-stem localization. To evaluate the framework, a Gazebo simulation was built featuring a UR3 arm with an attached laser pointer, an Intel RealSense camera, and virtual weed and crop models.


## Simulation & Execution

1. **Launch the UR3 and environment in Gazebo**
   Starts the UR3 robot, and spawns table, camera, weeds, and all other simulation components.

   ```bash
   roslaunch ur_gazebo ur3_bringup.launch
   ```

2. **Start MoveIt! in simulation mode**
   Brings up MoveIt! planning and execution nodes configured for the UR3.

   ```bash
   roslaunch ur3_moveit_config moveit_planning_execution.launch sim:=true
   ```

3. **Run the weed detection node**
   Begins real-time YOLO-based weed detection from the camera feed.

   ```bash
   rosrun moveit_ur3 weed_detector.py
   ```

4. **Execute the motion planner & laser weeding**
   Commands the UR3 to navigate to each detected weed and activate the laser pointer.

   ```bash
   rosrun moveit_ur3 move_ur3.py
   ```
## Dependencies

This workspace includes:

- **realsense2_description**  
  – URDF & Gazebo models for Intel RealSense cameras  
  – Link: <https://github.com/issaiass/realsense2_description>

- **universal_robot**  
  – ROS drivers, Gazebo models, and MoveIt! configs for UR3/UR5/UR10  
  – Link: <https://github.com/ros-industrial/universal_robot>

- **virtual_maize_field**  
  – Custom Gazebo world containing maize, weeds, and terrain models  
  – Link: <https://github.com/FieldRobotEvent/virtual_maize_field>

- **realsense_gazebo_plugin**  
  – Gazebo plugin for RealSense depth & RGB streams  
  – Link: <https://github.com/pal-robotics/realsense_gazebo_plugin>



