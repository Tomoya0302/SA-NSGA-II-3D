# SA-NSGA-II-3D

## Execution Environment

- Ubuntu 18.06
- ROS Melodic
- Robot: Niryo Ned

https://docs.niryo.com/dev/ros/v4.1.1/en/source/installation/ubuntu_18.html to set up the environment.

## Execution Procedure

1. Execute the following commands in the terminal to prepare the robot:
   ```
   roslaunch niryo_robot_bringup desktop_rviz_simulation.launch
   ```
1. In a new terminal, run the following commands:
   ```
   python3 layout_opt.py
   ```
1. In a new terminal, run the following commands:
   ```
   python2 layout_ned2_surrogate.py
   ```
