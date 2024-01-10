#!/usr/bin/env python

from socket import socket, AF_INET, SOCK_STREAM
from contextlib import closing # if python2
from moveit_msgs.msg import RobotState as RS
from sensor_msgs.msg import JointState
from niryo_robot_python_ros_wrapper import *
import rospy
import moveit_commander
import pickle
import numpy as np

HOST = 'localhost'
PORT = 51001
MAX_MESSAGE = 2048

class Robot:
    def __init__(self, real_ot=True):
        self.real_ot = real_ot
        self.simulation = True
        self.offset_position = [0.0, 0.50, -1.25, 0.0, 0.0, 0.0]
        self.joint_state = JointState()
        self.moveit_robot_state = RS()

        # from technical specifications (https://docs.niryo.com/product/ned/v4.0.0/en/source/hardware/technical_specifications.html)
        self.max_speed = np.array([2.6, 2.0, 2.5, 3.14, 3.14, 3.14])
        self.max_speed *= 0.4 # scaling

        self.model = pickle.load(open('model/model_rf.sav', 'rb'))

    def socket_up(self):
        print(' connectiong ... ')
        # with socket(AF_INET, SOCK_STREAM) as sock: # python3
        with closing(socket(AF_INET, SOCK_STREAM)) as sock: # python2
            sock.connect((HOST, PORT))
            self.main(sock)

    # client to server
    def send_message(self, sock, msg):
        while True:
            sock.send(msg.encode('utf-8'))
            break

    def set_up(self):
        # Initializing ROS node
        rospy.init_node('niryo_ned_example_python_ros_wrapper')
        # Connecting to the ROS wrapper & calibrating if needed
        self.robot = NiryoRosWrapper()
        self.robot.calibrate_auto()
        self.robot.move_to_sleep_pose()
        # get info
        self.move_group = moveit_commander.MoveGroupCommander('arm')
        self.robot_group = moveit_commander.RobotCommander()

    def main(self, sock):
        self.sock = sock
        self.set_up()
        self.offset_position = [0.0, 0.50, -1.25, 0.0, 0.0, 0.0]
        while True:
            finish_flag = False
            data = sock.recv(MAX_MESSAGE)
            data = pickle.loads(data)
            if data[0] == 7777: # continue (simulation)
                self.simulation = True
                # self.robot.move_to_sleep_pose()
                self.sock.send(pickle.dumps([0]))
            elif data[0] == 8888: # continue
                self.simulation = False
                # self.robot.move_to_sleep_pose()
                self.sock.send(pickle.dumps([0]))
            elif data[0] == 9999: # finish
                print('Finish')
                # self.robot.set_learning_mode(True)
                break
            while True:
                # move_joint
                target = sock.recv(MAX_MESSAGE)
                target = pickle.loads(target)
                
                if target[-1] == 1:
                    finish_flag = True
                pick_time = self.execute(target[:6], self.simulation)
                place_time = self.execute(target[6:12], self.simulation)
                self.sock.send(pickle.dumps([pick_time + place_time]))
                if finish_flag:
                    break
    
    def execute(self, target, simulation):
        # plan
        init_joint = self.offset_position
        try:
            target_joint = self.robot.inverse_kinematics(target[0], target[1], target[2], target[3], target[4], target[5])
        except NiryoRosWrapperException:
            return float('inf')
        input = np.array(target_joint) - np.array(init_joint)
        if self.real_ot:
            elapsed_time = self.model.predict([input])[0]
        else:
            elapsed_time = self.calc_from_joint(input)
        self.offset_position = target_joint
        return elapsed_time

    def calc_from_joint(self, target_joint):
        ot_list = np.zeros(len(target_joint))
        for i in range(len(target_joint)):
            ot_list[i] = np.abs(target_joint[i]) / self.max_speed[i]
        operation_time = np.max(ot_list)
        return operation_time

if __name__ == '__main__':
    robot = Robot(real_ot=True)
    robot.socket_up()