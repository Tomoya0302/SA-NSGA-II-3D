import rectangle_packing_solver as rps
import numpy as np
import yaml
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread
import pickle
import matplotlib.patches as patches
from matplotlib import pylab as plt
import dill
import time

DUMMY = 0
EXPANSION = 0.001 # mm -> m
WIDTH_LIMIT = 600
HEIGHT_LIMIT = 600

MAX_ITERATION = 2

SHOW_FIG = False # show and save layout every time or not

# Socket
HOST = 'localhost'
PORT = 51001
MAX_MESSAGE = 2048
NUM_THREAD = 1

class Layout:
    def __init__(self):
        self.color = ['royalblue', 'gold', 'brown', 'orangered', 'lightsalmon', 'greenyellow', 'limegreen', 'green', 'darkcyan', 'slategrey', 'lightpink', 'indigo',
                      'black','black','black','black','black','black','black','black','black','black','black','black','black','black','black','black','black','black','black','black']

    def area(self, gp, gn, gd, rot, sol, iter, show):
        seqtriple = rps.SequenceTriple(triple=(gp, gn, gd)) 
        floorplan = seqtriple.decode(problem=self.problem, rotations=rot)
        solution = rps.Solution(seqtriple,floorplan)
        floorplan_dict = self.upload(solution, sol, iter)
        if show:
            self.visualize(floorplan_dict, solution, sol, iter)
        return solution.floorplan.area, floorplan_dict

    def upload(self, solution, sol, iter):
        # upload the floorplan
        floorplan_dict = {}
        for i in range(len(solution.floorplan.positions)):
            pos_dict = solution.floorplan.positions[i]
            id = pos_dict['id']
            del pos_dict['id']
            floorplan_dict[id] = pos_dict
        if sol:
            file_name = './floorplan_log/floorplan_solution_' + str(iter) + '.yaml'
            with open(file_name, 'w') as fp:
                yaml.dump(floorplan_dict, fp)

        return floorplan_dict

    def visualize(self, floorplan, solution, sol, iter):
        plt.rcParams["font.size"] = 14
        positions = solution.floorplan.positions
        bounding_box = solution.floorplan.bounding_box

        # Figure settings
        bb_width = bounding_box[0]
        bb_height = bounding_box[1]
        fig = plt.figure(figsize=(10, 10 * bb_height / bb_width + 0.5))
        ax = plt.axes()
        ax.set_aspect("equal")
        plt.xlim([0, bb_width])
        plt.ylim([0, bb_height])
        plt.xlabel("X")
        plt.ylabel("Y")

        # Plot every rectangle
        for i, rectangle in enumerate(positions):
            r = patches.Rectangle(
                xy=(rectangle["x"], rectangle["y"]),
                width=rectangle["width"],
                height=rectangle["height"],
                edgecolor="#000000",
                facecolor=self.color[i],
                alpha=1.0,
                fill=True,
            )
            ax.add_patch(r)

            # Add text label
            centering_offset = 0.011
            center_x = rectangle["x"] + rectangle["width"] / 2 - bb_width * centering_offset
            center_y = rectangle["y"] + rectangle["height"] / 2 - bb_height * centering_offset
            ax.text(x=center_x, y=center_y, s=str(i), fontsize=16, color='black')

        plt.show()
        if not sol:
            file_name = './floorplan_log/floorplan_' + str(iter) + '.png'
        else:
            file_name = './floorplan_log/solution_' + str(iter) + '.png'
        fig.savefig(file_name)

    def visualize_read(self, sol, iter):
        file_name = './floorplan_log/floorplan_' + str(iter) + '.yaml'
        with open(file_name, 'r') as f:
            positions = yaml.load(f,Loader=yaml.Loader)
        plt.rcParams["font.size"] = 14
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes()
        ax.set_aspect("equal")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Plot every rectangle
        for i, rectangle in enumerate(positions):
            r = patches.Rectangle(
                xy=(rectangle["x"], rectangle["y"]),
                width=rectangle["width"],
                height=rectangle["height"],
                depth=rectangle["depth"],
                edgecolor="#000000",
                facecolor=self.color[i],
                alpha=1.0,
                fill=True,
            )
            ax.add_patch(r)

            # Add text label
            centering_offset = 0.011
            center_x = rectangle["x"] + rectangle["width"] / 2 # - bb_width * centering_offset
            center_y = rectangle["y"] + rectangle["height"] / 2 # - bb_height * centering_offset
            #center_y = rectangle["z"] + rectangle["depth"] / 2 # - bb_depth * centering_offset
            ax.text(x=center_x, y=center_y, s=str(i), fontsize=18, color='black')

        plt.show()
        if not sol:
            file_name = './floorplan_log/floorplan_' + str(iter) + '.png'
        else:
            file_name = './floorplan_log/solution_' + str(iter) + '.png'
        fig.savefig(file_name)

class Robot:
    def __init__(self, replay=None):
        self.dummy = DUMMY
        self.layout = Layout()
        self.host = HOST
        self.port = PORT
        self.num_thread = NUM_THREAD
        self.expansion = EXPANSION
        self.replay = replay
        self.clients = []
        self.z_rot = [0.0, 0.0, 1.5, 0.0]
        self.finish_flag = False
        self.count = 0
        self.max_iteration = MAX_ITERATION
        self.elapsed_time_list = []
        self.output_file = open('./floorplan_log/result.txt', 'w')
        self.output_file.write('Iter.\tArea\tOperatingTime\tFeasibility\tTime\n')
        self.output_file_ = open('./floorplan_log/result_.txt', 'w')
        self.output_file_.write('Iter.\tArea\tOperatingTime\tTime\n')

        with socket(AF_INET, SOCK_STREAM) as sock:
            sock.bind(('', self.port))
            sock.listen(NUM_THREAD)
            print('Waiting ...')
            try:
                con, addr = sock.accept()
            except KeyboardInterrupt:
                print('ERROR')
            print('[CONNECT] {}'.format(addr))
            self.clients.append((con, addr))

            if self.replay == None:
                self.simulation = 7777
            else:
                self.simulation = 8888

    def move(self):
        # load
        with open('./floorplan_log/floorplan_solution_' + str(self.replay) + '.yaml', 'r') as input:
            self.floorplan = yaml.safe_load(input)
        # Robot position
        self.robot_x = -120 * self.expansion ##fixed
        self.robot_y = 120 * self.expansion ##fixed
        self.robot_z = 0 * self.expansion ##fixed height

        self.move_joint([self.simulation]) # simulation
        self.sub_th = Thread(target=self.main, args=(), daemon=True)
        self.sub_th.start()

        while True:
            if self.finish_flag:
                self.finish_flag = False
                self.count += 1
                break

    def trajectory_generate(self, area, floorplan):
        self.floorplan = floorplan[0]
        self.area = area
        # Robot position
        self.robot_x = -120 * self.expansion ##fixed
        self.robot_y = 120 * self.expansion ##fixed
        self.robot_z = 0 * self.expansion ##fixed
        self.feasibility = True

        if self.feasibility:
            self.move_joint([self.simulation]) # simulation
            self.sub_th = Thread(target=self.main, args=(), daemon=True)
            self.sub_th.start()

            while True:
                if self.finish_flag:
                    self.finish_flag = False
                    self.count += 1
                    break

        return self.ros_time, self.feasibility

    def close_connection(self):
        self.move_joint([9999]) # finish
        self.output_file.close()
        self.output_file_.close()

    def move_joint(self, data):
        data = pickle.dumps(data, protocol=2) # if python2 -> protocol=2
        self.clients[0][0].sendto(data, self.clients[0][1])

    def main(self):
        self.feasibility = True
        time_list = np.zeros(len(self.floorplan)-self.dummy, dtype=float)
        data = self.clients[0][0].recv(MAX_MESSAGE)
        # Convert Python 2 "ObjectType" to Python 3 object
        dill._dill._reverse_typemap["ObjectType"] = object
        data = pickle.loads(data, encoding='bytes')

        for i in range(len(self.floorplan)-self.dummy):
            # move to parts
            target = [
                70 * self.expansion, #x
                -170 * self.expansion, #y 
                0 * self.expansion, #z
                self.z_rot[1],
                self.z_rot[2],
                self.z_rot[3],
                (self.floorplan[i]['x'] + self.floorplan[i]['width']  / 2) * self.expansion - self.robot_x, 
                (self.floorplan[i]['y'] + self.floorplan[i]['height'] / 2) * self.expansion - self.robot_y,
                (self.floorplan[i]['z'] + self.floorplan[i]['depth'] / 2) * self.expansion - self.robot_z,
                self.z_rot[1],
                self.z_rot[2],
                self.z_rot[3],
                int(i/(len(self.floorplan)-1-self.dummy))
            ]
            self.move_joint(target)

            # ROS time
            data = self.clients[0][0].recv(MAX_MESSAGE)
            # Convert Python 2 "ObjectType" to Python 3 object
            dill._dill._reverse_typemap["ObjectType"] = object
            data = pickle.loads(data, encoding='bytes')
            if data[0] == float('inf'):
                self.feasibility = False
            time_list[i] = data[0]
        
        self.ros_time = round(np.sum(time_list), 9)
        tm = time.time()
        if self.replay == None:
            print('Iteration :', self.count, ', Area :', self.area, ', OperatingTime :', self.ros_time, ', Feasibility :', self.feasibility, ', Time :', tm)
            self.output_file.write(str(self.count) + '\t' + str(self.area) + '\t' + str(self.ros_time) + '\t' + str(self.feasibility) + '\t' + str(tm) + '\n')#
            if self.feasibility:#
                self.output_file_.write(str(self.count) + '\t' + str(self.area) + '\t' + str(self.ros_time) + '\t' + str(tm) + '\n')#
        self.finish_flag = True