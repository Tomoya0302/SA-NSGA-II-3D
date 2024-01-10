import matplotlib.pyplot as plt
import numpy as np
import random
import graphlib
from typing import Any, Dict, List, Optional, Tuple

class SequenceTriple:
    def __init__(self, cube):
        self.cube = cube
        self.num_object = len(self.cube)

    def main(self, a, b, c):
        critical = {}
        tree = {}
        pr_solution = {}
        graph = {}

        for idx, direct in enumerate(['x', 'z', 'y']):
            criticalpath = []

            check = list(np.arange(self.num_object))
            if direct == 'x':
                for i in range(self.num_object):
                    x = b[i]
                    check.remove(b[i])
                    x_idx_3 = c.index(b[i])
                    for j in range(len(check)):
                        y = check[j]
                        y_idx_3 = c.index(y)
                        if x_idx_3 > y_idx_3:
                            r = str(b[i]) + ">" + str(check[j])
                            criticalpath.append(r)

            else:
                for i in range(self.num_object):
                    if direct == 'y':
                        x = a[i]
                    else:
                        x = a[-i-1]
                    check.remove(x)
                    x_idx_2 = b.index(x)
                    for j in range(len(check)):
                        y = check[j]
                        y_idx_2 = b.index(y)
                        if x_idx_2 > y_idx_2:
                            x_idx_3 = c.index(x)
                            y_idx_3 = c.index(y)
                            if x_idx_3 > y_idx_3:
                                r = str(x) + ">" + str(y)
                                criticalpath.append(r)
            critical[direct] = criticalpath
            tree_ = np.zeros((self.num_object, self.num_object),dtype = int) 
            graph_: Dict[int, List] = {i: [] for i in range(self.num_object)}
            for i in range(self.num_object):
                for j in range(self.num_object):
                    r = str(j) + '>' + str(i)
                    if r in critical[direct]:
                        tree_[i][j] = 1
                        graph_[j].append(i)
            tree[direct] = tree_
            graph[direct] = graph_ 
            topo_h = graphlib.TopologicalSorter(graph[direct])
            torder_h = list(topo_h.static_order())
                
            # # Calculate W (bounding box width) from G_h 
            if direct == 'x':
                dist_x = [self.cube[i]["width"] for i in range(self.num_object)]
                graph_x = graph[direct]
                for i in torder_h:
                    dist_x[i] += max([dist_x[e] for e in graph_x[i]], default=0)
                bb_width = max(dist_x)

            elif direct == 'y':
                dist_y = [self.cube[i]["height"] for i in range(self.num_object)]
                graph_y = graph[direct]
                for i in torder_h:
                    dist_y[i] += max([dist_y[e] for e in graph_y[i]], default=0)
                bb_height = max(dist_y)

            elif direct == 'z':    
                dist_z = [self.cube[i]["depth"] for i in range(self.num_object)]
                graph_z = graph[direct]
                for i in torder_h:
                    dist_z[i] += max([dist_z[e] for e in graph_z[i]], default=0)
                bb_depth = max(dist_z)
            
        # Calculate bottom-left positions
        positions = []
        for i in range(self.num_object):
            positions.append(
                {
                    "id": i,
                    "x": dist_x[i] - self.cube[i]["width"],  # distance from left edge
                    "y": dist_y[i] - self.cube[i]["height"],  # distande from bottom edge
                    "z": dist_z[i] - self.cube[i]["depth"],  # distande from front edge
                    "width": self.cube[i]["width"],
                    "height": self.cube[i]["height"],
                    "depth": self.cube[i]["depth"],
                }
            )
        return (positions, bb_width, bb_height, bb_depth)

    def volume(self, cube_length_x, cube_length_y, cube_length_z):
        cube_volume = cube_length_x * cube_length_y * cube_length_z
        return cube_volume 

    def check_gravity(self, solution):
        for i in range(len(solution)):
            if not solution[i]['y'] == 0:
                flag = False
                x_g = solution[i]['x'] + (solution[i]['width'] / 2)
                z_g = solution[i]['z'] + (solution[i]['depth'] / 2)
                y_g = solution[i]['y']
                for j in range(len(solution)):
                    if not solution[i]['id'] == solution[j]['id']:
                        if y_g == solution[j]['y'] + solution[j]['height']:
                            xx_min = solution[j]['x']
                            xx_max = solution[j]['x'] + solution[j]['width']
                            zz_min = solution[j]['z']
                            zz_max = solution[j]['z'] + solution[j]['depth']
                            if x_g >= xx_min and x_g <= xx_max and z_g >= zz_min and z_g <= zz_max:
                                flag = True
                if not flag:
                    return False
        return True

    def check(self, pr_solution):
        return True

    def visualize(self, pr_solution):
        color = []
        for num in range(self.num_object):
            color.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection='3d')

        positions = np.zeros((6, self.num_object))
        for i in range(self.num_object):
            positions[0][i] = (pr_solution[i]['x'])
            positions[1][i] = (pr_solution[i]['y'])
            positions[2][i] = (pr_solution[i]['z'])
            positions[3][i] = (pr_solution[i]['width'])
            positions[4][i] = (pr_solution[i]['height'])
            positions[5][i] = (pr_solution[i]['depth'])

        ax.bar3d(positions[0], positions[2], positions[1], 
                positions[3], positions[5], positions[4],
                color=color, alpha=0.5, edgecolor='black')
        
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)
        ax.set_title("layout", fontsize='20')
        plt.show()