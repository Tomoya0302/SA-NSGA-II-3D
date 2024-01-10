import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import graphlib
from typing import Any, Dict, List, Optional, Tuple

cube = [{"width": 20, "height":21, "depth": 12},
        {"width": 20, "height":21, "depth": 12},
        {"width": 20, "height":21, "depth": 22},
        {"width": 20, "height":21, "depth": 22},
        {"width": 20, "height":21, "depth": 22},
        {"width": 30, "height":31, "depth": 32},
        {"width": 30, "height":31, "depth": 32},
        {"width": 30, "height":31, "depth": 32},
        {"width": 40, "height":41, "depth": 42},
        {"width": 40, "height":41, "depth": 42},
        {"width": 50, "height":51, "depth": 52},
        {"width": 50, "height":51, "depth": 52},
        ]

class SequenceTriple:
    def __init__(self, cube):
        self.cube = cube
        self.num_object = len(self.cube)

    def main(self, a, b, c):
        critical = {}
        tree = {}
        pr_solution = {}
        graph = {}
        # print('a', a)
        # print('b', b)
        # print('c', c)

        for idx, direct in enumerate(['x', 'z', 'y']):
            print('direct:', direct)
            # criticalpath
            criticalpath = []

            # check = list(np.arange(self.num_object) + 1) #Seq3だけで動かした時、1スタート
            check = list(np.arange(self.num_object)) #layout.pyと繋いだ時、0スタート
            if direct == 'x':
                for i in range(self.num_object):
                    x = b[i] #配列bのi番目の値をxへ代入
                    # print('x:', x)
                    check.remove(b[i]) #checkから配列bのi番目の値を削除
                    x_idx_3 = c.index(b[i]) #配列bのi番目と同じ値を配列cで探し、index番号をx_idx_3に代入
                    for j in range(len(check)): #配列bのi番目以外の値がcheckに入っている。
                        # if j == x: #skip
                        #     continue
                        y = check[j]
                        # print(y)
                        y_idx_3 = c.index(y)
                        if x_idx_3 > y_idx_3: #配列cの2つの値のインデックス番号を比較する
                            r = str(b[i]) + ">" + str(check[j])
                            criticalpath.append(r)
                # print(criticalpath)


            else:
                for i in range(self.num_object):#何を回している？
                    if direct == 'y':
                        x = a[i] #配列aのi番目の値をxへ代入
                        # print('y:', x)
                    else:
                        x = a[-i-1] #配列aのi番目の値をxへ代入
                        # print('z:', x)
                    check.remove(x) #checkから配列bのi番目の値を削除
                    x_idx_2 = b.index(x) #配列aのi番目と同じ値を配列bで探し、index番号をx_idx_2に代入
                    for j in range(len(check)):
                        y = check[j]#j=1~4?or0~3?
                        y_idx_2 = b.index(y)
                        if x_idx_2 > y_idx_2: #配列bの2つの値のインデックス番号を比較する
                            x_idx_3 = c.index(x) #配列aのi番目と同じ値を配列cで探し、index番号をx_idx_3に代入
                            y_idx_3 = c.index(y) #配列checkのj番目と同じ値を配列cで探し、index番号をx_idx_3に代入
                            if x_idx_3 > y_idx_3: #配列cの2つの値のインデックス番号を比較する
                                r = str(x) + ">" + str(y)
                                criticalpath.append(r)
                # print('criticalpath',criticalpath)
            critical[direct] = criticalpath
            # print('critical',critical)

            # graph 
            tree_ = np.zeros((self.num_object, self.num_object),dtype = int)
            graph_: Dict[int, List] = {i: [] for i in range(self.num_object)}
            for i in range(self.num_object):
                for j in range(self.num_object):
                    # r = str(j+1) + '>' + str(i+1) #Seq3だけで動かした時、1スタート
                    r = str(j) + '>' + str(i) #layout_opt.pyだけで動かした時、0スタート
                    if r in critical[direct]:
                        tree_[i][j] = 1
                        # print(r)
                        graph_[j].append(i)
            tree[direct] = tree_
            # print('tree_after', tree_)
            graph[direct] = graph_ 
            # print('tree', tree)
            print('graph_ 大:小', graph_) #12345が01234になっている
            # print('graph' , graph) ###12345が01234になっている
            # print(type(graph))
          
            # Topological order of DAG (G_h)
            topo_h = graphlib.TopologicalSorter(graph[direct])
            # print('topo_h:', topo_h)
            torder_h = list(topo_h.static_order())
            print('torder:', torder_h)
                
            # # Calculate W (bounding box width) from G_h
            if direct == 'x':
                dist_x = [self.cube[i]["width"] for i in range(self.num_object)]
                print('dist_x:', dist_x)
                graph_x = graph[direct]
                print(graph_x)
                for i in torder_h: #最大値をとる
                    dist_x[i] += max([dist_x[e] for e in graph_x[i]], default=0)
                    # print('dist_x_added:', dist_x[i])
                bb_width = max(dist_x)
                print('bb_width:', bb_width)

            elif direct == 'y':
                dist_y = [self.cube[i]["height"] for i in range(self.num_object)]
                print('dist_y:', dist_y)
                graph_y = graph[direct]
                print(graph_y)
                for i in torder_h:
                    dist_y[i] += max([dist_y[e] for e in graph_y[i]], default=0)
                    # print('dist_y_added:', dist_y[i])
                bb_height = max(dist_y)
                print('bb_height:', bb_height)

            elif direct == 'z':    
                dist_z = [self.cube[i]["depth"] for i in range(self.num_object)]
                print('dist_z:', dist_z)
                graph_z = graph[direct]
                print(graph_z)
                for i in torder_h:
                    dist_z[i] += max([dist_z[e] for e in graph_z[i]], default=0)
                    # print('dist_z_added:', dist_z[i])
                bb_depth = max(dist_z)
                print('bb_depth:', bb_depth)
            
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
        # print('position', positions)
            # return Floorplan(bounding_box=(bb_width, bb_height, bb_depth), positions=positions)
        return (positions, bb_width, bb_height, bb_depth)

    def volume(self, cube_length_x, cube_length_y, cube_length_z):
        cube_volume = cube_length_x * cube_length_y * cube_length_z
        return cube_volume
                

        '''# graph -> parameters
            pr_solution_ = np.zeros(self.num_object, dtype=int) #pr_solution：provisional solution 暫定解
            cp = np.full(self.num_object, -1, dtype=int)
            for i in range(self.num_object):
                for j in range(self.num_object):
                    if tree[direct][i][j] == 1:
                        r = pr_solution_[i] + self.cube[i][idx]
                        if r > pr_solution_[j]:
                            pr_solution_[j] = r
                            cp[j] = i
                            pr_solution_ = np.where(cp==j, pr_solution_+r, pr_solution_)
            pr_solution[direct] = pr_solution_
        # print(pr_solution)
        # new_key_assign = {'x':'x', 'y':'z', 'z':'y'}
        # pr_solution = {new_key_assign.get(key):value for key, value in pr_solution.items()}
        return pr_solution'''

    def check(self, pr_solution):
        # pr_np = np.rot90(np.array([list(pr_solution[i]) for i in ['x', 'y', 'z']]))
        # for num in range(self.num_object - 1):
        #     if pr_np[num] in pr_np[num+1:]:
        #         print(pr_np[num])
        #         print(pr_np[num+1:])
        #         return False
        return True
    '''
    def get_left_bottom_front_edge(self, pr_solution):
        orderedNames = list(pr_solution.keys())
        dataMatrix = np.array([list(pr_solution[i]) for i in orderedNames])
        print(dataMatrix)
        return dataMatrix

    def get_g(self, cube, left_bottom_front_edge):
        # orderedNames = list(pr_solution.keys())
        # dataMatrix = np.array([list(pr_solution[i]) for i in orderedNames])
        # print(dataMatrix)
        center_of_gravity_x_z = left_bottom_front_edge
        print(center_of_gravity_x_z)
        center_of_gravity_x_z[0] += ((np.array(cube)/2).T).astype('int32')[0] #cubeを転置して座標に代入している
        center_of_gravity_x_z[2] += ((np.array(cube)/2).T).astype('int32')[2] #cubeを転置して座標に代入している
        # dataMatrix[0] += ((np.array(cube)/2).T).astype('int32')[0] #cubeを転置して座標に代入している
        # dataMatrix[2] += ((np.array(cube)/2).T).astype('int32')[2]
        print(center_of_gravity_x_z)
        return center_of_gravity_x_z
    
    def get_right_top_behind_edge(self, cube, left_bottom_front_edge):
        print('right_top_behind')
        right_top_behind_edge = left_bottom_front_edge
        print(right_top_behind_edge)
        right_top_behind_edge[0] += ((np.array(cube)).T).astype('int32')[0] #cubeを転置して座標に代入している
        right_top_behind_edge[1] += ((np.array(cube)).T).astype('int32')[1] #cubeを転置して座標に代入している
        right_top_behind_edge[2] += ((np.array(cube)).T).astype('int32')[2] #cubeを転置して座標に代入している
        print(right_top_behind_edge)
        return right_top_behind_edge


    #作成中
    def feasible_g(self, pr_solution, cube, left_bottom_front_edge, center_of_gravity_x_z, right_top_behind_edge):
    #     #浮いているか判別
    #     top = [] 
    #     ground = []
    #     floating = []
    #     print(top)
          # all_cube_number = 0
    #     # while all_cube_number < len(dataMatrix[1]):
    #         # if dataMatrix[1] == 0:
    #             # print(dataMatrix[1])
    #         # else: 
    #             # for i in dataMatrix[1]:
    #                 # if i == top:
    #                     # if dataMatrix[0]:
    #         # all_cube_number += 1
    #             # ground.append('true')
    #             # break

        for i in left_bottom_front_edge[1]: #cubeの下面をすべて回している
            # print(i)
            print('youso')
            youso = left_bottom_front_edge[1, 0]
            print(youso)
            # print(left_bottom_front_edge[0][0])
            # if i == 0:
            #     print('It is 0')
            # elif i != 0:
            #     print('It is not 0')
            #     above_cube = i
            #     # print(floating_cube)
            #     for j in right_top_behind_edge[1]: #cubeの上面をすべて回している
            #         if j != above_cube:
            #             print('It is not feasibility')
            #         elif j == above_cube:
            #             # for k in center_of_gravity_x_z[0]: #cubeの重心が他のx,zの範囲内にあるかを判別する
            #                 # if k > left_bottom_front_edge[0][i]
            #             print('2floar desune!')


    #             # print(i)
    #             # cube_g = i
    #             # for j in upper:
    #                 # if j == cube_g:
    #                     ##高さが一致した時、XとZの幅の中に重心があるか、min_x < g_x < max_x, min_z < g_z < max_z

'''

    def visualize(self, pr_solution):
        print(pr_solution)
        # print(cube)
        color = []
        # color = ['red', 'blue', 'yellow', 'black', 'orange']
        color = ['royalblue', 'gold', 'brown', 'orangered', 'lightsalmon', 'greenyellow', 'limegreen', 'green', 'darkcyan', 'slategrey', 'lightpink', 'indigo']
        # for num in range(self.num_object):
            # color.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
        fig = plt.figure(figsize=(4, 4)) # 図の設定
        ax = fig.add_subplot(projection='3d') # 3D用の設定

        positions = np.zeros((6, self.num_object))
        for i in range(self.num_object):
            positions[0][i] = (pr_solution[i]['x'])
            positions[1][i] = (pr_solution[i]['y'])
            positions[2][i] = (pr_solution[i]['z'])
            positions[3][i] = (pr_solution[i]['width'])
            positions[4][i] = (pr_solution[i]['height'])
            positions[5][i] = (pr_solution[i]['depth'])

        # ax.bar3d(positions[0], positions[1], positions[2], 
        #         positions[3], positions[4], positions[5],
        #         color=color, alpha=0.5, edgecolor='black') #12/16(positionのz,yが逆)

        ax.bar3d(positions[0], positions[2], positions[1], 
                positions[3], positions[5], positions[4],
                color=color, alpha=0.5, edgecolor='black')
        
        # ax.bar3d(pr_solution['x'], pr_solution['y'], pr_solution['z'], 
        #         np.rot90(self.cube)[2], np.rot90(self.cube)[0], np.rot90(self.cube)[1], 
        #         color=color, alpha=0.5, edgecolor='black')
        ax.set_xlabel('x') # x軸ラベル(xと表示,横幅)
        ax.set_ylabel('z') # y軸ラベル(zと表示,奥行)
        ax.set_zlabel('y') # z軸ラベル(yと表示,高さ)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)
        ax.set_title("layout", fontsize='20') # タイトル
        plt.show() # 描画

if __name__ == '__main__':

    # cube = [[10, 11, 12],
    #         [20, 21, 22],
    #         [30, 31, 32],
    #         [40, 41, 42],
    #         [50, 51, 52]]

    # cube = [{"width": 10, "height":11, "depth": 12},
    #         {"width": 20, "height":21, "depth": 22},
    #         {"width": 30, "height":31, "depth": 32},
    #         {"width": 40, "height":41, "depth": 42},
    #         {"width": 50, "height":51, "depth": 52},
    #         ]

    # a = [5, 1, 2, 3, 4]
    # b = [4, 2, 3, 5, 1]
    # c = [4, 3, 2, 5, 1]

    # a = [2, 4, 3, 1, 5]
    # b = [3, 4, 5, 2, 1]
    # c = [5, 4, 1, 3, 2]
    
    sequencetriple = SequenceTriple(cube)
    for i in range(10):
        a = list(np.arange(len(cube))+1)
        b = list(np.arange(len(cube))+1)
        c = list(np.arange(len(cube))+1)
        np.random.shuffle(a)
        np.random.shuffle(b)
        np.random.shuffle(c)
        pr_solution = []
        pr_solution = sequencetriple.main(a, b, c)
        print('pr_solution', pr_solution)
        cube_volume = sequencetriple.volume(pr_solution[1], pr_solution[2], pr_solution[3])
        print('cube_volume', cube_volume)
        # positions = sequencetriple.main(a, b, c)
        # print('Feasibility :', sequencetriple.check(pr_solution), pr_solution)
        # print(a)
        # print(b)
        # print(c)
        sequencetriple.visualize(pr_solution[0])
        # sequencetriple.visualize(positions)

'''
        left_bottom_front_edge = sequencetriple.get_left_bottom_front_edge(pr_solution)
        center_of_gravity_x_z = sequencetriple.get_g(cube, left_bottom_front_edge)
        right_top_behind_edge = sequencetriple.get_right_top_behind_edge(cube, left_bottom_front_edge)
        sequencetriple.feasible_g(pr_solution, cube, left_bottom_front_edge, center_of_gravity_x_z, right_top_behind_edge)
'''