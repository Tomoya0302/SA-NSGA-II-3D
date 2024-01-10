from layout_generation import Robot
from Seq3 import SequenceTriple
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.sms import SMSEMOA

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.running_metric import RunningMetricAnimation

cube = [{"width": 20, "height":30, "depth": 30},
        {"width": 20, "height":30, "depth": 30},
        {"width": 30, "height":20, "depth": 30},
        {"width": 30, "height":20, "depth": 30},
        {"width": 30, "height":30, "depth": 20},
        {"width": 30, "height":30, "depth": 20},
        {"width": 30, "height":30, "depth": 30},
        {"width": 30, "height":30, "depth": 30},
        {"width": 20, "height":20, "depth": 20},
        {"width": 20, "height":20, "depth": 20},
]

class RobotLayout(ElementwiseProblem):
    
    def __init__(self):
        self.n_rect = len(cube)
        super().__init__(
            n_var=self.n_rect * 3, 
            n_obj=2, 
            n_constr=1
            )
        self.iter = 0
        self.result = np.zeros((1, 2), dtype=float)

    def _evaluate(self, x, out, *args, **kwargs):

        a = x[0:self.n_rect].tolist()
        b = x[self.n_rect: 2*self.n_rect].tolist()
        c = x[2*self.n_rect:3*self.n_rect].tolist()
        a = list(map(int, a))
        b = list(map(int, b))
        c = list(map(int, c))

        layout = SequenceTriple(cube)
        floorplan = layout.main(a, b, c)
        area = layout.volume(floorplan[1],floorplan[2],floorplan[3])
        ros_time, feasibility = robot_real.trajectory_generate(area, floorplan)
        gravity = layout.check_gravity(floorplan[0])
        f1 = area
        f2 = ros_time
        if feasibility and gravity:
            g1 = 0
            self.result = np.append(self.result, [[f1, f2]], axis=0)
        else:
            g1 = 1
        out["F"] =  ([f1, f2])
        out["G"] = g1
        self.iter += 1

    def visualize(self, pr_solution):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(self.result[1:, 0], self.result[1:, 1], marker='.')
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        plt.show()

class LayoutSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        n_rect = problem.n_var // 3
        x1 = np.zeros((n_samples, n_rect))
        x2 = np.zeros((n_samples, n_rect))
        x3 = np.zeros((n_samples, n_rect))
        for i in range(n_samples):
            x1[i, :] = np.random.permutation(n_rect)
            x2[i, :] = np.random.permutation(n_rect)
            x3[i, :] = np.random.permutation(n_rect)
        return np.hstack((x1,x2,x3))

class LayoutOXCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def OrderCrossover(self, receiver, donor, start, end):
        shift = False
        donation = np.copy(donor[start:end + 1])
        donation_as_set = set(donation)
        y = []
        for k in range(len(receiver)):
            i = k if not shift else (start + k) % len(receiver)
            v = receiver[i]
            if v not in donation_as_set:
                y.append(v)
        y = np.concatenate([y[:start], donation, y[start:]]).astype(copy=False, dtype=int)
        return y

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)
        for i in range(n_matings):
            a, b = X[:, i, :]
            n = int(len(a)/3)
            a_dict = {A: B for A, B in zip(a[0:n], a[2*n:3*n])}
            b_dict = {A: B for A, B in zip(b[0:n], b[2*n:3*n])}
            start, end = np.sort(np.random.choice(n, 2, replace=False))
            for j in range(2):
                Y[0, i, j*n:(j+1)*n] = self.OrderCrossover(
                    a[j*n:(j+1)*n], b[j*n:(j+1)*n], start, end)
                Y[1, i, j*n:(j+1)*n] = self.OrderCrossover(
                    b[j*n:(j+1)*n], a[j*n:(j+1)*n], start, end)
            for k in range(n):
                Y[0, i, 2*n + k] = a_dict[Y[0, i, k]]
                Y[1, i, 2*n + k] = b_dict[Y[1, i, k]]
        return Y

class LayoutMutation(Mutation):
    def __init__(self, prob=1.0):
        super().__init__()
        self.prob = prob

    def mutation(self, y, start, end, inplace=True):
        y = y if inplace else np.copy(y)
        y[start:end + 1] = np.flip(y[start:end + 1])
        return y

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        _, n = Y.shape
        n = int(n/3)
        for i, y in enumerate(X):
            if np.random.random() < self.prob:
                start, end = np.sort(np.random.choice(n, 2, replace=False))
                for j in range(3):
                    Y[i, j*n:(j+1)*n] = self.mutation(y[j*n:(j+1)*n], start, end, inplace=True)
        return Y

if __name__ == '__main__':
    alg = 'NSGA2'
    pop_size = 200
    robot = RobotLayout()
    robot_real = Robot()

    if alg == 'NSGA2':
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=LayoutSampling(),
            crossover=LayoutOXCrossover(),
            mutation=LayoutMutation(),
            eliminate_duplicates=True
        )
    elif alg == 'RNSGA2':
        algorithm = RNSGA2(
            ref_points=np.array([[0,0]]),
            pop_size=pop_size,
            epsilon=0.001,
            normalization='front',
            extreme_points_as_reference_points=False,
            sampling=LayoutSampling(),
            crossover=LayoutOXCrossover(),
            mutation=LayoutMutation(),
            eliminate_duplicates=True
        )
    elif alg == 'UNSGA3':
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
        algorithm = UNSGA3(
            ref_dirs,
            pop_size=pop_size,
            epsilon=0.001,
            sampling=LayoutSampling(), 
            crossover=LayoutOXCrossover(),
            mutation=LayoutMutation(),
            eliminate_duplicates=True
        )
    elif alg == 'CTAEA':
        ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size)
        algorithm = CTAEA(
            ref_dirs,
            sampling=LayoutSampling(),
            crossover=LayoutOXCrossover(),
            mutation=LayoutMutation(),
            eliminate_duplicates=True
        )
    elif alg == 'SMSMEOA':
        algorithm = SMSEMOA(
            pop_size=pop_size,
            sampling=LayoutSampling(),
            crossover=LayoutOXCrossover(),
            mutation=LayoutMutation(),
            eliminate_duplicates=True
        )
    res = minimize(
        robot,
        algorithm,
        ('n_gen', 750),
        seed=1,
        verbose=True,
        save_history=True
    )
    running = RunningMetricAnimation(
        delta_gen=125,
        n_plots=6,
        key_press=False,
        do_show=True
    )

    print(type(res))
    print(type(res.F))
    print('resF', res.F)
    print('resX', res.X)
    np.save('floorplan_log/resf', res.F)
    np.save('floorplan_log/resx', res.X)

    Scatter().add(res.F).show()

    SHOW_FIG = True
    n_solution, n_var = res.X.shape
    n_box = n_var//3

    for algorithm in res.history:
        running.update(algorithm)
    for i in range(n_solution):
        x = res.X[i, :]
        a = x[0:n_box].tolist()
        b = x[n_box: 2*n_box].tolist()
        c = x[2*n_box:3*n_box].tolist()
        a = list(map(int, a))
        b = list(map(int, b))
        c = list(map(int, c))
        layout = SequenceTriple(cube)
        pr_solution = layout.main(a, b, c) 
        layout.visualize(pr_solution[0])
    robot.visualize(pr_solution[0])
    robot_real.close_connection()