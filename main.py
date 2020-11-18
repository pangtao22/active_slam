import time

import numpy as np
import numpy.random as random
import meshcat

from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.gurobi import GurobiSolver
import pydrake.solvers.mathematicalprogram as mp
from pydrake.symbolic import sqrt


#%%
class SlamFrontend:
    def __init__(self, num_landmarks: int, seed: int):
        random.seed(seed)

        self.nl = num_landmarks
        # edge length of the square in which the robot runs.
        bbox_length = 5.
        self.x_min = -bbox_length / 2
        self.y_min = -bbox_length / 2
        self.x_max = bbox_length / 2
        self.y_max = bbox_length / 2
        self.box_length = bbox_length

        self.r_sensor = bbox_length / 5

        # coordinates of landmarks.
        self.l_xy = random.rand(self.nl, 2) * bbox_length - bbox_length / 2

        # visualizer
        self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.vis.delete()
        self.draw_landmarks()

        # initialize robot
        self.X_WB = meshcat.transformations.rotation_matrix(
            np.pi/2, np.array([1., 0, 0]), np.zeros(3))
        material = meshcat.geometry.MeshLambertMaterial(
            color=0xfcfcfc, opacity=0.3)
        self.vis["robot"].set_object(
            meshcat.geometry.Cylinder(height=0.05, radius=self.r_sensor),
            material)
        self.vis["robot"].set_transform(self.X_WB)

        # noise
        self.sigma_odometry = 0.05
        self.sigma_range = 0.05

    def draw_landmarks(self):
        l_xyz = np.zeros((self.nl, 3))
        l_xyz[:, :2] = self.l_xy
        colors = np.zeros_like(l_xyz.T)
        colors[2] = 1.
        self.vis["landmarks"].set_object(
            meshcat.geometry.PointCloud(position=l_xyz.T,
                                        color=colors,
                                        size=0.1))

    def draw_robot(self, t_xy):
        """
        Updates robot pose in mehscat.
        :param t_xy: robot position in world frame.
        :return: None.
        """
        self.X_WB[0:2, 3] = t_xy
        self.vis["robot"].set_transform(self.X_WB)

    def get_odometry_measurements(self, t_xy, t_xy_previous):
        return t_xy - t_xy_previous + random.normal(
            scale=self.sigma_odometry, size=t_xy.size)

    def get_range_measurements(self, t_xy):
        """
        :param t_xy: robot pose (position) in world frame.
        :return:
        (1) idx_visible_l: indices into self.l_xy whose distance to t_xy is
            smaller than self.r_sensor.
        (2) noisy measurements of distances to self.l_xy[idx_visible_l]
        """
        l_xy_B = self.l_xy - t_xy
        d_l = np.linalg.norm(l_xy_B, axis=1)
        idx_visible_l = np.where(d_l < self.r_sensor)[0]
        d_l_noisy = d_l[idx_visible_l] + random.normal(
            scale=self.sigma_range, size=idx_visible_l.size)
        return idx_visible_l, d_l_noisy


class SlamBackend:
    def __init__(self, t_xy_e_0, frontend: SlamFrontend):
        """

        :param t_xy_e_0: initial estimated robot position,
            used to anchor the entire trajectory.
        """
        self.solver = GurobiSolver()
        self.t_xy_e_0 = t_xy_e_0.copy()

        # current beliefs
        self.t_xy_e_dict = {0: self.t_xy_e_0}
        self.l_xy_e_dict = dict()

        # past measurements
        self.odometry_measurements = dict()

        # key (int): landmark index
        # value: dict of (t: d). t: index of robot poses from which the
        #   landmark is visible. d: range measurement for the landmark from
        #   pose indexed by t.
        self.landmark_visibility = dict()

        # Taking stuff from frontend
        self.vis = frontend.vis
        self.l_xy = frontend.l_xy
        #TODO: load from config file?
        self.sigma_odometry = frontend.sigma_odometry
        self.sigma_range = frontend.sigma_range

    def update_landmark_measurements(
            self, t, idx_visible_l_list, d_l_measured_list):
        """
        inputs are the output of Frontend.get_range_measurements()
        :return: None.
        """
        for i_l, d_l in zip(idx_visible_l_list, d_l_measured_list):
            if i_l not in self.landmark_visibility:
                self.landmark_visibility[i_l] = dict()
            self.landmark_visibility[i_l][t] = d_l

    def update_odometry_measruements(self, t, odometry_measurement):
        """

        :param t: current time step (starting from 0).
        :param odometry_measurement: odometry measruement between t and t-1.
        :return:
        """
        self.odometry_measurements[t] = odometry_measurement

    def get_t_xy_initial_guess_array(self):
        """
        Use self.t_xy_e_dict to create an initial guess for the next bundle
            adjustment.
        Let t be the current exploration time step. As t_xy_e_dict is the
            estimate for time step t-1, its shape is (t - 1, 2), which does
            not include an estimate for the pose of the current time step.

        The initial guess for the current time step is computed as
            self.t_xy_e_dict[t - 1] + self.odometry_measurements[t].
        :return:
        """
        n_o = len(self.odometry_measurements)
        t_xy_e = np.zeros((n_o + 1, 2))
        for i in range(n_o):
            t_xy_e[i] = self.t_xy_e_dict[i]
        if n_o > 0:
            t_xy_e[n_o] = t_xy_e[n_o - 1] + self.odometry_measurements[n_o]

        return t_xy_e

    def get_l_xy_initial_guess_array(self):
        nl = len(self.landmark_visibility)
        l_xy_e = np.zeros((nl, 2))
        for i_k, k in enumerate(self.landmark_visibility.keys()):
            if k in self.l_xy_e_dict:
                l_xy_e[i_k] = self.l_xy_e_dict[k]
        return l_xy_e

    def draw_estimated_landmarks(self):
        nl = len(self.l_xy_e_dict)
        l_xy_e = np.zeros((nl, 3))
        for i, (k, l_xy_e_i) in enumerate(self.l_xy_e_dict.items()):
            l_xy_e[i][:2] = l_xy_e_i
            points = np.zeros((2, 3))
            points[0, :2] = l_xy_e_i
            points[1, :2] = self.l_xy[k]
            self.vis["landmarks_e"]["corrspondances"][str(k)].set_object(
                meshcat.geometry.Line(
                    meshcat.geometry.PointsGeometry(points.T)))

        point_colors = np.zeros_like(l_xy_e)
        point_colors[:, 1] = 1
        point_colors[:, 2] = 1
        self.vis["landmarks_e"]["points"].set_object(
            meshcat.geometry.PointCloud(
                position=l_xy_e.T, color=point_colors.T, size=0.1))

    def draw_robot_path(self, t_xy, color, prefix: str):
        for i in range(1, len(t_xy)):
            points = np.zeros((2, 3))
            points[0, :2] = t_xy[i - 1]
            points[1, :2] = t_xy[i]
            self.vis[prefix]["path"][str(i)].set_object(
                meshcat.geometry.Line(
                    meshcat.geometry.PointsGeometry(points.T)))

        t_xy3 = np.vstack((t_xy.T, np.zeros(len(t_xy)))).T
        point_colors = np.zeros_like(t_xy3)
        point_colors[:] = color
        self.vis[prefix]["points"].set_object(
            meshcat.geometry.PointCloud(
                position=t_xy3.T, color=point_colors.T, size=0.2))

    def draw_estimated_path(self):
        nt = len(self.t_xy_e_dict)
        t_xy_e = np.zeros((nt, 2))
        for i in range(nt):
            t_xy_e[i] = self.t_xy_e_dict[i]
        self.draw_robot_path(t_xy_e, color=[1, 0, 0], prefix="robot_poses_e")

    def calc_jacobian_and_b(self, t_xy_e, l_xy_e):
        """
        So that the linearization of
        :param t_xy_e (n_o + 1, 2)
        :param l_xy_e (nl, 2): coordinates of landmarks, ordered the same way as
            self.landmark_visibility.
        :return:
        """
        nl = len(self.landmark_visibility)  # number of landmarks
        n_o = len(self.odometry_measurements)  # number of odometries
        nl_measurements = 0  # number of landmark measurements
        for visible_landmarks_i in self.landmark_visibility.values():
            nl_measurements += len(visible_landmarks_i)

        n_rows = n_o * 2 + nl_measurements
        n_cols = (n_o + 1 + nl) * 2
        J = np.zeros((n_rows, n_cols))
        b = np.zeros(n_rows)

        # odometry
        for i in range(n_o):
            i0 = i * 2
            i1 = i0 + 2

            J[i0: i1, i0: i1] = -np.eye(2)
            J[i0: i1, i0 + 2: i1 + 2] = np.eye(2)
            b[i0: i1] = t_xy_e[i + 1] - t_xy_e[i] - \
                self.odometry_measurements[i + 1]

        i_row = n_o * 2
        for i_k, (k, visible_landmark) in enumerate(
                self.landmark_visibility.items()):
            for i, d_i in visible_landmark.items():
                dh = t_xy_e[i] - l_xy_e[i_k]
                h = np.sqrt((dh**2).sum())
                dh /= 2 * h

                J[i_row, i * 2: (i + 1) * 2] = dh

                j0 = (n_o + 1 + i_k) * 2
                j1 = j0 + 2
                J[i_row, j0: j1] = -dh

                b[i_row] = h - self.landmark_visibility[k][i]

                i_row += 1

        return J, b

    def run_bundle_adjustment(self):
        t_xy_e0 = self.get_t_xy_initial_guess_array()
        l_xy_e0 = self.get_l_xy_initial_guess_array()
        t_xy_e = t_xy_e0.copy()
        l_xy_e = l_xy_e0.copy()

        n_o = len(self.odometry_measurements)
        n_l = len(self.landmark_visibility)
        # Js = []

        steps_counter = 0
        while True:
            J, b = self.calc_jacobian_and_b(t_xy_e, l_xy_e)
            prog = mp.MathematicalProgram()

            dt_xy_e = prog.NewContinuousVariables(n_o + 1, 2, "dt_xy_e")
            dl_xy_e = prog.NewContinuousVariables(n_l, 2, "dl_xy_e")
            dx = np.hstack((dt_xy_e.ravel(), dl_xy_e.ravel()))

            nx = len(dx)
            prog.AddL2NormCost(J, -b, dx)
            prog.AddQuadraticCost((dx**2).sum())
            # prog.AddQuadraticCost(np.eye(nx) * 2, np.zeros(nx), dx)
            prog.AddQuadraticCost(((dt_xy_e[0]) ** 2).sum())

            result = self.solver.Solve(prog)
            optimal_cost = result.get_optimal_cost() / (n_o + 1)
            assert result.get_solution_result() == \
                   mp.SolutionResult.kSolutionFound

            # print("\nStep ", steps_counter)
            # print("optimal cost: ", result.get_optimal_cost())
            # print("jacobian norm", np.linalg.norm(J))
            # print("robot positions\n", t_xy_e)
            # print("landmark positions\n", l_xy_e)

            dt_xy_e_values = result.GetSolution(dt_xy_e)
            dl_xy_e_values = result.GetSolution(dl_xy_e)

            t_xy_e += dt_xy_e_values
            l_xy_e += dl_xy_e_values
            # Js.append(J)

            steps_counter += 1
            if optimal_cost < 1e-3 or steps_counter > 200:
                break

        self.update_beliefs(t_xy_e, l_xy_e)

        print("\nStep ", n_o)
        print("optimal cost: ", optimal_cost)
        print("total gradient steps: ", steps_counter)
        print("gradient norm: ", np.linalg.norm(J.T.dot(b)))
        print("\n")

    def update_beliefs(self, t_xy_e_values, l_xy_e_values):
        for i, t_xy_e_value in enumerate(t_xy_e_values):
            if i > 0:
                self.t_xy_e_dict[i] = t_xy_e_value

        self.l_xy_e_dict.clear()
        for k, l_xy_e_value in zip(self.landmark_visibility.keys(),
                                   l_xy_e_values):
            self.l_xy_e_dict[k] = l_xy_e_value

    def run_bundle_adjustment_snopt(self):
        n_o = len(self.odometry_measurements)

        prog = mp.MathematicalProgram()

        # prior for first pose.
        t_xy_e = prog.NewContinuousVariables(n_o + 1, 2, "t_xy_e")
        prog.AddQuadraticCost(
            ((t_xy_e[0] - self.t_xy_e_dict[0]) ** 2).sum() * 10000)

        # landmarks
        n_l = len(self.landmark_visibility)
        l_xy_e = prog.NewContinuousVariables(n_l, 2, "l_xy_e")
        s_l = prog.NewContinuousVariables(n_l, "s_l")  # slack variables.

        indices_landmark = []

        for i_k, (k, t_dict) in enumerate(self.landmark_visibility.items()):
            # print(i_k, k, pose_list)
            indices_landmark.append(k)
            for i_t, d_l_measured in t_dict.items():
                # def d_cost(z):
                #     t_xy_e_i = z[:2]
                #     l_xy_e_i = z[2:]
                #     e = d_l_measured**2 - ((t_xy_e_i - l_xy_e_i) ** 2).sum()
                #     e /= self.sigma_range ** 2
                #     return e**2
                #
                # prog.AddCost(d_cost,
                #              vars=[t_xy_e[i_t][0], t_xy_e[i_t][1],
                #                    l_xy_e[i_k][0], l_xy_e[i_k][1]])
                t_xy_e_i = t_xy_e[i_t]
                l_xy_e_i = l_xy_e[i_k]
                e = d_l_measured ** 2 - ((t_xy_e_i - l_xy_e_i) ** 2).sum()
                e /= self.sigma_range ** 2
                prog.AddConstraint(s_l[i_k] >= e)
                prog.AddConstraint(s_l[i_k] >= -e)

                prog.AddCost(s_l[i_k])

        # odometry
        for i, odometry_measurement in self.odometry_measurements.items():
            dt = t_xy_e[i] - t_xy_e[i-1] - odometry_measurement
            prog.AddCost((dt**2).sum() / self.sigma_odometry ** 2)

        # initial guess
        for i in range(len(self.t_xy_e_dict)):
            prog.SetInitialGuess(t_xy_e[i], self.t_xy_e_dict[i])

        for i_k, k in enumerate(indices_landmark):
            if k in self.l_xy_e_dict:
                prog.SetInitialGuess(l_xy_e[i_k], self.l_xy_e_dict[k])

        # %% save solution
        result = self.solver.Solve(prog)
        print(result.get_solution_result())
        assert result.get_solution_result() == mp.SolutionResult.kSolutionFound

        t_xy_e_values = result.GetSolution(t_xy_e)
        l_xy_e_values = result.GetSolution(l_xy_e)
        self.update_beliefs(t_xy_e_values, l_xy_e_values)

        print("Step: ", n_o)
        print("max error for landmarks:", np.max(result.GetSolution(s_l)))
        print("robot pose estimates\n", t_xy_e_values)
        print("landmark estimates\n", l_xy_e_values)


if __name__ == "__main__":
    #%%
    t_xy = np.zeros((9, 2))
    t_xy[0] = np.array([1.5, -1.3])
    t_xy[1] = t_xy[0] + [0, 1.]
    t_xy[2] = t_xy[1] + [-0.1, 0]
    t_xy[3] = t_xy[2] + [-0.3, 0]
    t_xy[4] = t_xy[3] + [-0.5, 0.5]
    t_xy[5] = t_xy[4] + [-1, -0.2]
    t_xy[6] = t_xy[5] + [-1, -1]
    t_xy[7] = t_xy[6] + [0, 0.5]
    t_xy[8] = t_xy[7] + [1.5, 2.5]

    frontend = SlamFrontend(num_landmarks=50, seed=16485)
    backend = SlamBackend(t_xy[0], frontend)

    frontend.draw_robot(t_xy[0])
    print(frontend.get_range_measurements(t_xy[0]))
    backend.draw_robot_path(t_xy, color=[0, 1, 0], prefix="robot_gt")


#%% first measurement
    for t in range(len(t_xy)):
        idx_visible_l_list, d_l_measured_list = \
            frontend.get_range_measurements(t_xy[t])
        backend.update_landmark_measurements(
            t, idx_visible_l_list, d_l_measured_list)

        if t > 0:
            odometry_measurement = frontend.get_odometry_measurements(
                t_xy=t_xy[t], t_xy_previous=t_xy[t - 1])
            backend.update_odometry_measruements(t, odometry_measurement)

        try:
            backend.run_bundle_adjustment()
        except AssertionError:
            break

        frontend.draw_robot(backend.t_xy_e_dict[t])
        backend.draw_estimated_path()
        backend.draw_estimated_landmarks()

        input("next?")

    #%%
    # backend.draw_estimated_robots()
    # backend.draw_robots(t_xy, color=[0, 1, 0], prefix="robot_gt")

#%%
    # t_xy_e0 = backend.get_t_xy_initial_guess_array()
    # l_xy_e0 = backend.get_l_xy_initial_guess_array()
    # t_xy_e = t_xy_e0.copy()
    # l_xy_e = l_xy_e0.copy()
    #
    # n_o = len(backend.odometry_measurements)
    # n_l = len(backend.landmark_visibility)
    # Js = []
    #
    # steps_counter = 0
    # while True:
    #     J, b = backend.calc_jacobian_and_b(t_xy_e, l_xy_e)
    #     prog = mp.MathematicalProgram()
    #
    #     dt_xy_e = prog.NewContinuousVariables(n_o + 1, 2, "dt_xy_e")
    #     dl_xy_e = prog.NewContinuousVariables(n_l, 2, "dl_xy_e")
    #     dx = np.hstack((dt_xy_e.ravel(), dl_xy_e.ravel()))
    #
    #     prog.AddL2NormCost(J, -b, dx)
    #     prog.AddQuadraticCost((dx**2).sum() * 0.01)
    #     prog.AddQuadraticCost(((dt_xy_e[0])**2).sum())
    #
    #     result = backend.solver.Solve(prog)
    #     optimal_cost = result.get_optimal_cost()
    #     print("\nStep ", steps_counter)
    #     print(result.get_solution_result())
    #     print("optimal cost: ", result.get_optimal_cost())
    #     print("jacobian norm", np.linalg.norm(J))
    #     print("robot positions\n", t_xy_e)
    #     print("landmark positions\n", l_xy_e)
    #
    #     dt_xy_e_values = result.GetSolution(dt_xy_e)
    #     dl_xy_e_values = result.GetSolution(dl_xy_e)
    #
    #     t_xy_e += dt_xy_e_values
    #     l_xy_e += dl_xy_e_values
    #     Js.append(J)
    #
    #     steps_counter += 1
    #     if optimal_cost < 1e-3:
    #         break
    #
    # print("\n")
    # print("ture robot position: ", t_xy)
    # print("true landmark positions\n",
    #       frontend.l_xy[list(backend.landmark_visibility.keys())])

    # %% save solution
