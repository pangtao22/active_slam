import numpy as np
import numpy.random as random
import meshcat

from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
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

        self.r_sensor = bbox_length / 10

        # coordinates of landmarks.
        self.l_xy = random.rand(self.nl, 2) * bbox_length - bbox_length / 2

        # visualizer
        self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.draw_landmarks()

        # initialize robot
        self.X_WB = meshcat.transformations.rotation_matrix(
            np.pi/2, np.array([1., 0, 0]), np.zeros(3))
        material = meshcat.geometry.MeshLambertMaterial(
            color=0xfcfcfc, opacity=0.7)
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
        self.solver = SnoptSolver()
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

    def run_bundle_adjustment(self):
        t = len(self.odometry_measurements)

        prog = mp.MathematicalProgram()

        # prior for first pose.
        t_xy_e = prog.NewContinuousVariables(t + 1, 2, "t_xy_e")
        prog.AddQuadraticCost(
            ((t_xy_e[0] - self.t_xy_e_dict[0]) ** 2).sum())

        # landmarks
        n_l = len(self.landmark_visibility)
        l_xy_e = prog.NewContinuousVariables(n_l, 2, "l_xy_e")
        s_l = prog.NewContinuousVariables(n_l, "s_l")  # slack variables.

        indices_landmark = []
        for i_k, (k, t_dict) in enumerate(self.landmark_visibility.items()):
            # print(i_k, k, pose_list)
            indices_landmark.append(k)
            for i_t, d_l_measured in t_dict.items():
                e2 = d_l_measured ** 2 - (
                        (t_xy_e[i_t] - l_xy_e[i_k]) ** 2).sum()
                prog.AddConstraint(s_l[i_k] >= e2)
                prog.AddConstraint(s_l[i_k] >= -e2)
                prog.AddCost(s_l[i_k])

        # odometry
        for i, odometry_measurement in self.odometry_measurements.items():
            dt = t_xy_e[i] - t_xy_e[i-1] - odometry_measurement
            prog.AddCost((dt**2).sum())

        # initial guess
        for i in range(len(self.t_xy_e_dict)):
            prog.SetInitialGuess(t_xy_e[i], self.t_xy_e_dict[i])

        for i_k, k in enumerate(indices_landmark):
            if k in self.l_xy_e_dict:
                prog.SetInitialGuess(l_xy_e[i_k], self.l_xy_e_dict[k])

        # %% save solution
        result = self.solver.Solve(prog)
        assert result.get_solution_result() == mp.SolutionResult.kSolutionFound

        t_xy_e_values = result.GetSolution(t_xy_e)
        for i, t_xy_e_value in enumerate(t_xy_e_values):
            if i > 0:
                self.t_xy_e_dict[i] = t_xy_e_value

        l_xy_e_values = result.GetSolution(l_xy_e)
        self.l_xy_e_dict.clear()
        for k, l_xy_e_value in zip(self.landmark_visibility.keys(),
                                   l_xy_e_values):
            self.l_xy_e_dict[k] = l_xy_e_value

        print("Step: ", t)
        print("max error for landmarks:", np.max(result.GetSolution(s_l)))
        print("robot pose estimates\n", t_xy_e_values)
        print("landmark estimates\n", l_xy_e_values)




if __name__ == "__main__":
    #%%
    t_xy = np.zeros((3, 2))
    t_xy[0] = np.array([1.5, -1.3])
    t_xy[1] = t_xy[0] + np.array([0, 1.])
    t_xy[2] = t_xy[1] + np.array([-0.1, 0])

    frontend = SlamFrontend(num_landmarks=50, seed=16485)
    backend = SlamBackend(t_xy[0], frontend)

    frontend.draw_robot(t_xy[0])
    print(frontend.get_range_measurements(t_xy[0]))


#%% first measurement
    t = 0
    idx_visible_l_list, d_l_measured_list = \
        frontend.get_range_measurements(t_xy[t])
    backend.update_landmark_measurements(
        t, idx_visible_l_list, d_l_measured_list)

#%%
    backend.run_bundle_adjustment()
    print("ture robot position: ", t_xy[t])
    print("true landmark positions\n",
          frontend.l_xy[list(backend.landmark_visibility.keys())])

#%% second measurement
    t = 1
    odometry_measurement = frontend.get_odometry_measurements(
        t_xy=t_xy[t], t_xy_previous=t_xy[t-1])
    backend.update_odometry_measruements(t, odometry_measurement)

    idx_visible_l_list, d_l_measured_list = \
        frontend.get_range_measurements(t_xy[t])
    backend.update_landmark_measurements(
        t, idx_visible_l_list, d_l_measured_list)

#%%
    backend.run_bundle_adjustment()
    print("ture robot position: ", t_xy[t])
    print("true landmark positions\n",
          frontend.l_xy[list(backend.landmark_visibility.keys())])


#%% third measurement
    t = 2
    odometry_measurement = frontend.get_odometry_measurements(
        t_xy=t_xy[t], t_xy_previous=t_xy[t-1])
    backend.update_odometry_measruements(t, odometry_measurement)

    idx_visible_l_list, d_l_measured_list = \
        frontend.get_range_measurements(t_xy[t])
    backend.update_landmark_measurements(
        t, idx_visible_l_list, d_l_measured_list)

#%%
    backend.run_bundle_adjustment()
    print("ture robot position: ", t_xy[t])
    print("true landmark positions\n",
          frontend.l_xy[list(backend.landmark_visibility.keys())])


