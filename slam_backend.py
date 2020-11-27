
import numpy as np
import meshcat

from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.gurobi import GurobiSolver
import pydrake.solvers.mathematicalprogram as mp
from pydrake.symbolic import sqrt

from slam_frontend import SlamFrontend, calc_angle_diff


class SlamBackend:
    def __init__(self, X_WB_e_0, frontend: SlamFrontend):
        """

        :param X_WB_e_0: initial estimated robot position,
            used to anchor the entire trajectory.
        """
        self.solver = GurobiSolver()
        self.X_WB_e_0 = X_WB_e_0.copy()

        # current beliefs
        self.X_WB_e_dict = {0: self.X_WB_e_0}
        self.l_xy_e_dict = dict()

        # past measurements
        self.odometry_measurements = dict()

        # key (int): landmark index
        # value: dict of {t: {"distance":d, "bearing": b}}.
        # t: index of robot poses from which the landmark is visible.
        # d: range measurement for the landmark from the pose indexed by t.
        # b: bearing measurement for the landmark from the pose indexed by t.
        self.landmark_measurements = dict()

        # Taking stuff from frontend
        self.vis = frontend.vis
        self.l_xy = frontend.l_xy
        #TODO: load from config file?
        self.sigma_odometry = frontend.sigma_odometry
        self.sigma_range = frontend.sigma_range
        self.sigma_bearing = frontend.sigma_bearing

        self.dim_l = 2  # dimension of landmarks.
        self.dim_X = 2  # dimension of poses.

    def update_landmark_measurements(
            self, t, idx_visible_l_list, d_l_measured_list, b_l_measured_list):
        """
        inputs are the output of Frontend.get_range_measurements()
        :return: None.
        """
        for i, i_l in enumerate(idx_visible_l_list):
            d_l = d_l_measured_list[i]
            b_l = b_l_measured_list[i]
            if i_l not in self.landmark_measurements:
                self.landmark_measurements[i_l] = dict()
            self.landmark_measurements[i_l][t] = \
                {"distance": d_l, "bearing": b_l}

    def update_odometry_measruements(self, t, odometry_measurement):
        """

        :param t: current time step (starting from 0).
        :param odometry_measurement: odometry measruement between t and t-1.
        :return:
        """
        self.odometry_measurements[t] = odometry_measurement

    def get_X_WB_belief(self):
        n = len(self.X_WB_e_dict)
        X_WB_e = np.zeros((n, self.dim_X))

        for i in range(n):
            X_WB_e[i] = self.X_WB_e_dict[i]

        return X_WB_e

    def get_l_xy_belief(self):
        n = len(self.l_xy_e_dict)
        l_xy_e = np.zeros((n, self.dim_l))

        for i, k in enumerate(self.l_xy_e_dict.keys()):
            l_xy_e[i] = self.l_xy_e_dict[k]

        return l_xy_e

    def get_odometry_measurements(self):
        n_o = len(self.odometry_measurements)
        odometry_measurements = np.zeros((n_o, self.dim_X))

        for i in range(n_o):
            odometry_measurements[i] = self.odometry_measurements[i + 1]

        return odometry_measurements

    def get_X_WB_initial_guess_array(self):
        """
        Use self.t_xy_e_dict to create an initial guess for the next bundle
            adjustment.
        Let t be the current exploration time step. As t_xy_e_dict is the
            estimate for time step t-1, its shape is (t - 1, 3), which does
            not include an estimate for the pose of the current time step.

        The initial guess for the current time step is computed as
            self.t_xy_e_dict[t - 1] + self.odometry_measurements[t].
        :return:
        """
        n_o = len(self.odometry_measurements)
        X_WB_e = np.zeros((n_o + 1, self.dim_X))
        for i in range(n_o):
            X_WB_e[i] = self.X_WB_e_dict[i]
        if n_o > 0:
            X_WB_e[n_o] = X_WB_e[n_o - 1] + self.odometry_measurements[n_o]

        # First ground truth is given.
        X_WB_e[0] = self.X_WB_e_0

        return X_WB_e

    def get_l_xy_initial_guess_array(self):
        nl = len(self.landmark_measurements)
        l_xy_e = np.ones((nl, 2))

        for i_k, k in enumerate(self.landmark_measurements.keys()):
            if k in self.l_xy_e_dict:
                l_xy_e[i_k] = self.l_xy_e_dict[k]
        return l_xy_e

    def marginalize_info_matrix(self, Omega, n_p, n_l):
        nX = n_p * self.dim_X
        m, n = Omega.shape
        assert m == n
        assert m == nX + n_l * self.dim_l

        A = Omega[:nX, :nX]
        B = Omega[:nX, nX:]
        C = Omega[nX:, nX:]

        return A - B.dot(np.linalg.inv(C)).dot(B.T)

    def calc_jacobian_and_b(self, X_WB_e, l_xy_e):
        """
        :param X_WB_e (n_o + 1, 3)
        :param l_xy_e (nl, 2): coordinates of landmarks, ordered the same way as
            self.landmark_measurements.
        :return:
        """
        dim_l = self.dim_l  # dimension of landmarks.
        dim_X = self.dim_X  # dimension of poses.

        nl = len(self.landmark_measurements)  # number of landmarks
        n_o = len(self.odometry_measurements)  # number of odometries
        nl_measurements = 0  # number of landmark measurements
        for visible_landmarks_i in self.landmark_measurements.values():
            nl_measurements += len(visible_landmarks_i)

        n_rows = n_o * dim_X + nl_measurements * dim_l + self.dim_X
        n_cols = (n_o + 1) * dim_X + nl * dim_l
        J = np.zeros((n_rows, n_cols))
        b = np.zeros(n_rows)
        sigmas = np.zeros(n_rows)

        # prior on first pose.
        J[-self.dim_X:, :self.dim_X] = np.eye(self.dim_X)
        sigmas[-self.dim_X:] = (self.sigma_odometry / 10)

        # odometry
        for i in range(n_o):
            i0 = i * dim_X
            i1 = i0 + dim_X

            J[i0: i1, i0: i1] = -np.eye(dim_X)
            J[i0: i1, i0 + dim_X: i1 + dim_X] = np.eye(dim_X)
            bi = b[i0: i1]
            sigmas_i = sigmas[i0: i1]
            # displacement
            bi[:2] = X_WB_e[i + 1, :2] - X_WB_e[i, :2] - \
                self.odometry_measurements[i + 1][:2]
            sigmas_i[:2] = self.sigma_odometry

        # range and bearing measurements.
        i_row = n_o * dim_X
        for i_k, (k, visible_robot_poses) in enumerate(
                self.landmark_measurements.items()):
            for i in visible_robot_poses.keys():
                # i: index of robot poses visible from
                d_ik_m = visible_robot_poses[i]["distance"]
                b_ik_m = visible_robot_poses[i]["bearing"]

                # range.
                d_xy = X_WB_e[i, :2] - l_xy_e[i_k]
                d = np.sqrt((d_xy**2).sum())
                d_xy /= d

                j_start = i * dim_X
                J[i_row, j_start: j_start + 2] += d_xy

                j0 = (n_o + 1) * dim_X + i_k * dim_l
                j1 = j0 + dim_l
                J[i_row, j0: j1] += -d_xy

                b[i_row] = d - d_ik_m
                sigmas[i_row] = self.sigma_range

                # bearing.
                i_row += 1
                xb = X_WB_e[i, 0]
                yb = X_WB_e[i, 1]
                xl = l_xy_e[i_k][0]
                yl = l_xy_e[i_k][1]
                dx = xl - xb
                dy = yl - yb
                d_arctan_D_dx = 1 / (1 + (dy/dx)**2) * (-dy/dx**2)
                d_arctan_D_dy = 1 / (1 + (dy/dx)**2) * (1/dx)
                d_arctan_D_dxy = np.array([d_arctan_D_dx, d_arctan_D_dy])

                J[i_row, j_start: j_start + 2] += -d_arctan_D_dxy
                J[i_row, j_start + 2] += -1
                J[i_row, j0: j1] += d_arctan_D_dxy

                b[i_row] = calc_angle_diff(b_ik_m, np.arctan2(dy, dx))
                sigmas[i_row] = self.sigma_bearing

                i_row += 1

        return J, b, sigmas

    def run_bundle_adjustment(self):
        dim_l = self.dim_l  # dimension of landmarks.
        dim_X = self.dim_X  # dimension of poses.

        X_WB_e0 = self.get_X_WB_initial_guess_array()
        l_xy_e0 = self.get_l_xy_initial_guess_array()
        X_WB_e = X_WB_e0.copy()
        l_xy_e = l_xy_e0.copy()

        n_o = len(self.odometry_measurements)
        n_p = n_o + 1  # number of poses.
        n_l = len(self.landmark_measurements)

        steps_counter = 0
        while True:
            J, b, sigmas = self.calc_jacobian_and_b(X_WB_e, l_xy_e)
            I = 1 / sigmas ** 2
            # I = np.eye(len(b))
            # I[-self.dim_X:] = 1000
            Omega = (J.T * I).dot(J)
            q = J.T.dot(I * b)
            c = (b * I).dot(b)
            prog = mp.MathematicalProgram()

            dX_WB_e = prog.NewContinuousVariables(n_p, dim_X, "dX_WB_e")
            dl_xy_e = prog.NewContinuousVariables(n_l, dim_l, "dl_xy_e")
            dx = np.hstack((dX_WB_e.ravel(), dl_xy_e.ravel()))

            nx = len(dx)
            prog.AddQuadraticCost(Omega, q, 0.5*c, dx)
            # prog.AddQuadraticCost(((dx / self.sigma_odometry)**2).sum())
            prog.AddQuadraticCost(np.eye(nx) * 2, np.zeros(nx), dx)
            # prog.AddQuadraticCost(10000 * ((dX_WB_e[0]) ** 2).sum())
            # prog.AddL2NormCost(J, -b, dx)

            result = self.solver.Solve(prog)
            optimal_cost = result.get_optimal_cost()
            assert result.get_solution_result() == \
                   mp.SolutionResult.kSolutionFound

            dX_WB_e_values = result.GetSolution(dX_WB_e)
            dl_xy_e_values = result.GetSolution(dl_xy_e)
            dx_values = result.GetSolution(dx)

            X_WB_e += dX_WB_e_values
            l_xy_e += dl_xy_e_values

            steps_counter += 1
            if steps_counter > 500 or \
                    np.linalg.norm(dx_values) / (n_p + n_l) < 5e-3:
                break

        self.update_beliefs(X_WB_e, l_xy_e)

        print("\nStep ", n_o)
        print("optimal cost: ", optimal_cost)
        print("b norm: ", np.linalg.norm(b))
        _, b, _ = self.calc_jacobian_and_b(X_WB_e, l_xy_e)
        print("b norm recalculated: ", np.linalg.norm(b))
        print("total gradient steps: ", steps_counter)
        print("gradient norm: ", np.linalg.norm(J.T.dot(b)))
        print("dx norm: ", np.linalg.norm(dx_values))
        Omega_pose = self.marginalize_info_matrix(Omega, n_p, n_l)
        print("Covariance diagonal\n",
              np.linalg.inv(Omega_pose).diagonal())
        print("\n")

        return X_WB_e, l_xy_e

    def update_beliefs(self, X_WB_e_values, l_xy_e_values):
        for i, X_WB_e in enumerate(X_WB_e_values):
            self.X_WB_e_dict[i] = X_WB_e

        self.l_xy_e_dict.clear()
        for k, l_xy_e_value in zip(self.landmark_measurements.keys(),
                                   l_xy_e_values):
            self.l_xy_e_dict[k] = l_xy_e_value

    def draw_estimated_path(self):
        nt = len(self.X_WB_e_dict)
        t_xy_e = np.zeros((nt, 2))
        for i in range(nt):
            t_xy_e[i] = self.X_WB_e_dict[i][:2]
        self.draw_robot_path(t_xy_e, color=[1, 0, 0], prefix="robot_poses_e")

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

    def draw_robot_path(self, X_WBs, color, prefix: str):
        t_xy = X_WBs[:, :2]
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