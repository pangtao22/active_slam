
import numpy as np
import meshcat
from scipy.linalg import sqrtm

from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.gurobi import GurobiSolver
import pydrake.solvers.mathematicalprogram as mp
from pydrake.symbolic import sqrt

from slam_frontend import SlamFrontend, calc_angle_diff


class SlamBackend:
    def __init__(self, X_WB_e_0, frontend: SlamFrontend):
        """
        Indices (consistent with Indelman2015, aka "the paper".)
            Constants:
            k: current time step.
            L: planning horizon.

            i: time step.
            j: landmark.
            l \in [k, k+L]: one of the future time steps up to L.


        :param X_WB_e_0: np array of shape (self.dim_X). Initial estimated
            robot position, used to anchor the entire trajectory.
        """
        self.solver = GurobiSolver()
        self.X_WB_e_0 = X_WB_e_0.copy()

        self.dim_l = 2  # dimension of landmarks.
        self.dim_X = 2  # dimension of poses.

        # current beliefs
        self.X_WB_e_dict = {0: self.X_WB_e_0}
        self.l_xy_e_dict = dict()

        # Information matrix of current robot poses and landmarks, which are
        # ordered as follows:
        # [X_WB_e0, ... X_WB_e_k, l_xy_e0, ... l_xy_eK]
        # The landmarks are ordered as self.l_xy_e_dict.keys().
        self.I_all = None

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

        self.sigma_dynamics = self.sigma_odometry

        self.r_range_max = frontend.r_range_max
        self.r_range_min = frontend.r_range_min

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

    def calc_range_derivatives(self, J, i_row, j_start_x, j_start_l, d_xy,
                               sigma=1.0):
        """

        :param i_row:
        :param j_start_x:
        :param j_start_l:
        :param d_xy: X_WB - l_xy
        :return:
        """
        d = np.sqrt((d_xy ** 2).sum())
        d_xy /= d
        J[i_row, j_start_x: j_start_x + 2] += d_xy / sigma
        J[i_row, j_start_l: j_start_l + self.dim_l] += -d_xy / sigma

        return d

    def calc_bearing_derivatives(self, J, i_row, j_start_x, j_start_l,
                                 X_WB, l_xy, sigma=1.0):
        xb = X_WB[0]
        yb = X_WB[1]
        xl = l_xy[0]
        yl = l_xy[1]
        dx = xl - xb
        dy = yl - yb
        d_arctan_D_dx = 1 / (1 + (dy / dx) ** 2) * (-dy / dx ** 2)
        d_arctan_D_dy = 1 / (1 + (dy / dx) ** 2) * (1 / dx)
        d_arctan_D_dxy = np.array([d_arctan_D_dx, d_arctan_D_dy])

        J[i_row, j_start_x: j_start_x + 2] += -d_arctan_D_dxy / sigma
        J[i_row, j_start_l: j_start_l + self.dim_l] += d_arctan_D_dxy / sigma

        return dx, dy

    @staticmethod
    def calc_num_landmark_measurements(landmark_measurements):
        nl_measurements = 0  # number of landmark measurements
        for visible_landmarks_i in landmark_measurements.values():
            nl_measurements += len(visible_landmarks_i)
        return nl_measurements

    def calc_jacobian_and_b(self, X_WB_e, l_xy_e):
        """
        :param X_WB_e (n_o + 1, 3)
        :param l_xy_e (nl, 2): coordinates of landmarks, ordered the same way as
            self.landmark_measurements.
        :param landmark_measurements:
        :return:
        """
        dim_l = self.dim_l  # dimension of landmarks.
        dim_X = self.dim_X  # dimension of poses.

        nl = len(self.landmark_measurements)  # number of landmarks
        n_o = len(self.odometry_measurements)  # number of odometries

        # number of landmark measurements
        nl_measurements = self.calc_num_landmark_measurements(
            self.landmark_measurements)

        n_rows = n_o * dim_X + nl_measurements * dim_l + self.dim_X
        n_cols = (n_o + 1) * dim_X + nl * 2
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
        for idx_j, (j, visible_robot_poses) in enumerate(
                self.landmark_measurements.items()):
            for i in visible_robot_poses.keys():
                # i: index of robot poses visible from
                d_ik_m = visible_robot_poses[i]["distance"]
                b_ik_m = visible_robot_poses[i]["bearing"]

                # range.
                d_xy = X_WB_e[i, :2] - l_xy_e[idx_j]

                j_start_x = i * dim_X
                j_start_l = (n_o + 1) * dim_X + idx_j * dim_l
                d = self.calc_range_derivatives(
                    J, i_row, j_start_x, j_start_l, d_xy)

                b[i_row] = d - d_ik_m
                sigmas[i_row] = self.sigma_range

                # bearing.
                i_row += 1
                dx, dy = self.calc_bearing_derivatives(
                    J, i_row, j_start_x, j_start_l,
                    X_WB_e[i], l_xy_e[idx_j])

                b[i_row] = calc_angle_diff(b_ik_m, np.arctan2(dy, dx))
                sigmas[i_row] = self.sigma_bearing

                i_row += 1

        return J, b, sigmas

    def calc_info_matrix(self, X_WB_e, l_xy_e):
        J, b, sigmas = self.calc_jacobian_and_b(X_WB_e, l_xy_e)
        I = 1 / sigmas ** 2
        Omega = (J.T * I).dot(J)
        q = J.T.dot(I * b)
        c = (b * I).dot(b)

        return Omega, q, c

    def calc_A_lower_half(self, dX_WB, l_xy_e):
        """
        Algorithm 4 in the paper.
        :param dX_WB: [l, self.dim_X]. Input for time steps [l : ].
        :return:
        """
        # X_WB_e: belief. e stands for "estimated".
        # X_WB_p: prediction. corresponds to quantities with an overbar.
        n_p = len(self.X_WB_e_dict)
        l = len(dX_WB)

        # future predictions.
        X_WB_p = self.calc_pose_predictions(dX_WB)

        # Find the visible landmarks for every X_WB_p[i].
        # Store in {i: [landmark order indices.]}, i.e. the same order as
        #   self.landmark_measurements.
        future_landmark_measurements = self.find_visible_landmarks(X_WB_p)
        nl = len(self.landmark_measurements)  # number of landmarks
        # number of new landmark measurements
        nl_measurements = self.calc_num_landmark_measurements(
            future_landmark_measurements)

        # Structure of new "state"
        # [X_WB_0, ... X_WB_{k},
        #  l_xy (all landmarks),
        #  X_WB_{k+1}, ... X_WB_{k+l}]
        n_Xk = n_p * self.dim_X + nl * self.dim_l
        n_rows = l * self.dim_X + 2 * nl_measurements
        n_cols = l * self.dim_X + n_Xk

        # A2 is the lower half of \mathcal{A} in Eq. (32), i.e. [F; H].
        A2 = np.zeros((n_rows, n_cols))

        # dynamics
        for i in range(l):
            i0 = i * self.dim_X
            i1 = i0 + self.dim_X

            A2[i0: i1, n_Xk + i0: n_Xk + i1] = \
                np.eye(self.dim_X) / self.sigma_dynamics

            if i == 0:
                A2[i0: i1, (n_p - 1) * self.dim_X: n_p * self.dim_X] = \
                    -np.eye(self.dim_X) / self.sigma_dynamics
            else:
                A2[i0: i1, n_Xk + i0 - self.dim_X: n_Xk + i1 - self.dim_X] = \
                    -np.eye(self.dim_X) / self.sigma_dynamics

        # observations
        i_row = l * self.dim_X
        for idx_j, (j, visible_robot_poses) in enumerate(
                future_landmark_measurements.items()):
            for i in visible_robot_poses.keys():
                d_xy = X_WB_p[i] - l_xy_e[idx_j]
                j_start_x = n_Xk + i * self.dim_X
                j_start_l = n_p * self.dim_X + idx_j * self.dim_l

                self.calc_range_derivatives(
                    A2, i_row, j_start_x, j_start_l, d_xy,
                    sigma=self.sigma_range)

                self.calc_bearing_derivatives(
                    A2, i_row + 1, j_start_x, j_start_l,
                    X_WB_p[i], l_xy_e[idx_j], sigma=self.sigma_bearing)

                i_row += 2

        return A2, X_WB_p

    def calc_inner_layer(self, dX_WB, l_xy_e, I_Xk):
        """

        :param dX_WB:
        :param X_WB_e:
        :param l_xy_e:
        :param I_Xk: information matrix of current trajetory and landmarks.
        :return:
        """
        n_Xk = I_Xk.shape[0]  # size of states up to now.
        l = len(dX_WB)

        def calc_I(A2):
            n_Xl = A2.shape[1] - n_Xk  # size of states into the future.
            n_X = n_Xk + n_Xl

            A21 = A2[:, :n_Xk]
            A22 = A2[:, n_Xk:]

            # I_X{k+l} = A.T.dot(A)
            I = np.zeros((n_X, n_X), dtype=dX_WB.dtype)
            I[:n_Xk, :n_Xk] = I_Xk + A21.T.dot(A21)
            I[:n_Xk, n_Xk:] = A21.T.dot(A22)
            I[n_Xk:, :n_Xk] = I[:n_Xk, n_Xk:].T
            I[n_Xk:, n_Xk:] = A22.T.dot(A22)

            return I

        # lower half of A
        A2_full, X_WB_p = self.calc_A_lower_half(dX_WB, l_xy_e)
        I_e = calc_I(A2_full)  # including landmarks.
        I_p = calc_I(A2_full[:l*self.dim_X])

        return I_e, I_p, X_WB_p
        # A1 = np.hstack([sqrtm(I_Xk), np.zeros((n_Xk, n_Xl))])
        # A = np.vstack([A1, A2])
        # return A.T.dot(A)

    def calc_pose_predictions(self, dX_WB):
        l = len(dX_WB)
        X_WB_p = np.zeros((l, self.dim_X))  # k + 1 : (k + l)
        k = len(self.X_WB_e_dict) - 1   # current time step.
        X_WB_e_k = self.X_WB_e_dict[k]

        for i in range(l):
            if i == 0:
                X_WB_p[i] = X_WB_e_k + dX_WB[i]
            else:
                X_WB_p[i] = X_WB_p[i - 1] + dX_WB[i]

        return X_WB_p

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
            Omega, q, c = self.calc_info_matrix(X_WB_e, l_xy_e)
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

        self.update_beliefs(X_WB_e, l_xy_e, Omega)

        print("\nStep ", n_o)
        print("optimal cost: ", optimal_cost)
        _, b, _ = self.calc_jacobian_and_b(X_WB_e, l_xy_e)
        print("b norm recalculated: ", np.linalg.norm(b))
        print("total gradient steps: ", steps_counter)
        print("dx norm: ", np.linalg.norm(dx_values))
        Omega_pose = self.marginalize_info_matrix(Omega, n_p, n_l)
        print("Marginalized covariance diagonal\n",
              np.linalg.inv(Omega_pose).diagonal())
        print("Covariance diagonal\n",
              np.linalg.inv(Omega).diagonal()[:self.dim_X * n_p])
        print("\n")

        return X_WB_e, l_xy_e

    def update_beliefs(self, X_WB_e_values, l_xy_e_values, I_all):
        for i, X_WB_e in enumerate(X_WB_e_values):
            self.X_WB_e_dict[i] = X_WB_e

        self.l_xy_e_dict.clear()
        for k, l_xy_e_value in zip(self.landmark_measurements.keys(),
                                   l_xy_e_values):
            self.l_xy_e_dict[k] = l_xy_e_value

        self.I_all = I_all

    def find_visible_landmarks(self, X_WB_p):
        future_landmark_measurements = dict()
        for i, X_WB in enumerate(X_WB_p):
            for j in self.l_xy_e_dict.keys():
                d_xy = self.l_xy_e_dict[j] - X_WB
                d = np.linalg.norm(d_xy)
                if self.r_range_min < d < self.r_range_max:
                    if j not in future_landmark_measurements:
                        future_landmark_measurements[j] = dict()
                    b = np.arctan2(d_xy[1], d_xy[0])
                    future_landmark_measurements[j][i] = {
                        "distance": d, "bearing": b}

        return future_landmark_measurements

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