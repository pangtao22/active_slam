import numpy as np
import numpy.random as random
import meshcat


def calc_angle_diff(angle1, angle2):
    return (np.pi + angle2 - angle1) % (np.pi * 2) - np.pi


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

        self.r_range_max = bbox_length / 5
        self.r_range_min = max(self.r_range_max/10, 0.1)

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
            meshcat.geometry.Cylinder(height=0.05, radius=self.r_range_max),
            material)
        self.vis["robot"].set_transform(self.X_WB)

        # noise
        self.sigma_odometry = 0.05
        self.sigma_range = 0.05
        self.kappa_bearing = 1 / 0.05**2  # von-mises distribution.

    def draw_landmarks(self):
        l_xyz = np.zeros((self.nl, 3))
        l_xyz[:, :2] = self.l_xy
        colors = np.zeros_like(l_xyz.T)
        colors[2] = 1.
        self.vis["landmarks"].set_object(
            meshcat.geometry.PointCloud(position=l_xyz.T,
                                        color=colors,
                                        size=0.1))

    def get_true_landmark_positions(self, landmark_idx):
        return self.l_xy[landmark_idx]

    def draw_robot(self, X_WB):
        """
        Updates robot pose in mehscat.
        :param t_xy: robot position in world frame.
        :return: None.
        """
        self.X_WB[0:2, 3] = X_WB[:2]
        self.vis["robot"].set_transform(self.X_WB)

    def get_odometry_measurements(self, X_WB, X_WB_previous):
        """

        :param X_WB: [x, y, theta]. Current robot configuration.
        :param X_WB_previous: Previous robot configuration.
        :return:
        """
        X_WB_m = np.zeros(3)

        X_WB_m[:2] = X_WB[:2] - X_WB_previous[:2] + random.normal(
            scale=self.sigma_odometry, size=2)
        X_WB_m[2] = random.vonmises(calc_angle_diff(X_WB_previous[2], X_WB[2]),
                                    self.kappa_bearing)

        return X_WB_m

    def get_landmark_measurements(self, X_WB_xy_theta):
        """
        :param X_WB_xy_theta: robot pose ([x, y, theta]]) in world frame.
        :return:
        (1) idx_visible_l: indices into self.l_xy whose distance to t_xy is
            smaller than self.r_sensor.
        (2) noisy measurements of distances to self.l_xy[idx_visible_l]
        """
        X_WB = meshcat.transformations.rotation_matrix(
            X_WB_xy_theta[2], np.array([0, 0, 1.]))
        X_WB[:2, 3] = X_WB_xy_theta[:2]
        l_xyz_W = np.zeros((self.nl, 3))
        l_xyz_W[:, :2] = self.l_xy

        X_BW = np.linalg.inv(X_WB)
        l_xyz_B = (X_BW[:3, :3].dot(l_xyz_W.T)).T + X_BW[:3, 3]
        d_l = np.linalg.norm(l_xyz_B, axis=1)
        theta_l_B = np.arctan2(l_xyz_B[:, 1], l_xyz_B[:, 0])

        angle_in_range = np.any(
            [theta_l_B < -np.pi * 0.75, theta_l_B > -np.pi * 0.25], axis=0)
        is_visible = np.all(
            [d_l > self.r_range_min, d_l < self.r_range_max, angle_in_range],
            axis=0)
        idx_visible_l = np.where(is_visible)[0]
        d_l_noisy = d_l[idx_visible_l] + random.normal(
            scale=self.sigma_range, size=idx_visible_l.size)
        bearings_noisy = random.vonmises(theta_l_B[idx_visible_l],
                                        self.kappa_bearing)

        return idx_visible_l, d_l_noisy, bearings_noisy
