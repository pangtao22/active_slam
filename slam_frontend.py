import numpy as np
import numpy.random as random
import meshcat


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
