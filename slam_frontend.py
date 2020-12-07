import numpy as np
import numpy.random as random
import meshcat


def calc_angle_diff(angle1, angle2):
    return (np.pi + angle2 - angle1) % (np.pi * 2) - np.pi


def calc_angle_sum(angle1, angle2):
    assert -np.pi <= angle1 <= np.pi
    assert -np.pi <= angle2 <= np.pi
    sum12 = angle1 + angle2

    if sum12 > np.pi * 2:
        return sum12 - 2 * np.pi
    if sum12 < -np.pi * 2:
        return sum12 + 2 * np.pi
    return sum12


def set_orthographic_camera_xy(vis: meshcat.Visualizer):
    # use orthographic camera, show xy plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-5, right=5, bottom=-5, top=5, near=-1000, far=1000)
    vis['/Cameras/default/rotated'].set_object(camera)
    vis['/Cameras/default/rotated/<object>'].set_property(
        "position", [0, 0, 0])
    vis['/Cameras/default'].set_transform(
        meshcat.transformations.translation_matrix([0, 0, 1]))


class SlamFrontend:
    def __init__(self, num_landmarks: int, seed: int):
        random.seed(seed)

        self.nl = num_landmarks
        # edge length of the square in which the robot runs.
        bbox_length = 1
        self.box_length = bbox_length

        self.r_range_max = 1
        self.r_range_min = 0.2

        # coordinates of landmarks.
        self.l_xy = random.rand(self.nl, 2) * bbox_length - bbox_length / 2
                    # np.array([-1.5, 1.3])
        # self.l_xy = np.array([[0, 0.5], [-0.6, 0.1], [-0.7, -0.2]])

        # visualizer
        self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.vis.delete()
        x_min = -min(self.l_xy[:, 0]) - 1
        x_max = max(self.l_xy[:, 0]) + 1
        y_min = -min(self.l_xy[:, 1]) - 1
        y_max = max(self.l_xy[:, 1]) + 1
        set_orthographic_camera_xy(self.vis)
        self.draw_landmarks()

        # initialize robot
        self.X_WB = meshcat.transformations.rotation_matrix(
            np.pi/2, np.array([1., 0, 0]), np.zeros(3))
        material0 = meshcat.geometry.MeshLambertMaterial(
            color=0xff0000, opacity=1.0)
        material1 = meshcat.geometry.MeshLambertMaterial(
            color=0xff0000, opacity=0.2)
        self.vis["robot"]["sensing"].set_object(
            meshcat.geometry.Cylinder(height=0.005, radius=self.r_range_max),
            material1)
        self.vis["robot"]["body"].set_object(
            meshcat.geometry.Cylinder(height=0.2, radius=0.075),
            material0)
        self.vis["robot"].set_transform(self.X_WB)

        # noise
        self.sigma_odometry = 0.1
        self.sigma_range = 0.05
        self.sigma_bearing = 0.05
        # von-mises distribution.
        self.kappa_bearing = 1 / self.sigma_bearing ** 2

    def draw_box(self, prefix, lengths, color, p):
        self.vis[prefix].set_object(
            meshcat.geometry.Box(lengths),
            meshcat.geometry.MeshLambertMaterial(color=color))
        X = np.eye(4)
        X[:2, 3] = p
        self.vis[prefix].set_transform(X)

    def draw_landmarks(self):
        for i, l_xy_i in enumerate(self.l_xy):
            prefix = "landmarks/{}".format(i)
            self.draw_box(prefix, [0.1, 0.1, 0.1], 0x0000ff, l_xy_i)

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
        X_WB_m = np.zeros(2)

        X_WB_m[:2] = X_WB[:2] - X_WB_previous[:2] + random.normal(
            scale=self.sigma_odometry, size=2)

        return X_WB_m

    def get_landmark_measurements(self, X_WB_xy):
        """
        :param X_WB_xy: robot pose ([x, y]]) in world frame.
        :return:
        (1) idx_visible_l: indices into self.l_xy whose distance to t_xy is
            smaller than self.r_sensor.
        (2) noisy measurements of distances to self.l_xy[idx_visible_l]
        """
        X_WB = meshcat.transformations.rotation_matrix(0, np.array([0, 0, 1.]))
        X_WB[:2, 3] = X_WB_xy
        l_xyz_W = np.zeros((self.nl, 3))
        l_xyz_W[:, :2] = self.l_xy

        X_BW = np.linalg.inv(X_WB)
        l_xyz_B = (X_BW[:3, :3].dot(l_xyz_W.T)).T + X_BW[:3, 3]
        d_l = np.linalg.norm(l_xyz_B, axis=1)
        theta_l_B = np.arctan2(l_xyz_B[:, 1], l_xyz_B[:, 0])

        is_visible = np.all(
            [d_l > self.r_range_min, d_l < self.r_range_max], axis=0)
        idx_visible_l = np.where(is_visible)[0]
        d_l_noisy = d_l[idx_visible_l] + random.normal(
            scale=self.sigma_range, size=idx_visible_l.size)
        bearings_noisy = random.vonmises(
            theta_l_B[idx_visible_l], self.kappa_bearing)

        return idx_visible_l, d_l_noisy, bearings_noisy
