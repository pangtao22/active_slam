import time

import numpy as np
import numpy.random as random

from slam_frontend import SlamFrontend
from slam_backend import SlamBackend


X_WBs = np.zeros((9, 3))
X_WBs[0, :2] = np.array([1.5, -1.3])
X_WBs[1, :2] = X_WBs[0, :2] + [0, 1.]
X_WBs[2, :2] = X_WBs[1, :2] + [-0.1, 0]
X_WBs[3, :2] = X_WBs[2, :2] + [-0.3, 0]
X_WBs[4, :2] = X_WBs[3, :2] + [-0.5, 0.5]
X_WBs[5, :2] = X_WBs[4, :2] + [-1, -0.2]
X_WBs[6, :2] = X_WBs[5, :2] + [-1, -1]
X_WBs[7, :2] = X_WBs[6, :2] + [0, 0.5]
X_WBs[8, :2] = X_WBs[7, :2] + [1.5, 2.5]
X_WBs[:, 2] = np.random.rand(9) * 2 * np.pi - np.pi

frontend = SlamFrontend(num_landmarks=50, seed=16485)
backend = SlamBackend(X_WBs[0], frontend)

frontend.draw_robot(X_WBs[0])
print(frontend.get_landmark_measurements(X_WBs[0]))
backend.draw_robot_path(X_WBs, color=[0, 1, 0], prefix="robot_gt")
#%% first measurement
for t in range(len(X_WBs)):
    idx_visible_l_list, d_l_measured_list = \
        frontend.get_landmark_measurements(X_WBs[t])
    backend.update_landmark_measurements(
        t, idx_visible_l_list, d_l_measured_list)

    if t > 0:
        odometry_measurement = frontend.get_odometry_measurements(
            X_WB=X_WBs[t], X_WB_previous=X_WBs[t - 1])
        backend.update_odometry_measruements(t, odometry_measurement)

    try:
        backend.run_bundle_adjustment()
    except AssertionError:
        break

    frontend.draw_robot(backend.X_WB_e_dict[t])
    backend.draw_estimated_path()
    backend.draw_estimated_landmarks()

    input("next?")

#%%
# backend.draw_estimated_robots()
# backend.draw_robots(t_xy, color=[0, 1, 0], prefix="robot_gt")

