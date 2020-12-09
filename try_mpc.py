import time

import numpy as np
from matplotlib import cm
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt

from slam_frontend import SlamFrontend
from slam_backend import SlamBackend

#%% Initialization.
# goals
X_WB_goals = np.zeros((9, 2))
X_WB_goals[:5, 1] = np.arange(1, 6)
for i in range(5, 9):
    X_WB_goals[i] = X_WB_goals[i - 1] - 1

X_WB = np.zeros(2)
X_WB_gt = [X_WB.copy()]
i = 0

frontend = SlamFrontend(num_landmarks=15, seed=16485)
backend = SlamBackend(frontend)
frontend.draw_robot(X_WB)
backend.draw_robot_path(X_WB_goals, color=0xff00ff, prefix="goals",
                        idx_segment=0, size=0.1)

# Initialize landmark measurements, assuming the robot starts at the origin.
idx_visible_l_list, d_l_measured_list, bearings_measured_list = \
    frontend.get_landmark_measurements(X_WB)
backend.update_landmark_measurements(
    i, idx_visible_l_list, d_l_measured_list, bearings_measured_list)
i += 1

# Run bundle adjustment.
X_WB_e, l_xy_e = backend.run_bundle_adjustment(is_printing=False)

input("continue?")
#%% gradient descent in the loop
start_time = time.time()

L = 5
alphas = np.zeros(len(X_WB_goals))
alphas[5:] = 1

# logging
alpha_log = []
sqrt_position_cov_trace_log = []

for i_goal, X_WB_g in enumerate(X_WB_goals):
    # Plan.
    dX_WB0 = backend.initialize_dX_WB_with_goal(
        X_WB_e[-1], X_WB_g, L, max_step=0.3)
    dX_WB, result = backend.run_gradient_descent(
        dX_WB0, X_WB_e, l_xy_e, X_WB_g, backprop=True)

    # initial trajectory.
    # X_WB0 = backend.calc_pose_predictions(dX_WB0)
    # X_WB0 = np.vstack([X_WB_e[-1], X_WB0])

    # draw initial trajectory
    # backend.draw_robot_path(
    #     X_WB0, color=0xff00ff, prefix="robot_poses_init",
    #     idx_segment=i_goal, size=0.075)

    print("\nGoal No. {}\n".format(i_goal), result)

    # Move and collect measurements.
    for dX_WB_i in dX_WB:
        X_WB += dX_WB_i

        # measure landmarks.
        idx_visible_l_list, d_l_measured_list, bearings_measured_list = \
            frontend.get_landmark_measurements(X_WB)
        backend.update_landmark_measurements(
            i, idx_visible_l_list, d_l_measured_list, bearings_measured_list)

        # odometry
        odometry_measurement = frontend.get_odometry_measurements(
            X_WB=X_WB, X_WB_previous=X_WB_gt[-1])
        backend.update_odometry_measruements(i, odometry_measurement)

        X_WB_e, l_xy_e = backend.run_bundle_adjustment(is_printing=False)

        frontend.draw_robot(X_WB)

        # updates.
        X_WB_gt.append(X_WB.copy())

        alpha = backend.calc_alpha(
            np.zeros_like(dX_WB), X_WB_e, l_xy_e)
        backend.draw_estimated_path_segment(None, 0, covariance_scale=1.0)
        backend.draw_estimated_landmarks()
        # draw ground truth
        backend.draw_robot_path(
            np.array(X_WB_gt[-2:]),
            color=int(rgb2hex(cm.jet(alpha))[1:], 16),
            prefix="robot_poses_gt",
            idx_segment=i, size=0.075)

        i += 1

        # logging
        alpha_log.append(alpha)
        sqrt_position_cov_trace_log.append(
            backend.calc_sqrt_postion_cov_trace())

    # termination
    reached_goal = np.linalg.norm(X_WB_e[-1] - X_WB_goals[-1]) < 0.5
    reached_goal = reached_goal and result["alpha"] < 0.5
    if i > 100 or reached_goal:
        break

    input("next?")

print("--- %s seconds ---" % (time.time() - start_time))


#%%
plt.plot(alpha_log)
plt.show()

#%%
plt.plot(sqrt_position_cov_trace_log)
plt.show()

