import numpy as np

from slam_frontend import SlamFrontend
from slam_backend import SlamBackend

#%% Initialization.
# goals
X_WB_goals = np.zeros((10, 2))
X_WB_goals[:5, 1] = np.arange(1, 6)
for i in range(5, 10):
    X_WB_goals[i] = X_WB_goals[i - 1] - 1
# X_WB_goals[5:, 1] = np.arange(4, -1, -1)
#
# X_WB_goals = np.zeros((10, 2))
# X_WB_goals[1, :2] = X_WB_goals[0, :2] + [0, 1.]
# X_WB_goals[2, :2] = X_WB_goals[1, :2] + [-0.1, 0]
# X_WB_goals[3, :2] = X_WB_goals[2, :2] + [-0.3, 0]
# X_WB_goals[4, :2] = X_WB_goals[3, :2] + [-0.5, 0.5]
# X_WB_goals[5, :2] = X_WB_goals[4, :2] + [-1, -0.2]
# X_WB_goals[6, :2] = X_WB_goals[5, :2] + [-1, -1]
# X_WB_goals[7, :2] = X_WB_goals[6, :2] + [0, 0.5]
# X_WB_goals[8, :2] = X_WB_goals[7, :2] + [1.5, 2.5]
# X_WB_goals[9, :2] = X_WB_goals[8, :2] + [1.5, -2]


X_WB = np.zeros(2)
X_WB_gt = [X_WB.copy()]
i = 0

frontend = SlamFrontend(num_landmarks=15, seed=16485)
backend = SlamBackend(frontend)
frontend.draw_robot(X_WB)
backend.draw_robot_path(X_WB_goals, color=0x00ff00, prefix="goals",
                        idx_segment=0, size=0.075)

# Initialize landmark measurements, assuming the robot starts at the origin.
idx_visible_l_list, d_l_measured_list, bearings_measured_list = \
    frontend.get_landmark_measurements(X_WB)
backend.update_landmark_measurements(
    i, idx_visible_l_list, d_l_measured_list, bearings_measured_list)
i += 1

# Run bundle adjustment.
X_WB_e, l_xy_e = backend.run_bundle_adjustment(is_printing=False)

#%% gradient descent in the loop
L = 5
alphas = np.zeros(len(X_WB_goals))
alphas[5:] = 1


for i_goal, X_WB_g in enumerate(X_WB_goals):
    # Plan.
    dX_WB0 = backend.initialize_dX_WB_with_goal(X_WB_e[-1], X_WB_g, L)
    dX_WB, result = backend.run_gradient_descent(
        dX_WB0, X_WB_e, l_xy_e, X_WB_g, alpha=alphas[i_goal])

    # initial trajectory.
    X_WB0 = backend.calc_pose_predictions(dX_WB0)
    X_WB0 = np.vstack([X_WB_e[-1], X_WB0])

    # draw initial trajectory
    backend.draw_robot_path(
        X_WB0, color=0xff00ff, prefix="robot_poses_init",
        idx_segment=i_goal, size=0.075)

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
        i += 1

    backend.draw_estimated_path_segment(L + 1, i_goal, covariance_scale=0.5)
    backend.draw_estimated_landmarks()
    # draw ground truth
    backend.draw_robot_path(
        np.array(X_WB_gt[-(L + 1):]), color=0xffff00, prefix="robot_poses_gt",
        idx_segment=i_goal, size=0.075)

    print("next?")




