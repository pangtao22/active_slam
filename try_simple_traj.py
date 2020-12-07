import numpy as np
import numpy.random as random

from pydrake.math import inv
from pydrake.autodiffutils import (initializeAutoDiff, autoDiffToValueMatrix,
    autoDiffToGradientMatrix)

from slam_frontend import SlamFrontend, calc_angle_diff
from slam_backend import SlamBackend

#%%
# X_WBs = np.zeros((2, 2))
# X_WBs[1, :2] = X_WBs[0, :2] + [0, 1.7]

X_WBs = np.zeros((10, 2))
X_WBs[1, :2] = X_WBs[0, :2] + [0, 1.]
X_WBs[2, :2] = X_WBs[1, :2] + [-0.1, 0]
X_WBs[3, :2] = X_WBs[2, :2] + [-0.3, 0]
X_WBs[4, :2] = X_WBs[3, :2] + [-0.5, 0.5]
X_WBs[5, :2] = X_WBs[4, :2] + [-1, -0.2]
X_WBs[6, :2] = X_WBs[5, :2] + [-1, -1]
X_WBs[7, :2] = X_WBs[6, :2] + [0, 0.5]
X_WBs[8, :2] = X_WBs[7, :2] + [1.5, 2.5]
X_WBs[9, :2] = X_WBs[8, :2] + [1.5, -2]


frontend = SlamFrontend(num_landmarks=5, seed=16485)
backend = SlamBackend(frontend)

frontend.draw_robot(X_WBs[0])
print(frontend.get_landmark_measurements(X_WBs[0]))
backend.draw_robot_path(X_WBs, color=0x00ff00, prefix="goals",
                        idx_segment=0, size=0.075)

#%%
X_WB_e0 = backend.get_X_WB_initial_guess_array()
l_xy_e0 = backend.get_l_xy_initial_guess_array()

#%% follow prescribed trajectories
for t in range(10):
    idx_visible_l_list, d_l_measured_list, bearings_measured_list = \
        frontend.get_landmark_measurements(X_WBs[t])
    backend.update_landmark_measurements(
        t, idx_visible_l_list, d_l_measured_list, bearings_measured_list)

    if t > 0:
        odometry_measurement = frontend.get_odometry_measurements(
            X_WB=X_WBs[t], X_WB_previous=X_WBs[t - 1])
        backend.update_odometry_measruements(t, odometry_measurement)

    X_WB_e, l_xy_e = backend.run_bundle_adjustment()

    frontend.draw_robot(backend.X_WB_e_dict[t])
    backend.draw_estimated_path_segment(None, 0)
    backend.draw_estimated_landmarks()
    print("robot pose estimated: ", X_WB_e[-1])
    print("robot_pose true: ", X_WBs[t])

    input("next?")


#%%
dX_WB = np.zeros((10, 2))
dX_WB[:5, 0] = 0.5
dX_WB[5:, 0] = -0.5

# dX_WB = np.random.rand(10, 2)
X_WB_p = backend.calc_pose_predictions(dX_WB)

#%% Algorithm 4.
X_WB_e_list = backend.get_X_WB_belief()
l_xy_e_list = backend.get_l_xy_belief()

Omega, q, c = backend.calc_info_matrix(X_WB_e_list, l_xy_e_list)
result = backend.calc_A_lower_half(dX_WB, l_xy_e_list)
result = backend.calc_inner_layer(dX_WB, l_xy_e_list, Omega)

Cov_e = inv(result["I_e"])
Cov_p = inv(result["I_p"])
print("Cov_e", Cov_e.diagonal())
print("Cov_p", Cov_p.diagonal())


#%%
result = backend.calc_objective(dX_WB, X_WB_e_list, l_xy_e_list, X_WB_p[-1],
                                alpha=0.5)

#%% Autodiff
dX_WB_ad = initializeAutoDiff(dX_WB.ravel())
dX_WB_ad.resize(dX_WB.shape)
a = 0
for i in range((len(dX_WB_ad))):
    a += dX_WB_ad[i].sum() * i
a.derivatives().reshape(dX_WB.shape)

#%%
A2_ad = backend.calc_A_lower_half(dX_WB_ad, l_xy_e_list)
result = backend.calc_inner_layer(dX_WB_ad, l_xy_e_list, Omega)

I_e_ad = result["I_e"]
I_p = result["I_p"]
# X_WB_p_ad = result["X_WB_p"]

Cov_e_ad = inv(I_e_ad)
Cov_p_ad = inv(I_p)
print("Cov_e_ad", autoDiffToValueMatrix(Cov_e_ad).diagonal())
print("Cov_p_ad", autoDiffToValueMatrix(Cov_p_ad).diagonal())

#%%
X_WB_g = X_WB_p[5]  # arbitray
c_ad = backend.calc_objective(
    dX_WB_ad, X_WB_e_list, l_xy_e_list, X_WB_g, alpha=0.5)
print(c_ad["c"].derivatives())


#%% numerical diff
backend.calc_objective_gradient(
    dX_WB, X_WB_e_list, l_xy_e_list, X_WB_g, 0.5)

# %% numerical diff
Dc = np.zeros_like(dX_WB.ravel())
c0 = backend.calc_objective(dX_WB, X_WB_e_list, l_xy_e_list, X_WB_p[-1],
                            alpha=0.5)["c"]

h = 1e-3
for i in range(len(Dc)):
    dX_WB_new = dX_WB.copy()
    dX_WB_new.ravel()[i] += h
    c_new = backend.calc_objective(
        dX_WB_new, X_WB_e_list, l_xy_e_list, X_WB_p[-1], alpha=0.5)['c']
    Dc[i] = (c_new - c0) / h


#%%
print(Dc)
print(c_ad.derivatives())


#%%
def calc_odometry_error(X_WB_list, X_I_I1_list):
    n = len(X_WB_list)
    assert len(X_I_I1_list) == n - 1

    error_position = 0.
    for i in range(n - 1):
        dxy = (X_WB_list[i+1][:2] - X_WB_list[i][:2]) - X_I_I1_list[i][:2]
        error_position += (dxy ** 2).sum()

        a = calc_angle_diff(X_WB_list[i][2], X_WB_list[i+1][2])

    return error_position


#%%
X_WB_e_list = backend.get_X_WB_belief()
l_xy_e_list = backend.get_l_xy_belief()

odometry_measurements = backend.get_odometry_measurements()

l_xy_gt = frontend.get_true_landmark_positions(list(backend.l_xy_e_dict.keys()))
#%%
error_position, error_angle = calc_odometry_error(
    X_WB_e_list, odometry_measurements)
print(error_position, error_angle)

error_position, error_angle = calc_odometry_error(
    X_WBs, odometry_measurements)
print(error_position, error_angle)

#%%
J, b = backend.calc_jacobian_and_b(X_WB_e_list, l_xy_e_list)
_, b2 = backend.calc_jacobian_and_b(X_WB_e, l_xy_e_list)
_, b_gt = backend.calc_jacobian_and_b(X_WBs, l_xy_gt)

print(np.linalg.norm(b), np.linalg.norm(b2), np.linalg.norm(b_gt))

#%%
# print(backend.X_WB_e_dict)
# print(frontend.get_true_landmark_positions(list(backend.l_xy_e_dict.keys())))
# print(backend.l_xy_e_dict)
