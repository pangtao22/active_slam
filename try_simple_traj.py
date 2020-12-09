import cProfile
import timeit

import numpy as np

from pydrake.math import inv as inv_pydrake
from pydrake.autodiffutils import (initializeAutoDiff, autoDiffToValueMatrix,
    autoDiffToGradientMatrix)

import torch
from slam_frontend import SlamFrontend, calc_angle_diff
from slam_backend import SlamBackend

#%%
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

frontend = SlamFrontend(num_landmarks=50, seed=16485,
                        bbox_length=5, landmarks_offset=[-1.5, 1.3])

backend = SlamBackend(frontend)

frontend.draw_robot(X_WBs[0])
print(frontend.get_landmark_measurements(X_WBs[0]))
backend.draw_robot_path(X_WBs, color=0xff00ff, prefix="goals",
                        idx_segment=0, size=0.075)

input("next?")

#%%
X_WB_e0 = backend.get_X_WB_initial_guess_array()
l_xy_e0 = backend.get_l_xy_initial_guess_array()

#%% follow prescribed trajectories
for t in range(2):
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
    backend.draw_estimated_path_segment(None, 0, covariance_scale=2)
    backend.draw_estimated_landmarks()
    print("robot pose estimated: ", X_WB_e[-1])
    print("robot_pose true: ", X_WBs[t])

    input("next?")


#%%
dX_WB_np = np.zeros((10, 2))
dX_WB_np[:5, 0] = 0.5
dX_WB_np[5:, 0] = -0.5

dX_WB_torch = torch.tensor(dX_WB_np, requires_grad=True)
X_WB_g = X_WBs[0]

# dX_WB = np.random.rand(10, 2)
# X_WB_p = backend.calc_pose_predictions(dX_WB)

X_WB_e_list = backend.get_X_WB_belief()
l_xy_e_list = backend.get_l_xy_belief()
Omega, q, c = backend.calc_info_matrix(X_WB_e_list, l_xy_e_list)

#%%
# Omega, q, c = backend.calc_info_matrix(X_WB_e_list, l_xy_e_list)
result_np = backend.calc_A_lower_half(dX_WB_np, l_xy_e_list)
result_torch = backend.calc_A_lower_half(dX_WB_torch, l_xy_e_list)
(result_torch['H']**2).sum().backward()
print(dX_WB_torch.grad)

#%%
if torch.is_tensor(dX_WB_torch):
    inv = torch.inverse
else:
    inv = inv_pydrake

Cov_e = inv(result_torch["I_e"])
Cov_p = inv(result_torch["I_p"])
print("Cov_e", Cov_e.diagonal())
print("Cov_p", Cov_p.diagonal())


#%% pytorch
result_torch = backend.calc_objective(
    dX_WB_torch, X_WB_e_list, l_xy_e_list, X_WB_g, alpha=0.5)
result_torch['c'].backward()
print("pytorch derivatives\n", dX_WB_torch.grad)

#%% Autodiff
dX_WB_ad = initializeAutoDiff(dX_WB_np.ravel())
dX_WB_ad.resize(dX_WB_torch.shape)
a = 0
for i in range((len(dX_WB_ad))):
    a += dX_WB_ad[i].sum() * i
a.derivatives().reshape(dX_WB_torch.shape)

#%%
result_ad = backend.calc_objective(
    dX_WB_ad, X_WB_e_list, l_xy_e_list, X_WB_g, alpha=0.5)
print(result_ad["c"].derivatives())

#%%
result_ad = backend.calc_inner_layer(dX_WB_ad, l_xy_e_list, Omega)
(result_ad['H']**2).sum().derivatives()

#%% numerical diff
backend.calc_objective_gradient_finite_difference(
    dX_WB_np, X_WB_e_list, l_xy_e_list, X_WB_g, 0.5)


#%%
X_WB_p = backend.calc_pose_predictions(dX_WB_ad)
X_WB_p.sum().derivatives()


#%% profiling
