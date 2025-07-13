import torch
from source.scheme.quick_scheme import quick
from source.scheme.sud_scheme import second_upwind_difference_scheme


def velocity_solver(scheme, u, v, pressure, u_e, v_n, tau, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inner_epochs):
    a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    y_division_x = delta_y / (2.0 * delta_x)
    x_division_y = delta_x / (2.0 * delta_y)

    d_e = (tau[3: x_nums + 3, 2: y_nums + 2] + tau[2: x_nums + 2, 2: y_nums + 2]) * y_division_x
    d_w = (tau[1: x_nums + 1, 2: y_nums + 2] + tau[2: x_nums + 2, 2: y_nums + 2]) * y_division_x
    d_n = (tau[2: x_nums + 2, 3: y_nums + 3] + tau[2: x_nums + 2, 2: y_nums + 2]) * x_division_y
    d_s = (tau[2: x_nums + 2, 1: y_nums + 1] + tau[2: x_nums + 2, 2: y_nums + 2]) * x_division_y

    # f_e = u_e[1: x_nums + 1, :] * y_delta
    # f_w = u_e[: x_nums, :] * y_delta
    # f_n = v_n[:, 1: y_nums + 1] * x_delta
    # f_s = v_n[:, : y_nums] * x_delta

    u_source = (pressure[3: x_nums + 3, 2: y_nums + 2] - pressure[1: x_nums + 1, 2: y_nums + 2]) * delta_y / (2.0 * density * inlet_velocity ** 2)
    v_source = (pressure[2: x_nums + 2, 3: y_nums + 3] - pressure[2: x_nums + 2, 1: y_nums + 1]) * delta_x / (2.0 * density * inlet_velocity ** 2)

    u_source_numpy = u_source.cpu().numpy()
    v_source_numpy = v_source.cpu().numpy()

    if scheme == "QUICK":
        # Quick格式
        a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p = quick(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y)
    elif scheme == "SUD":
        # 二阶迎风格式
        a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p = second_upwind_difference_scheme(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y)

    for epoch in range(inner_epochs):
        u_old, v_old = u.clone(), v.clone()
        u[2: x_nums + 2, 2: y_nums + 2] = (a_ww * u[: x_nums, 2: y_nums + 2] + a_w * u[1: x_nums + 1, 2: y_nums + 2]
                                         + a_e * u[3: x_nums + 3, 2: y_nums + 2] + a_ee * u[4: x_nums + 4, 2: y_nums + 2]
                                         + a_ss * u[2: x_nums + 2, : y_nums] + a_s * u[2: x_nums + 2, 1: y_nums + 1]
                                         + a_n * u[2: x_nums + 2, 3: y_nums + 3] + a_nn * u[2: x_nums + 2, 4: y_nums + 4]
                                         + u_source) / a_p
        v[2: x_nums + 2, 2: y_nums + 2] = (a_ww * v[: x_nums, 2: y_nums + 2] + a_w * v[1: x_nums + 1, 2: y_nums + 2]
                                         + a_e * v[3: x_nums + 3, 2: y_nums + 2] + a_ee * v[4: x_nums + 4, 2: y_nums + 2]
                                         + a_ss * v[2: x_nums + 2, : y_nums] + a_s * v[2: x_nums + 2, 1: y_nums + 1]
                                         + a_n * v[2: x_nums + 2, 3: y_nums + 3] + a_nn * v[2: x_nums + 2, 4: y_nums + 4]
                                         + v_source) / a_p

        u[2: x_nums + 2, 2: y_nums + 2] = torch.where(tau[2: x_nums + 2, 2: y_nums + 2] < 1.0e30, u[2: x_nums + 2, 2: y_nums + 2], 0.0)
        v[2: x_nums + 2, 2: y_nums + 2] = torch.where(tau[2: x_nums + 2, 2: y_nums + 2] < 1.0e30, v[2: x_nums + 2, 2: y_nums + 2], 0.0)

        u_numpy = u.cpu().numpy()
        v_numpy = v.cpu().numpy()
        u_error, v_error = torch.max(torch.abs(u_old - u) * 100 / inlet_velocity), torch.max(torch.abs(v_old - v) * 100 / inlet_velocity)
        print("epoch = " + str(epoch) + ", uError = " + str(u_error.cpu().numpy()) + ", vError = " + str(v_error.cpu().numpy()))
        if u_error.cpu().numpy() < 1.0e-6 and v_error.cpu().numpy() < 1.0e-6:
            break

    d_e_numpy = d_e.cpu().numpy()
    d_w_numpy = d_w.cpu().numpy()
    d_n_numpy = d_n.cpu().numpy()
    d_s_numpy = d_s.cpu().numpy()

    a_ww_numpy = a_ww.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_e_numpy = a_e.cpu().numpy()
    a_ee_numpy = a_ee.cpu().numpy()
    a_ss_numpy = a_ss.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_nn_numpy = a_nn.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()

    a = 1

    return u, v


def fvm_solver(scheme, u, v, pressure, u_e, v_n, tau, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity,inner_epochs, outer_epochs, device):

    u, v = velocity_solver(scheme, u, v, pressure, u_e, v_n, tau, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inner_epochs)



    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    p_numpy = pressure.cpu().numpy()
    tau_numpy = tau.cpu().numpy()
    return u, v, pressure
