import torch
from source.scheme.quick_scheme import quick
from source.scheme.sud_scheme import second_upwind_difference_scheme
from source.scheme.cd_scheme import central_difference_scheme
from source.scheme.fud_scheme import first_upwind_difference_scheme
from source.scheme.hybrid_scheme import hybrid_scheme
from source.interface_velocity import update_interface_velocities


def velocity_solver(scheme, u, v, pressure, u_e, v_n, tau, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inner_epochs):
    a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    y_division_x = delta_y / (2.0 * delta_x)
    x_division_y = delta_x / (2.0 * delta_y)

    tau_numpy = tau.cpu().numpy()

    d_e = (tau[3: x_nums + 3, 2: y_nums + 2] + tau[2: x_nums + 2, 2: y_nums + 2]) * y_division_x
    d_w = (tau[1: x_nums + 1, 2: y_nums + 2] + tau[2: x_nums + 2, 2: y_nums + 2]) * y_division_x
    d_n = (tau[2: x_nums + 2, 3: y_nums + 3] + tau[2: x_nums + 2, 2: y_nums + 2]) * x_division_y
    d_s = (tau[2: x_nums + 2, 1: y_nums + 1] + tau[2: x_nums + 2, 2: y_nums + 2]) * x_division_y

    d_e_numpy = d_e.cpu().numpy()
    d_w_numpy = d_w.cpu().numpy()
    d_n_numpy = d_n.cpu().numpy()
    d_s_numpy = d_s.cpu().numpy()

    # f_e = u_e[1: x_nums + 1, :] * y_delta
    # f_w = u_e[: x_nums, :] * y_delta
    # f_n = v_n[:, 1: y_nums + 1] * x_delta
    # f_s = v_n[:, : y_nums] * x_delta

    pressure_numpy = pressure.cpu().numpy()

    u_source = (pressure[3: x_nums + 3, 2: y_nums + 2] - pressure[1: x_nums + 1, 2: y_nums + 2]) * delta_y / (2.0 * density * inlet_velocity ** 2)
    v_source = (pressure[2: x_nums + 2, 3: y_nums + 3] - pressure[2: x_nums + 2, 1: y_nums + 1]) * delta_x / (2.0 * density * inlet_velocity ** 2)

    u_source_numpy = u_source.cpu().numpy()
    v_source_numpy = v_source.cpu().numpy()

    if scheme == "QUICK":
        # Quick格式
        a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p[2: x_nums + 2, 2: y_nums + 2] = quick(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y)
    elif scheme == "SUD":
        # 二阶迎风格式
        a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p[2: x_nums + 2, 2: y_nums + 2] = second_upwind_difference_scheme(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y)
    elif scheme == "CD":
        # 中心差分格式
        a_w, a_e, a_s, a_n, a_p[2: x_nums + 2, 2: y_nums + 2] = central_difference_scheme(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y)
    elif scheme == "FUD":
        # 一阶迎风格式
        a_w, a_e, a_s, a_n, a_p[2: x_nums + 2, 2: y_nums + 2] = first_upwind_difference_scheme(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y)
    elif scheme == "Hybrid":
        # 一阶迎风格式
        a_w, a_e, a_s, a_n, a_p[2: x_nums + 2, 2: y_nums + 2] = hybrid_scheme(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y)

    a_p[:, 0], a_p[:, 1] = a_p[:, 2], a_p[:, 2]
    a_p[:, -1], a_p[:, -2] = a_p[:, -3], a_p[:, -3]

    a_p[0, :], a_p[1, :] = a_p[2, :], a_p[2, :]
    a_p[-1, :], a_p[-2, :] = a_p[-3, :], a_p[-3, :]

    a_e_numpy = a_e.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()

    for epoch in range(inner_epochs):
        u_old, v_old = u.clone(), v.clone()
        u[2: x_nums + 2, 2: y_nums + 2] = (a_e * u[3: x_nums + 3, 2: y_nums + 2] + a_ee * u[4: x_nums + 4, 2: y_nums + 2]
                                         + a_w * u[1: x_nums + 1, 2: y_nums + 2] + a_ww * u[: x_nums, 2: y_nums + 2]
                                         + a_n * u[2: x_nums + 2, 3: y_nums + 3] + a_nn * u[2: x_nums + 2, 4: y_nums + 4]
                                         + a_s * u[2: x_nums + 2, 1: y_nums + 1] + a_ss * u[2: x_nums + 2, : y_nums]
                                         + u_source) / a_p[2: x_nums + 2, 2: y_nums + 2]
        v[2: x_nums + 2, 2: y_nums + 2] = (a_e * v[3: x_nums + 3, 2: y_nums + 2] + a_ee * v[4: x_nums + 4, 2: y_nums + 2]
                                         + a_w * v[1: x_nums + 1, 2: y_nums + 2] + a_ww * v[: x_nums, 2: y_nums + 2]
                                         + a_n * v[2: x_nums + 2, 3: y_nums + 3] + a_nn * v[2: x_nums + 2, 4: y_nums + 4]
                                         + a_s * v[2: x_nums + 2, 1: y_nums + 1] + a_ss * v[2: x_nums + 2, : y_nums]
                                         + v_source) / a_p[2: x_nums + 2, 2: y_nums + 2]

        u[2: x_nums + 2, 2: y_nums + 2] = torch.where(tau[2: x_nums + 2, 2: y_nums + 2] < 1.0e30, u[2: x_nums + 2, 2: y_nums + 2], 0.0)
        v[2: x_nums + 2, 2: y_nums + 2] = torch.where(tau[2: x_nums + 2, 2: y_nums + 2] < 1.0e30, v[2: x_nums + 2, 2: y_nums + 2], 0.0)

        u_numpy = u.cpu().numpy()
        v_numpy = v.cpu().numpy()
        u_error, v_error = torch.max(torch.abs(u_old - u) * 100 / inlet_velocity), torch.max(torch.abs(v_old - v) * 100 / inlet_velocity)
        print("epoch = " + str(epoch) + ", uError = " + str(u_error.cpu().numpy()) + ", vError = " + str(v_error.cpu().numpy()))
        if u_error.cpu().numpy() < 1.0e-6 and v_error.cpu().numpy() < 1.0e-6:
            break

    u[2: x_nums + 2, -1], u[2: x_nums + 2, -2] = u[2: x_nums + 2, -3],u[2: x_nums + 2, -3]
    # u[0, :], u[1, :] = u[2, :], u[2, :]
    # u[-1, :], u[-2, :] = u[-3, :], u[-3, :]

    v[2: x_nums + 2, -1], v[2: x_nums + 2, -2] = v[2: x_nums + 2, -3], v[2: x_nums + 2, -3]
    # v[0, :], v[1, :] = v[2, :], v[2, :]
    # v[-1, :], v[-2, :] = v[-3, :], v[-3, :]

    d_e_numpy = d_e.cpu().numpy()
    d_w_numpy = d_w.cpu().numpy()
    d_n_numpy = d_n.cpu().numpy()
    d_s_numpy = d_s.cpu().numpy()

    # a_ee_numpy = a_ee.cpu().numpy()
    # a_ww_numpy = a_ww.cpu().numpy()
    # a_nn_numpy = a_nn.cpu().numpy()
    # a_ss_numpy = a_ss.cpu().numpy()

    a_e_numpy = a_e.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()

    return u, v, a_p


def pressure_solver(pressure, u_e, v_n, tau, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inlet_pressure, inner_epochs):
    p_prime = torch.zeros_like(pressure)

    a_pe = (delta_y ** 2) * (1.0 / a_p[3: x_nums + 3, 2: y_nums + 2] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2]) / (density * inlet_velocity ** 2)
    a_pw = (delta_y ** 2) * (1.0 / a_p[1: x_nums + 1, 2: y_nums + 2] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2]) / (density * inlet_velocity ** 2)
    a_pn = (delta_x ** 2) * (1.0 / a_p[2: x_nums + 2, 3: y_nums + 3] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2]) / (density * inlet_velocity ** 2)
    a_ps = (delta_x ** 2) * (1.0 / a_p[2: x_nums + 2, 1: y_nums + 1] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2]) / (density * inlet_velocity ** 2)
    a_pp = a_pe + a_pw + a_pn + a_ps

    a_p_numpy = a_p.cpu().numpy()
    a_pe_numpy = a_pe.cpu().numpy()
    a_pw_numpy = a_pw.cpu().numpy()
    a_pn_numpy = a_pn.cpu().numpy()
    a_ps_numpy = a_ps.cpu().numpy()
    a_pp_numpy = a_pp.cpu().numpy()

    for epoch in range(inner_epochs):
        p_old = p_prime.clone()
        p_prime[2: x_nums + 2, 2: y_nums + 2] \
            = ((a_pe * p_prime[3: x_nums + 3, 2: y_nums + 2] + a_pw * p_prime[1: x_nums + 1, 2: y_nums + 2]
                + a_pn * p_prime[2: x_nums + 2, 3: y_nums + 3] + a_ps * p_prime[2: x_nums + 2, 1: y_nums + 1]
                + (u_e[: x_nums, :] - u_e[1: x_nums + 1, :]) * delta_y
                + (v_n[:, : y_nums] - v_n[:, 1: y_nums + 1]) * delta_x) / a_pp)

        p_prime[2: x_nums + 2, 2: y_nums + 2] = torch.where(tau[2: x_nums + 2, 2: y_nums + 2] < 1.0e30, p_prime[2: x_nums + 2, 2: y_nums + 2], 0.0)

        p_error = torch.max(torch.abs(p_old[2: x_nums + 2, 2: y_nums + 2] - p_prime[2: x_nums + 2, 2: y_nums + 2]) * 100 / inlet_pressure)
        print("epoch = " + str(epoch) + ", pError = " + str(p_error.cpu().numpy()))
        if p_error.cpu().numpy() < 1.0e-6:
            break

    p_prime[2: x_nums + 2, -1], p_prime[2: x_nums + 2, -2] = p_prime[2: x_nums + 2, -3], p_prime[2: x_nums + 2, -3]
    p_prime_numpy = p_prime.cpu().numpy()

    return p_prime


def correct_params(u, v, u_e, v_n, p_prime, tau, a_p, x_nums, y_nums, delta_x, delta_y, alpha, density, inlet_velocity):
    u[2: x_nums + 2, 2: y_nums + 2] += 0.5 * alpha * (p_prime[1: x_nums + 1, 2: y_nums + 2] - p_prime[3: x_nums + 3, 2: y_nums + 2]) * delta_y / (density * inlet_velocity ** 2 * a_p[2: x_nums + 2, 2: y_nums + 2])
    v[2: x_nums + 2, 2: y_nums + 2] += 0.5 * alpha * (p_prime[2: x_nums + 2, 1: y_nums + 1] - p_prime[2: x_nums + 2, 3: y_nums + 3]) * delta_x / (density * inlet_velocity ** 2 * a_p[2: x_nums + 2, 2: y_nums + 2])

    u[2: x_nums + 2, 2: y_nums + 2] = torch.where(tau[2: x_nums + 2, 2: y_nums + 2] < 1.0e30, u[2: x_nums + 2, 2: y_nums + 2], 0.0)
    v[2: x_nums + 2, 2: y_nums + 2] = torch.where(tau[2: x_nums + 2, 2: y_nums + 2] < 1.0e30, v[2: x_nums + 2, 2: y_nums + 2], 0.0)

    u_e += 0.5 * alpha * (1.0 / a_p[1: x_nums + 2, 2: y_nums + 2] + 1.0 / a_p[2: x_nums + 3, 2: y_nums + 2]) * (p_prime[1: x_nums + 2, 2: y_nums + 2] - p_prime[2: x_nums + 3, 2: y_nums + 2]) * delta_y / (density * inlet_velocity ** 2)
    v_n += 0.5 * alpha * (1.0 / a_p[2: x_nums + 2, 1: y_nums + 2] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 3]) * (p_prime[2: x_nums + 2, 1: y_nums + 2] - p_prime[2: x_nums + 2, 2: y_nums + 3]) * delta_x / (density * inlet_velocity ** 2)

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    u_e_numpy = u_e.cpu().numpy()
    v_n_numpy = v_n.cpu().numpy()

    return u, v, u_e, v_n


def fvm_solver(scheme, u, v, pressure, u_e, v_n, tau, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inlet_pressure, inner_epochs, outer_epochs, alpha):

    for epoch in range(outer_epochs):
        u_old, v_old = u.clone(), v.clone()
        u, v, a_p = velocity_solver(scheme, u, v, pressure, u_e, v_n, tau, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inner_epochs)
        u_e, v_n = update_interface_velocities(u, v, pressure, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity)
        p_prime = pressure_solver(pressure, u_e, v_n, tau, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inlet_pressure, inner_epochs)
        u, v, u_e, v_n = correct_params(u, v, u_e, v_n, p_prime, tau, a_p, x_nums, y_nums, delta_x, delta_y, alpha, density, inlet_velocity)


    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    p_numpy = pressure.cpu().numpy()
    tau_numpy = tau.cpu().numpy()
    return u, v, pressure
