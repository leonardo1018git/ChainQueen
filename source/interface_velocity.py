import torch


def update_interface_velocities(u, v, pressure, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity):
    a_p_numpy= a_p.cpu().numpy()
    # u_ep = (u[1: x_nums + 2, 2: y_nums + 2] + u[2: x_nums + 3, 2: y_nums + 2]) / 2.0
    # p_eep = (pressure[3: x_nums + 4, 2: y_nums + 2] - pressure[1: x_nums + 2, 2: y_nums + 2]) / (4.0 * a_pu[1: x_nums + 2, :])
    # p_ew = (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[: x_nums + 1, 2: y_nums + 2]) / (4.0 * a_pu[: x_nums + 1, :])
    # a_pe = 0.5 * (1.0 / a_pu[1: x_nums + 2, :] + 1.0 / a_pu[: x_nums + 1, :]) * (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[2: x_nums + 3, 2: y_nums + 2])
    #
    # u_e = u_ep + (p_eep + p_ew + a_pe) * delta_y / (density * inlet_velocity ** 2)
    u_e = ((u[1: x_nums + 2, 2: y_nums + 2] + u[2: x_nums + 3, 2: y_nums + 2]) / 2.0
           + ((pressure[3: x_nums + 4, 2: y_nums + 2] - pressure[1: x_nums + 2, 2: y_nums + 2]) / (4.0 * a_p[2: x_nums + 3, 2: y_nums + 2])
              + (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[: x_nums + 1, 2: y_nums + 2]) / (4.0 * a_p[1: x_nums + 2, 2: y_nums + 2])
              + 0.5 * (1.0 / a_p[2: x_nums + 3, 2: y_nums + 2] + 1.0 / a_p[1: x_nums + 2, 2: y_nums + 2]) * (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[2: x_nums + 3, 2: y_nums + 2]))
           * delta_y / (density * inlet_velocity ** 2))

    # v_np = (v[2: x_nums + 2, 1: y_nums + 2] + v[2: x_nums + 2, 2: y_nums + 3]) / 2.0
    # p_nnp = (pressure[2: x_nums + 2, 3: y_nums + 4] - pressure[2: x_nums + 2, 1: y_nums + 2]) / (4.0 * a_pv[:, 1: y_nums + 2])
    # p_ns = (pressure[2: x_nums + 2, 2: y_nums + 3] - pressure[2: x_nums + 2, : y_nums + 1]) / (4.0 * a_pv[:, : y_nums + 1])
    # a_pn = 0.5 * (1.0 / a_pv[:, 1: y_nums + 2] + 1.0 / a_pv[:, : y_nums + 1]) * (pressure[2: x_nums + 2, 1: y_nums + 2] - pressure[2: x_nums + 2, 2: y_nums + 3])
    # v_n = v_np + (p_nnp + p_ns + a_pn) * delta_x / (density * inlet_velocity ** 2)

    v_n = ((v[2: x_nums + 2, 1: y_nums + 2] + v[2: x_nums + 2, 2: y_nums + 3]) / 2.0
           + ((pressure[2: x_nums + 2, 3: y_nums + 4] - pressure[2: x_nums + 2, 1: y_nums + 2]) / (4.0 * a_p[2: x_nums + 2, 2: y_nums + 3])
              + (pressure[2: x_nums + 2, 2: y_nums + 3] - pressure[2: x_nums + 2, : y_nums + 1]) / (4.0 * a_p[2: x_nums + 2, 1: y_nums + 2])
              + 0.5 * (1.0 / a_p[2: x_nums + 2, 2: y_nums + 3] + 1.0 / a_p[2: x_nums + 2, 1: y_nums + 2]) * (pressure[2: x_nums + 2, 1: y_nums + 2] - pressure[2: x_nums + 2, 2: y_nums + 3]))
           * delta_x / (density * inlet_velocity ** 2))

    # u_ep_numpy  = u_ep.cpu().numpy()
    # p_eep_numpy = p_eep.cpu().numpy()
    # p_ew_numpy = p_ew.cpu().numpy()
    # a_pe_numpy = a_pe.cpu().numpy()
    u_e_numpy = u_e.cpu().numpy()

    # v_np_numpy = v_np.cpu().numpy()
    # p_nnp_numpy = p_nnp.cpu().numpy()
    # p_ns_numpy = p_ns.cpu().numpy()
    # a_pn_numpy = a_pn.cpu().numpy()
    v_n_numpy = v_n.cpu().numpy()

    return u_e, v_n
