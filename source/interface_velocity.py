import torch


def update_interface_velocities(u, v, pressure, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, alpha):
    a_pu = torch.vstack((a_p[0, :], a_p, a_p[-1, :]))
    # u_ep = (u[1: x_nums + 2, 2: y_nums + 2] + u[2: x_nums + 3, 2: y_nums + 2]) / 2.0
    # p_eep = (pressure[3: x_nums + 4, 2: y_nums + 2] - pressure[1: x_nums + 2, 2: y_nums + 2]) / (4.0 * a_pu[1: x_nums + 2, :])
    # p_ew = (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[: x_nums + 1, 2: y_nums + 2]) / (4.0 * a_pu[: x_nums + 1, :])
    # a_pe = 0.5 * (1.0 / a_pu[1: x_nums + 2, :] + 1.0 / a_pu[: x_nums + 1, :]) * (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[2: x_nums + 3, 2: y_nums + 2])
    #
    # u_e = (u_ep + (p_eep + p_ew + a_pe)) * delta_y / (density * inlet_velocity ** 2)
    u_e = (((u[1: x_nums + 2, 2: y_nums + 2] + u[2: x_nums + 3, 2: y_nums + 2]) / 2.0
           + ((pressure[3: x_nums + 4, 2: y_nums + 2] - pressure[1: x_nums + 2, 2: y_nums + 2]) / (4.0 * a_pu[1: x_nums + 2, :])
              + (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[: x_nums + 1, 2: y_nums + 2]) / (4.0 * a_pu[: x_nums + 1, :])
              + 0.5 * (1.0 / a_pu[1: x_nums + 2, :] + 1.0 / a_pu[: x_nums + 1, :])
              * (pressure[1: x_nums + 2, 2: y_nums + 2] - pressure[2: x_nums + 3, 2: y_nums + 2]))) * delta_y
           / (density * inlet_velocity ** 2))

    a_pv = torch.hstack((a_p[:, [0]], a_p, a_p[:, [-1]]))
    v_np = (v[2: x_nums + 2, 1: y_nums + 2] + v[2: x_nums + 2, 2: y_nums + 3]) / 2.0
    p_nnp = (v[2: x_nums + 2, 3 + y_nums +4])

    a_pu_numpy = a_pu.cpu().numpy()
    # u_ep_numpy  = u_ep.cpu().numpy()
    # p_eep_numpy = p_eep.cpu().numpy()
    # p_ew_numpy = p_ew.cpu().numpy()
    # a_pe_numpy = a_pe.cpu().numpy()
    u_e_numpy = u_e.cpu().numpy()

    a_pv_numpy = a_pv.cpu().numpy()

    return u_e, v_n
