import torch


def central_difference(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity):
    # f_e = u_e[2: x_nums + 2, 2: y_nums + 2] * delta
    # f_w = u_e[1: x_nums + 1, 2: y_nums + 2] * delta
    # f_n = v_n[2: x_nums + 2, 2: y_nums + 2] * delta
    # f_s = v_n[2: x_nums + 2, 1: y_nums + 1] * delta
    #
    # f_e_numpy = f_e.cpu().numpy()
    # f_w_numpy = f_w.cpu().numpy()
    # f_n_numpy = f_n.cpu().numpy()
    # f_s_numpy = f_s.cpu().numpy()

    a_e = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[3: x_nums + 3, 2: y_nums + 2].eq(0), 1.0e-30,
                      d_e - 0.5 * u_e[2: x_nums + 2, 2: y_nums + 2] * delta)

    a_w = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[1: x_nums + 1, 2: y_nums + 2].eq(0), 1.0e-30,
                      d_w + 0.5 * u_e[1: x_nums + 1, 2: y_nums + 2] * delta)

    a_n = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[2: x_nums + 2, 3: y_nums + 3].eq(0), 1.0e-30,
                      d_n - 0.5 * v_n[2: x_nums + 2, 2: y_nums + 2] * delta)

    a_s = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[2: x_nums + 2, 1: y_nums + 1].eq(0), 1.0e-30,
                      d_s + 0.5 * v_n[2: x_nums + 2, 1: y_nums + 1] * delta)

    a_p[2: x_nums + 2, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 0.0,
                                                    (a_e + a_w + a_n + a_s + (u_e[2: x_nums + 2, 2: y_nums + 2]
                                                                              - u_e[1: x_nums + 1, 2: y_nums + 2]) * delta
                                                     + (v_n[2: x_nums + 2, 2: y_nums + 2]
                                                        - v_n[2: x_nums + 2, 1: y_nums + 1]) * delta
                                                     + hydraulic_diameter * delta ** 2 / (inlet_velocity * delta_time)))

    a_e_numpy = a_e.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    return a_p, a_e, a_w, a_n, a_s
