import torch


def quick(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y):
    alpha_e = (u_e[2: x_nums + 2, 2: y_nums + 2] > 0.0).int()
    alpha_w = (u_e[1: x_nums + 1, 2: y_nums + 2] > 0.0).int()
    alpha_n = (v_n[2: x_nums + 2, 2: y_nums + 2] > 0.0).int()
    alpha_s = (v_n[2: x_nums + 2, 1: y_nums + 1] > 0.0).int()

    # f_e = u_e[2: x_nums + 2, 2: y_nums + 2] * delta_y
    # f_w = u_e[1: x_nums + 1, 2: y_nums + 2] * delta_y
    # f_n = v_n[2: x_nums + 2, 2: y_nums + 2] * delta_x
    # f_s = v_n[2: x_nums + 2, 1: y_nums + 1] * delta_x

    a_e = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[3: x_nums + 3, 2: y_nums + 2].eq(0) , 1.0e-30,
                      d_e - 0.375 * alpha_e * u_e[2: x_nums + 2, 2: y_nums + 2] * delta_y
                      - 0.125 * (1.0 - alpha_w) * u_e[1: x_nums + 1, 2: y_nums + 2] * delta_y
                      - 0.75 * (1.0 - alpha_e) * u_e[2: x_nums + 2, 2: y_nums + 2] * delta_y)

    a_ee = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[4: x_nums + 4, 2: y_nums + 2].eq(0) , 1.0e-30,
                      0.125 * (1.0 - alpha_e) * u_e[2: x_nums + 2, 2: y_nums + 2] * delta_y)

    a_w = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[1: x_nums + 1, 2: y_nums + 2].eq(0) , 1.0e-30,
                      d_w + 0.75 * alpha_w * u_e[1: x_nums + 1, 2: y_nums + 2] * delta_y
                      + 0.125 * alpha_e * u_e[2: x_nums + 2, 2: y_nums + 2] * delta_y
                      + 0.375 * (1 - alpha_w) * u_e[1: x_nums + 1, 2: y_nums + 2] * delta_y)

    a_ww = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[: x_nums, 2: y_nums + 2].eq(0) , 1.0e-30,
                      -0.125 * alpha_w * u_e[1: x_nums + 1, 2: y_nums + 2] * delta_y)

    a_n = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[2: x_nums + 2, 3: y_nums + 3].eq(0), 1.0e-30,
                      d_n - 0.375 * alpha_n * v_n[2: x_nums + 2, 2: y_nums + 2] * delta_x
                      - 0.125 * (1.0 - alpha_s) * v_n[2: x_nums + 2, 1: y_nums + 1] * delta_x
                      - 0.75 * (1.0 - alpha_n) * v_n[2: x_nums + 2, 2: y_nums + 2] * delta_x)

    a_nn = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                       | flow_regions[2: x_nums + 2, 4: y_nums + 4].eq(0), 1.0e-30,
                       0.125 * (1.0 - alpha_n) * v_n[2: x_nums + 2, 2: y_nums + 2] * delta_x)

    a_s = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                      | flow_regions[2: x_nums + 2, 1: y_nums + 1].eq(0), 1.0e-30,
                      d_s + 0.75 * alpha_s * v_n[2: x_nums + 2, 1: y_nums + 1] * delta_x
                      + 0.125 * alpha_n * v_n[2: x_nums + 2, 2: y_nums + 2] * delta_x
                      + 0.375 * (1.0 - alpha_s) * v_n[2: x_nums + 2, 1: y_nums + 1] * delta_x)

    a_ss = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0)
                       | flow_regions[2: x_nums + 2, : y_nums].eq(0), 1.0e-30,
                       -0.125 * alpha_s * v_n[2: x_nums + 2, 1: y_nums + 1] * delta_x)

    a_p[2: x_nums + 2, 2: y_nums + 2] = (a_e + a_ee + a_w + a_ww + a_n + a_nn + a_s + a_ss
                                         + (u_e[2: x_nums + 2, 2: y_nums + 2] - u_e[1: x_nums + 1, 2: y_nums + 2]) * delta_y
                                         + (v_n[2: x_nums + 2, 2: y_nums + 2] - v_n[2: x_nums + 2, 1: y_nums + 1]) * delta_x)

    a_e_numpy = a_e.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    return a_p, a_e, a_ee, a_w, a_ww, a_n, a_nn, a_s, a_ss
