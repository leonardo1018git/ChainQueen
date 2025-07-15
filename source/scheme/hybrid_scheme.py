import torch


def hybrid_scheme(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y):
    # f_e = (u_e[1: x_nums + 1, :] * delta_y)
    # f_w = (u_e[: x_nums, :] * delta_y)
    # f_n = (v_n[:, 1: y_nums + 1] * delta_x)
    # f_s = (v_n[:, : y_nums] * delta_x)
    zeros = torch.zeros_like(u_e[: x_nums, :])

    a_w = torch.max(torch.max(u_e[: x_nums, :] * delta_y, d_w + 0.5 * u_e[: x_nums, :] * delta_y), zeros)
    a_e = torch.max(torch.max(-u_e[1: x_nums + 1, :] * delta_y, d_e - 0.5 * u_e[1: x_nums + 1, :] * delta_y), zeros)
    a_s = torch.max(torch.max(v_n[:, : y_nums] * delta_x, d_s + 0.5 * v_n[:, : y_nums] * delta_x), zeros)
    a_n = torch.max(torch.max(-v_n[:, 1: y_nums + 1] * delta_x, d_n - 0.5 * v_n[:, 1: y_nums + 1] * delta_x), zeros)

    a_p = a_w + a_e + a_s + a_n + ((u_e[1: x_nums + 1, :] * delta_y) - (u_e[: x_nums, :] * delta_y)) + ((v_n[:, 1: y_nums + 1] * delta_x) - (v_n[:, : y_nums] * delta_x))

    return a_w, a_e, a_s, a_n, a_p