import torch


def first_upwind_difference_scheme(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y):
    # f_e = (u_e[1: x_nums + 1, :] * delta_y)
    # f_w = (u_e[: x_nums, :] * delta_y)
    # f_n = (v_n[:, 1: y_nums + 1] * delta_x)
    # f_s = (v_n[:, : y_nums] * delta_x)
    zeros = torch.zeros_like(u_e[: x_nums, :])

    a_w = d_w + torch.max(u_e[: x_nums, :] * delta_y, zeros)
    a_e = d_e + torch.max(-u_e[1: x_nums + 1, :] * delta_y, zeros)
    a_s = d_s + torch.max(v_n[:, : y_nums] * delta_x, zeros)
    a_n = d_n + torch.max(-v_n[:, 1: y_nums + 1] * delta_y, zeros)

    a_p = a_w + a_e + a_s + a_n + ((u_e[1: x_nums + 1, :] * delta_y) - (u_e[: x_nums, :] * delta_y)) + ((v_n[:, 1: y_nums + 1] * delta_x) - (v_n[:, : y_nums] * delta_x))

    return a_w, a_e, a_s, a_n, a_p
