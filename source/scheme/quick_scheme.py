import torch


def quick(u_e, v_n, d_e, d_w, d_n, d_s, x_nums, y_nums, delta_x, delta_y):
    alpha_e = torch.where(u_e[1: x_nums + 1, :] > 0.0, 1.0, 0.0)
    alpha_w = torch.where(u_e[: x_nums, :] > 0.0, 1.0, 0.0)
    alpha_n = torch.where(v_n[:, 1: y_nums + 1] > 0.0, 1.0, 0.0)
    alpha_s = torch.where(v_n[:, : y_nums] > 0.0, 1.0, 0.0)

    a_ww = -0.125 * alpha_w * (u_e[: x_nums, :] * delta_y)
    a_w = d_w + 0.75 * alpha_w * (u_e[: x_nums, :] * delta_y) + 0.125 * alpha_e * (u_e[1: x_nums + 1, :] * delta_y) + 0.375 * (1 - alpha_w) * (u_e[: x_nums, :] * delta_y)
    a_e = d_e - 0.375 * alpha_e * u_e[1: x_nums + 1, :] * delta_y - 0.125 * (1.0 - alpha_w) * u_e[: x_nums, :] * delta_y - 0.75 * (1.0 - alpha_e) * u_e[1: x_nums + 1, :] * delta_y
    a_ee = 0.125 * (1.0 - alpha_e) * u_e[1: x_nums + 1, :] * delta_y
    a_ss = -0.125 * alpha_s * (v_n[:, : y_nums] * delta_x)
    a_s = d_s + 0.75 * alpha_s * (v_n[:, : y_nums] * delta_x) + 0.125 * alpha_n * (v_n[:, 1: y_nums + 1] * delta_x) + 0.375 * (1.0 - alpha_s) * (v_n[:, : y_nums] * delta_x)
    a_n = d_n - 0.375 * alpha_n * (v_n[:, 1: y_nums + 1] * delta_x) - 0.125 * (1.0 - alpha_s) * (v_n[:, : y_nums] * delta_x) - 0.75 * (1.0 - alpha_n) * (v_n[:, 1: y_nums + 1] * delta_x)
    a_nn = 0.125 * (1.0 - alpha_n) * (v_n[:, 1: y_nums + 1] * delta_x)
    a_p = a_ww + a_w + a_e + a_ee + a_ss + a_s + a_nn + (u_e[1: x_nums + 1, :] - u_e[: x_nums, :]) * delta_y + (v_n[:, 1: y_nums + 1] - v_n[:, : y_nums]) * delta_x

    return a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p