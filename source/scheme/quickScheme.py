import torch


def quick(u_e, v_n, d_e, d_w, d_n, d_s, xNums, yNums, xDelta, yDelta):
    alpha_e = torch.where(u_e[1: xNums + 1, :] > 0.0, 1.0, 0.0)
    alpha_w = torch.where(u_e[: xNums, :] > 0.0, 1.0, 0.0)
    alpha_n = torch.where(v_n[:, 1: yNums + 1] > 0.0, 1.0, 0.0)
    alpha_s = torch.where(v_n[:, : yNums] > 0.0, 1.0, 0.0)

    a_ww = -0.125 * alpha_w * (u_e[: xNums, :] * yDelta)
    a_w = d_w + 0.75 * alpha_w * (u_e[: xNums, :] * yDelta) + 0.125 * alpha_e * (u_e[1: xNums + 1, :] * yDelta) + 0.375 * (1 - alpha_w) * (u_e[: xNums, :] * yDelta)
    a_e = d_e - 0.375 * alpha_e * u_e[1: xNums + 1, :] * yDelta - 0.125 * (1.0 - alpha_w) * u_e[: xNums, :] * yDelta - 0.75 * (1.0 - alpha_e) * u_e[1: xNums + 1, :] * yDelta
    a_ee = 0.125 * (1.0 - alpha_e) * u_e[1: xNums + 1, :] * yDelta
    a_ss = -0.125 * alpha_s * (v_n[:, : yNums] * xDelta)
    a_s = d_s + 0.75 * alpha_s * (v_n[:, : yNums] * xDelta) + 0.125 * alpha_n * (v_n[:, 1: yNums + 1] * xDelta) + 0.375 * (1.0 - alpha_s) * (v_n[:, : yNums] * xDelta)
    a_n = d_n - 0.375 * alpha_n * (v_n[:, 1: yNums + 1] * xDelta) - 0.125 * (1.0 - alpha_s) * (v_n[:, : yNums] * xDelta) - 0.75 * (1.0 - alpha_n) * (v_n[:, 1: yNums + 1] * xDelta)
    a_nn = 0.125 * (1.0 - alpha_n) * (v_n[:, 1: yNums + 1] * xDelta)
    a_p = a_ww + a_w + a_e + a_ee + a_ss + a_s + a_nn + (u_e[1: xNums + 1, :] - u_e[: xNums, :]) * yDelta + (v_n[:, 1: yNums + 1] - v_n[:, : yNums]) * xDelta

    return a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p