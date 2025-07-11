import torch


def velocitySolver(scheme, u, v, tau, xNums, yNums, xDelta, yDelta):
    # Quick格式
    if scheme == "Quick":
        tau_e = tau[3: xNums + 3, 2: yNums + 2]
        tau_w = tau[1: xNums + 1, 2: yNums + 2]
        tau_n = tau[2: xNums + 2, 3: yNums + 3]
        tau_s = tau[2: xNums + 2, 1: yNums + 1]

        d_e = tau[3: xNums + 3, 2: yNums + 2] * yDelta / xDelta
        d_w = tau[1: xNums + 1, 2: yNums + 2] * yDelta / xDelta
        d_n = tau[2: xNums + 2, 3: yNums + 3] * xDelta / yDelta
        d_s = tau[2: xNums + 2, 1: yNums + 1] * xDelta / yDelta

        f_e = (u[2: xNums + 2, 2: yNums + 2] + u[3: xNums + 3, 2: yNums + 2]) / 2.0
        f_w = (u[2: xNums + 2, 2: yNums + 2] + u[1: xNums + 1, 2: yNums + 2]) / 2.0
        f_n = (v[2: xNums + 2, 2: yNums + 2] + v[2: xNums + 2, 3: yNums + 3]) / 2.0
        f_s = (v[2: xNums + 2, 2: yNums + 2] + v[2: xNums + 2, 1: yNums + 1]) / 2.0

        alpha_e = torch.where(f_e > 0.0, 1.0, 0.0)
        alpha_w = torch.where(f_w > 0.0, 1.0, 0.0)
        alpha_n = torch.where(f_n > 0.0, 1.0, 0.0)
        alpha_s = torch.where(f_s > 0.0, 1.0, 0.0)

        a_ww = -0.125 * alpha_w * f_w
        a_w = d_w + 0.75 * alpha_w * f_w + 0.125 * alpha_e * f_e + 0.375 * (1 - alpha_w) * f_w
        a_e = d_e - 0.375 * alpha_e * f_e - 0.125 * (1.0 - alpha_w) * f_w - 0.75 * (1.0 - alpha_e) * f_e
        a_ee = 0.125 * (1.0 - alpha_e) * f_e

        tau_numpy = tau.cpu().numpy()
        tau_e_numpy = tau_e.cpu().numpy()
        tau_w_numpy = tau_w.cpu().numpy()
        tau_n_numpy = tau_n.cpu().numpy()
        tau_s_numpy = tau_s.cpu().numpy()

        d_e_numpy = d_e.cpu().numpy()
        d_w_numpy = d_w.cpu().numpy()
        d_n_numpy = d_n.cpu().numpy()
        d_s_numpy = d_s.cpu().numpy()

        f_e_numpy = f_e.cpu().numpy()
        f_w_numpy = f_w.cpu().numpy()
        f_n_numpy = f_n.cpu().numpy()
        f_s_numpy = f_s.cpu().numpy()

        a_ww_numpy = a_ww.cpu().numpy()
        a_w_numpy = a_ww.cpu().numpy()


    return u, v


def fvmSolver(scheme, u, v, pressure, tau, xNums, yNums, xDelta, yDelta, device):

    u, v = velocitySolver(scheme, u, v, tau, xNums, yNums, xDelta, yDelta)



    uNumpy = u.cpu().numpy()
    vNumpy = v.cpu().numpy()
    pNumpy = pressure.cpu().numpy()
    tauNumpy = tau.cpu().numpy()
    return u, v, pressure
