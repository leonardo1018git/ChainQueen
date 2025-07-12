import torch
from source.scheme.quickScheme import quick


def velocitySolver(scheme, u, v, pressure, u_e, v_n, tau, xNums, yNums, xDelta, yDelta, density, inletVelocity, innerEpochs):
    a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    yDivisionX = yDelta / xDelta
    xDivisionY = xDelta / yDelta

    d_e = 2.0 * tau[3: xNums + 3, 2: yNums + 2] * tau[2: xNums + 2, 2: yNums + 2] / (tau[3: xNums + 3, 2: yNums + 2] + tau[2: xNums + 2, 2: yNums + 2]) * yDivisionX
    d_w = 2.0 * tau[1: xNums + 1, 2: yNums + 2] * tau[2: xNums + 2, 2: yNums + 2] / (tau[1: xNums + 1, 2: yNums + 2] + tau[2: xNums + 2, 2: yNums + 2]) * yDivisionX
    d_n = 2.0 * tau[2: xNums + 2, 3: yNums + 3] * tau[2: xNums + 2, 2: yNums + 2] / (tau[2: xNums + 2, 3: yNums + 3] + tau[2: xNums + 2, 2: yNums + 2]) * xDivisionY
    d_s = 2.0 * tau[2: xNums + 2, 1: yNums + 1] * tau[2: xNums + 2, 2: yNums + 2] / (tau[2: xNums + 2, 1: yNums + 1] + tau[2: xNums + 2, 2: yNums + 2]) * xDivisionY

    # f_e = u_e[1: xNums + 1, :] * yDelta
    # f_w = u_e[: xNums, :] * yDelta
    # f_n = v_n[:, 1: yNums + 1] * xDelta
    # f_s = v_n[:, : yNums] * xDelta

    uSource = (pressure[3: xNums + 3, 2: yNums + 2] - pressure[1: xNums + 1, 2: yNums + 2]) * yDelta / (2.0 * density * inletVelocity ** 2)
    vSource = (pressure[2: xNums + 2, 3: yNums + 3] - pressure[2: xNums + 2, 1: yNums + 1]) * xDelta / (2.0 * density * inletVelocity ** 2)

    uSource_numpy = uSource.cpu().numpy()
    vSource_numpy = vSource.cpu().numpy()

    # Quick格式
    if scheme == "Quick":
        a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn, a_p = quick(u_e, v_n, d_e, d_w, d_n, d_s, xNums, yNums, xDelta, yDelta)

    for epoch in range(innerEpochs):
        uOld, vOld = u.clone(), v.clone()
        u[2: xNums + 2, 2: yNums + 2] = (a_ww * u[: xNums, 2: yNums + 2] + a_w * u[1: xNums + 1, 2: yNums + 2]
                                         + a_e * u[3: xNums + 3, 2: yNums + 2] + a_ee * u[4: xNums + 4, 2: yNums + 2]
                                         + a_ss * u[2: xNums + 2, : yNums] + a_s * u[2: xNums + 2, 1: yNums + 1]
                                         + a_n * u[2: xNums + 2, 3: yNums + 3] + a_nn * u[2: xNums + 2, 4: yNums + 4]
                                         + uSource) / a_p
        v[2: xNums + 2, 2: yNums + 2] = (a_ww * v[: xNums, 2: yNums + 2] + a_w * v[1: xNums + 1, 2: yNums + 2]
                                         + a_e * v[3: xNums + 3, 2: yNums + 2] + a_ee * v[4: xNums + 4, 2: yNums + 2]
                                         + a_ss * v[2: xNums + 2, : yNums] + a_s * v[2: xNums + 2, 1: yNums + 1]
                                         + a_n * v[2: xNums + 2, 3: yNums + 3] + a_nn * v[2: xNums + 2, 4: yNums + 4]
                                         + vSource) / a_p

        u_numpy = u.cpu().numpy()
        v_numpy = v.cpu().numpy()
        uError, vError = torch.max(torch.abs(uOld - u) * 100 / inletVelocity), torch.max(torch.abs(vOld - v) * 100 / inletVelocity)
        print("epoch = " + str(epoch) + ", uError = " + str(uError.cpu().numpy()) + ", vError = " + str(vError.cpu().numpy()))
        print()

    d_e_numpy = d_e.cpu().numpy()
    d_w_numpy = d_w.cpu().numpy()
    d_n_numpy = d_n.cpu().numpy()
    d_s_numpy = d_s.cpu().numpy()

    a_ww_numpy = a_ww.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_e_numpy = a_e.cpu().numpy()
    a_ee_numpy = a_ee.cpu().numpy()
    a_ss_numpy = a_ss.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_nn_numpy = a_nn.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()

    return u, v


def fvmSolver(scheme, u, v, pressure, u_e, v_n, tau, xNums, yNums, xDelta, yDelta, density, inletVelocity,innerEpochs, outerEpochs, device):

    u, v = velocitySolver(scheme, u, v, pressure, u_e, v_n, tau, xNums, yNums, xDelta, yDelta, density, inletVelocity, innerEpochs)



    uNumpy = u.cpu().numpy()
    vNumpy = v.cpu().numpy()
    pNumpy = pressure.cpu().numpy()
    tauNumpy = tau.cpu().numpy()
    return u, v, pressure
