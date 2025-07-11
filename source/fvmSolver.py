import torch


def fvmSolver(u, v, pressure, tau, xDelta, yDelta):
    c = 1.0

    uNumpy = u.cpu().numpy()
    vNumpy = v.cpu().numpy()
    pNumpy = pressure.cpu().numpy()
    tauNumpy = tau.cpu().numpy()
    return u, v, pressure
