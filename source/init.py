import torch


def initFlowParams(xNums, yNums, density, viscosity, inletVelocities, inletPressure, device):
    # 流动区域
    flowRegions = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.bool, device=device)

    u = torch.where(flowRegions == True, inletVelocities[0], 0.0)
    v = torch.where(flowRegions == True, inletVelocities[1], 0.0)
    pressure = torch.where(flowRegions == True, inletPressure, 0.0)

    u_e = torch.zeros((xNums + 1, yNums), dtype=torch.float32, device=device)
    v_n = torch.zeros((xNums, yNums + 1), dtype=torch.float32, device=device)

    tauInit = viscosity / (density * (inletVelocities[0] ** 2 + inletVelocities[1] ** 2) ** 0.5)
    tau = torch.where(flowRegions == True, tauInit, 1.0e30)

    return u, v, pressure, u_e, v_n, tau
