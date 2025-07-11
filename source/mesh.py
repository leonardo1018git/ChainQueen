import torch


def initFlowParams(xGridNums, yGridNums, density, viscosity, inletVelocities, inletPressure):
    u = torch.zeros((xGridNums + 4, yGridNums + 4), dtype=torch.float32, device='cuda')
    v = torch.zeros((xGridNums + 4, yGridNums + 4), dtype=torch.float32, device='cuda')

    pressure = torch.zeros((xGridNums + 4, yGridNums + 4), dtype=torch.float32, device='cuda')
    tauInit = viscosity / (density * (inletVelocities[0] ** 2 + inletVelocities[1] ** 2) ** 0.5)
    tau = torch.full((xGridNums + 4, yGridNums + 4), tauInit, dtype=torch.float32, device='cuda')

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
    ], dtype=torch.bool, device='cuda')

    u = torch.where(flowRegions == True, inletVelocities[0], 0.0)
    v = torch.where(flowRegions == True, inletVelocities[1], 0.0)
    pressure = torch.where(flowRegions == True, inletPressure, 0.0)
    tau = torch.where(flowRegions == True, tau, 1.0e30)

    return u, v, pressure, tau
