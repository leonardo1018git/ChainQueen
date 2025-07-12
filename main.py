import torch
from source.init import initFlowParams
from source.fvmSolver import fvmSolver


if __name__ == '__main__':
    xLength, yLength = 1.0, 1.0
    xNums, yNums = 5, 5
    density, viscosity = 1.0, 1.0
    scheme = "Quick"

    inletVelocities = [1.0, 0.0]
    inletVelocity = (inletVelocities[0] ** 2 +  inletVelocities[1] ** 2) ** 0.5
    re = density * inletVelocity * xLength / viscosity

    xDelta, yDelta = xLength / xNums, yLength / yNums  # x轴和y轴方向的单元格长度
    eta = xLength * re ** (-0.75)                      # Kolmogorov长度

    if eta > xDelta and eta > yDelta:
        innerEpochs, outerEpochs = 1000, 1000
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inletPressure = 0.0
        u, v, pressure, u_e, v_n, tau = initFlowParams(xNums, yNums, density, viscosity, inletVelocities, inletPressure, device)
        u, v, pressure = fvmSolver(scheme, u, v, pressure, u_e, v_n, tau, xNums, yNums, xDelta, yDelta, density, inletVelocity, innerEpochs, outerEpochs, device)
    else:
        print("单元格长度小于Kolmogorov长度，不能进行DNS计算！！！")

    print("done...")
