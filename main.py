import torch
from source.mesh import initFlowParams
from source.fvmSolver import fvmSolver


if __name__ == '__main__':
    xLength, yLength = 1.0, 1.0
    xGridNums, yGridNums = 5, 5
    density, viscosity = 1.0, 1.0

    inletVelocities = [1.0, 1.0]
    re = density * (inletVelocities[0] ** 2 + inletVelocities[1] ** 2) ** 0.5 * xLength / viscosity

    xDelta, yDelta = xLength / xGridNums, yLength / yGridNums  # x轴和y轴方向的单元格长度
    eta = xLength * re ** (-0.75)                              # Kolmogorov长度

    if eta > xDelta and eta > yDelta:
        innerEpochs, outerEpochs = 1000, 1000
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inletPressure = 1.0
        u, v, pressure, tau = initFlowParams(xGridNums, yGridNums, density, viscosity, inletVelocities, inletPressure)

        flowDistribution = fvmSolver(u, v, pressure, tau, xDelta, yDelta)
    else:
        print("单元格长度小于Kolmogorov长度，不能进行DNS计算！！！")

    print("done...")
