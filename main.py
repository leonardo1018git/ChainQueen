import torch
from source.init import init_flow_params
from source.fvm_solver import fvm_solver


if __name__ == '__main__':
    x_length, y_length = 1.0, 1.0
    x_nums, y_nums = 5, 5
    density, viscosity = 1.0, 1.0
    scheme_dict = {0: "QUICK", 1: "SUD", 2: "CD", 3: "FUD", 4: "Hybrid"}
    scheme = scheme_dict[2]
    alpha = 0.001

    inlet_velocities = [1.0, 1.0]
    inlet_velocity = (inlet_velocities[0] ** 2 +  inlet_velocities[1] ** 2) ** 0.5
    re = density * inlet_velocity * x_length / viscosity

    delta_x, delta_y = x_length / x_nums, y_length / y_nums  # x轴和y轴方向的单元格长度
    eta = x_length * re ** (-0.75)                           # Kolmogorov长度

    if eta > delta_x and eta > delta_y:
        inner_epochs, outer_epochs = 1000, 1000
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        u, v, p_prime, u_e, v_n, tau, a_p = init_flow_params(x_nums, y_nums, density, viscosity, inlet_velocities, device)
        u, v, p_prime = fvm_solver(scheme, u, v, p_prime, u_e, v_n, tau, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inner_epochs, outer_epochs, alpha)
    else:
        print("单元格长度小于Kolmogorov长度，不能进行DNS计算！！！")

    print("done...")
