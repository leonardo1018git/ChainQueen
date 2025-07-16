import torch
from source.init import init_flow_params
from source.fvm_solver import fvm_solver


if __name__ == '__main__':
    x_length, y_length = 1.0, 1.0
    x_nums, y_nums = 5, 5
    density, viscosity = 1.0, 1.0
    scheme_flag = 4
    alpha = 0.2

    scheme = None
    if scheme_flag == 0:
        scheme = "QUICK"
    elif scheme_flag == 1:
        scheme = "SUD"
    elif scheme_flag == 2:
        scheme = "CD"
    elif scheme_flag == 3:
        scheme = "FUD"
    elif scheme_flag == 4:
        scheme = "Hybrid"
    else:
        print("不支持的数值格式！！！")

    inlet_velocities = [1.0, 0.5]
    inlet_velocity = (inlet_velocities[0] ** 2 +  inlet_velocities[1] ** 2) ** 0.5
    re = density * inlet_velocity * x_length / viscosity

    delta_x, delta_y = x_length / x_nums, y_length / y_nums  # x轴和y轴方向的单元格长度
    eta = x_length * re ** (-0.75)                      # Kolmogorov长度

    if eta > delta_x and eta > delta_y:
        inner_epochs, outer_epochs = 1000, 1000
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inlet_pressure = 0.0
        u, v, pressure, pressure_prime, u_e, v_n, tau, a_p = init_flow_params(x_nums, y_nums, density, viscosity, inlet_velocities, inlet_pressure, device)
        u, v, pressure = fvm_solver(scheme, u, v, pressure, pressure_prime, u_e, v_n, tau, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, alpha, inner_epochs, outer_epochs, device)
    else:
        print("单元格长度小于Kolmogorov长度，不能进行DNS计算！！！")

    print("done...")
