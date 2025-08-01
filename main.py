import torch
from source.init import init_flow_params
from source.fvm_solver import fvm_solver


if __name__ == '__main__':
    re_init = 4.0
    density = 1.0e3

    operation_time = 1
    delta_time = 0.001

    x_length, y_length = 0.5, 0.5
    hydraulic_diameter = x_length       # 水力直径
    x_nums, y_nums = 5, 5
    inlet_velocities = [1.0, 1.0]
    inlet_pressure = 1.0e5
    scheme_flag = 0                     # 选择离散格式：0: "QUICK", 1: "SUD", 2: "CD", 3: "FUD", 4: "Hybrid"
    decoupled_flag = 1                  # 选择速度压力解耦算法：0: "SIMPLE", 1: "SIMPLE_C"

    scheme_dict = {0: "QUICK", 1: "SUD", 2: "CD", 3: "FUD", 4: "Hybrid"}
    decoupled_dict = {0: "SIMPLE", 1: "SIMPLE_C"}

    scheme = scheme_dict[scheme_flag]
    decoupled = decoupled_dict[decoupled_flag]

    inlet_velocity = (inlet_velocities[0] ** 2 + inlet_velocities[1] ** 2) ** 0.5

    uv_alpha, p_alpha = 0.7, 0.2
    if decoupled == "SIMPLE_C":
        p_alpha = 1.0
    else:
        pass

    delta_x, delta_y = x_length / x_nums, y_length / y_nums        # x轴和y轴方向的单元格长度
    if delta_x == delta_y:
        co = inlet_velocity * delta_time / delta_x
        if co < 1.0:
            delta = delta_y / x_length                                 # 单元格长度与水力直径的比值
            eta = x_length * re_init ** (-0.75)                        # Kolmogorov长度
            if eta > delta_x and eta > delta_y:
                inner_epochs, outer_epochs = 1000, 300
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                u, v, p, u0, v0, p_prime, u_e, v_n, re, a_p, flow_regions = init_flow_params(x_nums, y_nums, re_init, inlet_velocities, inlet_pressure, device)

                time_step = int(operation_time / delta_time)
                for epoch in range(time_step):
                    u, v, p = fvm_solver(scheme, decoupled, u, v, p, u0, v0, p_prime, u_e, v_n, re, a_p, flow_regions, hydraulic_diameter, x_nums, y_nums, delta, delta_time, density, inlet_velocity, inner_epochs, outer_epochs, uv_alpha, p_alpha)
                    u_numpy, v_numpy, p_numpy = u.cpu().numpy(), v.cpu().numpy(), p.cpu().numpy()

                    u0, v0 = u.clone(), v.clone()
            else:
                print("单元格长度小于Kolmogorov长度，不能进行DNS计算！！！")
        else:
            print("库伦数小于1，不能进行DNS计算！！！")
    else:
        print("x和y方向的单元格长度不相等，请重新设置！！！")

    print("done...")
