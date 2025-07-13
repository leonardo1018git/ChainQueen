import torch


def init_flow_params(x_nums, y_nums, density, viscosity, inlet_velocities, inlet_pressure, device):
    # 流动区域
    flow_regions = torch.tensor([
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

    u = torch.where(flow_regions == True, inlet_velocities[0], 0.0)
    v = torch.where(flow_regions == True, inlet_velocities[1], 0.0)
    pressure = torch.where(flow_regions == True, inlet_pressure, 0.0)

    u_e = torch.zeros((x_nums + 1, y_nums), dtype=torch.float32, device=device)
    v_n = torch.zeros((x_nums, y_nums + 1), dtype=torch.float32, device=device)

    tau_init = viscosity / (density * (inlet_velocities[0] ** 2 + inlet_velocities[1] ** 2) ** 0.5)
    tau = torch.where(flow_regions == True, tau_init, 1.0e30)

    return u, v, pressure, u_e, v_n, tau
