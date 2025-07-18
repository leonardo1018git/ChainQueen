import torch


def init_flow_params(x_nums, y_nums, density, viscosity, inlet_velocities, inlet_pressure, device):
    # 流动区域
    flow_regions = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [2, 2, 1, 1, 9, 1, 1, 4, 4],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.int16, device=device)

    u = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), inlet_velocities[0], 0.0)
    v = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), inlet_velocities[1], 0.0)
    pressure = torch.where(flow_regions > 0, inlet_pressure, 0.0)

    u_e = torch.zeros((x_nums + 1, y_nums), dtype=torch.float32, device=device)
    v_n = torch.zeros((x_nums, y_nums + 1), dtype=torch.float32, device=device)

    tau_init = viscosity / (density * (inlet_velocities[0] ** 2 + inlet_velocities[1] ** 2) ** 0.5)
    tau = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), tau_init, 1.0e30)

    a_p = torch.zeros_like(u, dtype=torch.float32, device=device)

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()

    return u, v, pressure, u_e, v_n, tau, a_p
