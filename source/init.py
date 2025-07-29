import torch


def init_flow_params(x_nums, y_nums, density, viscosity, inlet_velocities, inlet_pressure, device):
    # 流动区域
    flow_regions = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [2, 2, 1, 1, 0, 1, 1, 4, 4],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [2, 2, 1, 1, 1, 1, 1, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.int16, device=device)

    u = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), inlet_velocities[0], 0.0)
    v = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), inlet_velocities[1], 0.0)
    p = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), inlet_pressure, 0.0)

    p_prime = torch.zeros_like(p, dtype=torch.float32, device=device)                    # 压力修正值
    u_e = torch.zeros((x_nums + 3, y_nums + 4), dtype=torch.float32, device=device)                 # 右界面流速
    v_n = torch.zeros((x_nums + 4, y_nums + 3), dtype=torch.float32, device=device)                 # 上界面流速

    tau_init = viscosity / (density * (inlet_velocities[0] ** 2 + inlet_velocities[1] ** 2) ** 0.5)
    tau = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), tau_init, 1.0e30)

    a_p = torch.full((x_nums + 4, y_nums + 4), 1.0e-30, dtype=torch.float32, device=device)

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    p_prime_numpy = p_prime.cpu().numpy()

    tau_numpy = tau.cpu()
    #
    # p_prime = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), 0.5, 0.0)

    return u, v, p, p_prime, u_e, v_n, tau, a_p, flow_regions
