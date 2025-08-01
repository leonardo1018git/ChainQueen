import torch


def init_flow_params(x_nums, y_nums, re_init, inlet_velocities, inlet_pressure, device):
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

    u0 = u.clone()
    v0 = v.clone()

    p_prime = torch.zeros_like(p, dtype=torch.float32, device=device)                                   # 压力修正值
    u_e = torch.zeros((x_nums + 3, y_nums + 4), dtype=torch.float32, device=device)                     # 右界面流速
    v_n = torch.zeros((x_nums + 4, y_nums + 3), dtype=torch.float32, device=device)                     # 上界面流速
    re = torch.where((flow_regions == 1) | (flow_regions == 2) | (flow_regions == 4), re_init, 1.0e-30) # 雷诺数

    a_p = torch.full((x_nums + 4, y_nums + 4), 1.0e-30, dtype=torch.float32, device=device)

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    p_prime_numpy = p_prime.cpu().numpy()

    re_numpy = re.cpu()

    return u, v, p, u0, v0, p_prime, u_e, v_n, re, a_p, flow_regions
