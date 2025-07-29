import torch


def simple_solver(p, p_prime, u_e, v_n, a_p, flow_regions, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity, inner_epochs, uv_alpha, p_alpha):
    u_e_numpy = u_e.cpu().numpy()
    v_n_numpy = v_n.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    a_pe = torch.where(flow_regions[3: x_nums + 3, 2: y_nums + 2].eq(0)
                       | flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 1.0e-30,
                       uv_alpha * (delta_y ** 2) * (1.0 / a_p[3: x_nums + 3, 2: y_nums + 2]
                                                    + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2])
                       / (2 * density * inlet_velocity ** 2))

    a_pw = torch.where(flow_regions[1: x_nums + 1, 2: y_nums + 2].eq(0)
                       | flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 1.0e-30,
                       uv_alpha * (delta_y ** 2) * (1.0 / a_p[1: x_nums + 1, 2: y_nums + 2]
                                                    + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2])
                       / (2 * density * inlet_velocity ** 2))

    a_pn = torch.where(flow_regions[2: x_nums + 2, 3: y_nums + 3].eq(0)
                       | flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 1.0e-30,
                       uv_alpha * (delta_x ** 2) * (1.0 / a_p[2: x_nums + 2, 3: y_nums + 3]
                                                    + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2])
                       / (2 * density * inlet_velocity ** 2))

    a_ps = torch.where(flow_regions[2: x_nums + 2, 1: y_nums + 1].eq(0)
                       | flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 1.0e-30,
                       uv_alpha * (delta_x ** 2) * (1.0 / a_p[2: x_nums + 2, 1: y_nums + 1]
                                                    + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 2])
                       / (2 * density * inlet_velocity ** 2))

    a_pp = a_pe + a_pw + a_pn + a_ps

    a_p_numpy = a_p.cpu().numpy()
    a_pe_numpy = a_pe.cpu().numpy()
    a_pw_numpy = a_pw.cpu().numpy()
    a_pn_numpy = a_pn.cpu().numpy()
    a_ps_numpy = a_ps.cpu().numpy()
    a_pp_numpy = a_pp.cpu().numpy()

    p_error_old = 0.0
    for epoch in range(inner_epochs):
        p_old = p_prime.clone()
        p_prime[2: x_nums + 2, 2: y_nums + 2] \
            = ((a_pe * p_prime[3: x_nums + 3, 2: y_nums + 2] + a_pw * p_prime[1: x_nums + 1, 2: y_nums + 2]
                + a_pn * p_prime[2: x_nums + 2, 3: y_nums + 3] + a_ps * p_prime[2: x_nums + 2, 1: y_nums + 1]
                + (u_e[1: x_nums + 1, 2: y_nums + 2] - u_e[2: x_nums + 2, 2: y_nums + 2]) * delta_y
                + (v_n[2: x_nums + 2, 1: y_nums + 1] - v_n[2: x_nums + 2, 2: y_nums + 2]) * delta_x) / a_pp)

        p_prime[2: x_nums + 2, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 0.0,
                                                            p_prime[2: x_nums + 2, 2: y_nums + 2])

        error = (p_old[2: x_nums + 2, 2: y_nums + 2] - p_prime[2: x_nums + 2, 2: y_nums + 2]) * 100 / p_prime[2: x_nums + 2, 2: y_nums + 2]
        p_error = torch.max(torch.abs(torch.where(torch.isnan(error), 0.0, error))).cpu().numpy().item()
        # print("epoch = " + str(epoch) + ", pError = " + str(p_error.cpu().numpy()))
        if p_error < 1.0e-6:
            break

        p_error_error = abs(p_error_old - p_error) * 100 / max(p_error, 1.0e-30)
        if p_error_error < 1.0e-6:
            break
        p_error_old = p_error

    p_prime_numpy = p_prime.cpu().numpy()

    # 修正压力分布
    p[2: x_nums + 2, 2: y_nums + 2] += p_alpha * p_prime[2: x_nums + 2, 2: y_nums + 2]
    p[2: x_nums + 2, -1], p[2: x_nums + 2, -2] = p[2: x_nums + 2, -3], p[2: x_nums + 2, -3]

    return p, p_prime