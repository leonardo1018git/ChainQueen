import torch


def update_interface_velocities(u, v, p_prime, a_p, x_nums, y_nums, delta_x, delta_y, density, inlet_velocity):
    a_p_numpy= a_p.cpu().numpy()
    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    p_prime_numpy = p_prime.cpu().numpy()

    u_ep = (u[1: x_nums + 2, 2: y_nums + 2] + u[2: x_nums + 3, 2: y_nums + 2]) / 2.0
    u_ep_numpy = u_ep.cpu().numpy()

    p_eep = (p_prime[3: x_nums + 4, 2: y_nums + 2] - p_prime[1: x_nums + 2, 2: y_nums + 2]) / (2.0 * a_p[2: x_nums + 3, 2: y_nums + 2])
    p_eep_numpy = p_eep.cpu().numpy()

    p_ew = (p_prime[2: x_nums + 3, 2: y_nums + 2] - p_prime[: x_nums + 1, 2: y_nums + 2]) / (2.0 * a_p[1: x_nums + 2, 2: y_nums + 2])
    p_ew_numpy = p_ew.cpu().numpy()

    a_pe = (1.0 / a_p[1: x_nums + 2, 2: y_nums + 2] + 1.0 / a_p[2: x_nums + 3, 2: y_nums + 2]) * (p_prime[1: x_nums + 2, 2: y_nums + 2] - p_prime[2: x_nums + 3, 2: y_nums + 2])
    a_pe_numpy = a_pe.cpu().numpy()

    u_e_s = u_ep + (p_eep + p_ew + a_pe) * delta_y / (2 * density * inlet_velocity ** 2)
    u_e_s_numpy = u_e_s.cpu().numpy()

    u_e = ((u[1: x_nums + 2, 2: y_nums + 2] + u[2: x_nums + 3, 2: y_nums + 2]) / 2.0
           + ((p_prime[3: x_nums + 4, 2: y_nums + 2] - p_prime[1: x_nums + 2, 2: y_nums + 2])
              / (2.0 * a_p[2: x_nums + 3, 2: y_nums + 2])
              + (p_prime[2: x_nums + 3, 2: y_nums + 2] - p_prime[: x_nums + 1, 2: y_nums + 2])
              / (2.0 * a_p[1: x_nums + 2, 2: y_nums + 2])
              + (1.0 / a_p[1: x_nums + 2, 2: y_nums + 2] + 1.0 / a_p[2: x_nums + 3, 2: y_nums + 2])
              * (p_prime[1: x_nums + 2, 2: y_nums + 2] - p_prime[2: x_nums + 3, 2: y_nums + 2]))
           * delta_y / (2 * density * inlet_velocity ** 2))

    # v_np = (v[2: x_nums + 2, 1: y_nums + 2] + v[2: x_nums + 2, 2: y_nums + 3]) / 2.0
    # v_np_numpy = v_np.cpu().numpy()
    #
    # p_nnp = (p_prime[2: x_nums + 2, 3: y_nums + 4] - p_prime[2: x_nums + 2, 1: y_nums + 2]) / (2.0 * a_p[2: x_nums + 2, 2: y_nums + 3])
    # p_nnp_numpy = p_nnp.cpu().numpy()
    #
    # p_ns = (p_prime[2: x_nums + 2, 2: y_nums + 3] - p_prime[2: x_nums + 2, : y_nums + 1]) / (2.0 * a_p[2: x_nums + 2, 1: y_nums + 2])
    # p_ns_numpy = p_ns.cpu().numpy()
    #
    # a_pn = (1.0 / a_p[2: x_nums + 2, 1: y_nums + 2] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 3]) * (p_prime[2: x_nums + 2, 1: y_nums + 2] - p_prime[2: x_nums + 2, 2: y_nums + 3])
    # a_pn_numpy = a_pn.cpu().numpy()
    #
    # a = (p_nnp + p_ns + a_pn) * delta_x / (2.0 * density * inlet_velocity ** 2)
    # a_numpy = a.cpu().numpy()
    # v_n = v_np + (p_nnp + p_ns + a_pn) * delta_x / (2.0 * density * inlet_velocity ** 2)

    v_n = ((v[2: x_nums + 2, 1: y_nums + 2] + v[2: x_nums + 2, 2: y_nums + 3]) / 2.0
           + ((p_prime[2: x_nums + 2, 3: y_nums + 4] - p_prime[2: x_nums + 2, 1: y_nums + 2])
              / (2.0 * a_p[2: x_nums + 2, 2: y_nums + 3])
              + (p_prime[2: x_nums + 2, 2: y_nums + 3] - p_prime[2: x_nums + 2, : y_nums + 1])
              / (2.0 * a_p[2: x_nums + 2, 1: y_nums + 2])
              + (1.0 / a_p[2: x_nums + 2, 1: y_nums + 2] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 3])
              * (p_prime[2: x_nums + 2, 1: y_nums + 2] - p_prime[2: x_nums + 2, 2: y_nums + 3]))
           * delta_x / (2.0 * density * inlet_velocity ** 2))

    u_e_numpy = u_e.cpu().numpy()
    v_n_numpy = v_n.cpu().numpy()

    return u_e, v_n
