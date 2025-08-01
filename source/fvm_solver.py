import torch
from source.scheme.quick_scheme import quick
from source.scheme.sud_scheme import second_upwind_difference
from source.scheme.cd_scheme import central_difference
from source.scheme.fud_scheme import first_upwind_difference
from source.scheme.hybrid_scheme import hybrid
from source.decoupled.simple_simplec import simple_solver


# 传热与流体流动的数值计算228页，4.5小节稳态问题同位网格的SIMPLE算法
def velocity_solver(scheme, decoupled, u, v, p, u0, v0, u_e, v_n, re, a_p, flow_regions, hydraulic_diameter, x_nums, y_nums, delta, delta_time, density, inlet_velocity, inner_epochs, uv_alpha):
    a_ww, a_w, a_e, a_ee, a_ss, a_s, a_n, a_nn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    re_numpy = re.cpu().numpy()
    u_e_numpy, v_n_numpy = u_e.cpu().numpy(), v_n.cpu().numpy()

    d_e = 2.0 / (re[2 : x_nums + 2, 2: y_nums + 2] + re[3: x_nums + 3, 2: y_nums + 2])
    d_w = 2.0 / (re[2 : x_nums + 2, 2: y_nums + 2] + re[1: x_nums + 1, 2: y_nums + 2])
    d_n = 2.0 / (re[2 : x_nums + 2, 2: y_nums + 2] + re[2: x_nums + 2, 3: y_nums + 3])
    d_s = 2.0 / (re[2 : x_nums + 2, 2: y_nums + 2] + re[2: x_nums + 2, 1: y_nums + 1])

    d_e_numpy = d_e.cpu().numpy()
    d_w_numpy = d_w.cpu().numpy()
    d_n_numpy = d_n.cpu().numpy()
    d_s_numpy = d_s.cpu().numpy()

    # f_e = u_e[2: x_nums + 2, 2: y_nums + 2] * delta
    # f_w = u_e[1: x_nums + 1, 2: y_nums + 2] * delta
    # f_n = v_n[2: x_nums + 2, 2: y_nums + 2] * delta
    # f_s = v_n[2: x_nums + 2, 1: y_nums + 1] * delta
    #
    # f_e_numpy = f_e.cpu().numpy()
    # f_w_numpy = f_w.cpu().numpy()
    # f_n_numpy = f_n.cpu().numpy()
    # f_s_numpy = f_s.cpu().numpy()

    p_numpy = p.cpu().numpy()

    u_source = (p[1: x_nums + 1, 2: y_nums + 2] - p[3: x_nums + 3, 2: y_nums + 2]) * delta / (2.0 * density * inlet_velocity ** 2) + u0[2: x_nums + 2, 2: y_nums + 2] * hydraulic_diameter * delta ** 2 / (inlet_velocity * delta_time)
    v_source = (p[2: x_nums + 2, 1: y_nums + 1] - p[2: x_nums + 2, 3: y_nums + 3]) * delta / (2.0 * density * inlet_velocity ** 2) + v0[2: x_nums + 2, 2: y_nums + 2] * hydraulic_diameter * delta ** 2 / (inlet_velocity * delta_time)

    u_source_numpy = u_source.cpu().numpy()
    v_source_numpy = v_source.cpu().numpy()

    # a_p_quick, _, _, _, _, _, _, _, _ = quick(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    # a_p_sud, _, _, _, _, _, _, _, _ = second_upwind_difference(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    # a_p_cd, _, _, _, _ = central_difference(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    # a_p_fud, _, _, _, _ = first_upwind_difference(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    # a_p_hyb, _, _, _, _ = hybrid(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)

    # a_p_quick_numpy = a_p_quick.cpu().numpy()
    # a_p_sud_numpy = a_p_sud.cpu().numpy()
    # a_p_cd_numpy = a_p_cd.cpu().numpy()
    # a_p_fud_numpy = a_p_fud.cpu().numpy()
    # a_p_hyb_numpy = a_p_hyb.cpu().numpy()

    if scheme == "QUICK":
        # Quick格式
        a_p, a_e, a_ee, a_w, a_ww, a_n, a_nn, a_s, a_ss = quick(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    elif scheme == "SUD":
        # 二阶迎风格式
        a_p, a_e, a_ee, a_w, a_ww, a_n, a_nn, a_s, a_ss = second_upwind_difference(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    elif scheme == "CD":
        # 中心差分格式
        a_p, a_e, a_w, a_n, a_s = central_difference(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    elif scheme == "FUD":
        # 一阶迎风格式
        a_p, a_e, a_w, a_n, a_s = first_upwind_difference(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)
    elif scheme == "Hybrid":
        # 混合格式
        a_p, a_e, a_w, a_n, a_s = hybrid(a_p, flow_regions, u_e, v_n, d_e, d_w, d_n, d_s, hydraulic_diameter, x_nums, y_nums, delta, delta_time, inlet_velocity)

    a_p[2: x_nums + 2, -1], a_p[2: x_nums + 2, -2] = a_p[2: x_nums + 2, -3], a_p[2: x_nums + 2, -3]
    a_p[2: x_nums + 2, 0], a_p[2: x_nums + 2, 1] = a_p[2: x_nums + 2, 2], a_p[2: x_nums + 2, 2]

    a_e_numpy = a_e.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()

    u_error_old, v_error_old = 0.0, 0.0
    for epoch in range(inner_epochs):
        u_old, v_old = u.clone(), v.clone()
        # ae_ue = a_e * u[3: x_nums + 3, 2: y_nums + 2] + a_ee * u[4: x_nums + 4, 2: y_nums + 2]
        # aw_uw = a_w * u[1: x_nums + 1, 2: y_nums + 2] + a_ww * u[: x_nums, 2: y_nums + 2]
        # an_un = a_n * u[2: x_nums + 2, 3: y_nums + 3] + a_nn * u[2: x_nums + 2, 4: y_nums + 4]
        # as_us = a_s * u[2: x_nums + 2, 1: y_nums + 1] + a_ss * u[2: x_nums + 2, : y_nums]
        #
        # ae_ue_numpy = ae_ue.cpu().numpy()
        # aw_uw_numpy = aw_uw.cpu().numpy()
        # an_un_numpy = an_un.cpu().numpy()
        # as_us_numpy = as_us.cpu().numpy()
        #
        # sum_an = ae_ue_numpy + aw_uw_numpy + an_un_numpy + as_us_numpy
        # an_un = sum_an + u_source_numpy
        # an_vn = sum_an + v_source_numpy

        # 数值传热学214页，6.5.1.5小节，速度与压力修正值的亚松弛
        u[2: x_nums + 2, 2: y_nums + 2] = (uv_alpha * (a_e * u[3: x_nums + 3, 2: y_nums + 2] + a_ee * u[4: x_nums + 4, 2: y_nums + 2]
                                                      + a_w * u[1: x_nums + 1, 2: y_nums + 2] + a_ww * u[: x_nums, 2: y_nums + 2]
                                                      + a_n * u[2: x_nums + 2, 3: y_nums + 3] + a_nn * u[2: x_nums + 2, 4: y_nums + 4]
                                                      + a_s * u[2: x_nums + 2, 1: y_nums + 1] + a_ss * u[2: x_nums + 2, : y_nums]
                                                      + u_source) / a_p[2: x_nums + 2, 2: y_nums + 2]
                                           + (1 - uv_alpha) * u[2: x_nums + 2, 2: y_nums + 2])
        v[2: x_nums + 2, 2: y_nums + 2] = (uv_alpha * (a_e * v[3: x_nums + 3, 2: y_nums + 2] + a_ee * v[4: x_nums + 4, 2: y_nums + 2]
                                                      + a_w * v[1: x_nums + 1, 2: y_nums + 2] + a_ww * v[: x_nums, 2: y_nums + 2]
                                                      + a_n * v[2: x_nums + 2, 3: y_nums + 3] + a_nn * v[2: x_nums + 2, 4: y_nums + 4]
                                                      + a_s * v[2: x_nums + 2, 1: y_nums + 1] + a_ss * v[2: x_nums + 2, : y_nums]
                                                      + v_source) / a_p[2: x_nums + 2, 2: y_nums + 2]
                                           + (1 - uv_alpha) * v[2: x_nums + 2, 2: y_nums + 2])

        u[2: x_nums + 2, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 0.0, u[2: x_nums + 2, 2: y_nums + 2])
        v[2: x_nums + 2, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 0.0, v[2: x_nums + 2, 2: y_nums + 2])

        u_numpy = u.cpu().numpy()
        v_numpy = v.cpu().numpy()
        u_error = torch.max(torch.abs(u_old - u) * 100 / inlet_velocity).cpu().numpy().item()
        v_error = torch.max(torch.abs(v_old - v) * 100 / inlet_velocity).cpu().numpy().item()
        # print("epoch = " + str(epoch) + ", uError = " + str(u_error) + ", vError = " + str(v_error))
        if u_error < 1.0e-6 and v_error < 1.0e-6:
            break

        u_error_error = abs(u_error_old - u_error) * 100 / max(u_error, 1.0e-30)
        v_error_error = abs(v_error_old - v_error) * 100 / max(v_error, 1.0e-30)
        if u_error_error < 1.0e-6 and v_error_error < 1.0e-6:
            break
        u_error_old, v_error_old = u_error, v_error

    u[2: x_nums + 2, -1], u[2: x_nums + 2, -2] = u[2: x_nums + 2, -3],u[2: x_nums + 2, -3]
    # u[0, :], u[1, :] = u[2, :], u[2, :]
    # u[-1, :], u[-2, :] = u[-3, :], u[-3, :]

    v[2: x_nums + 2, -1], v[2: x_nums + 2, -2] = v[2: x_nums + 2, -3], v[2: x_nums + 2, -3]
    # v[0, :], v[1, :] = v[2, :], v[2, :]
    # v[-1, :], v[-2, :] = v[-3, :], v[-3, :]

    d_e_numpy = d_e.cpu().numpy()
    d_w_numpy = d_w.cpu().numpy()
    d_n_numpy = d_n.cpu().numpy()
    d_s_numpy = d_s.cpu().numpy()

    # a_ee_numpy = a_ee.cpu().numpy()
    # a_ww_numpy = a_ww.cpu().numpy()
    # a_nn_numpy = a_nn.cpu().numpy()
    # a_ss_numpy = a_ss.cpu().numpy()

    a_e_numpy = a_e.cpu().numpy()
    a_w_numpy = a_w.cpu().numpy()
    a_n_numpy = a_n.cpu().numpy()
    a_s_numpy = a_s.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    p_numpy = p.cpu().numpy()

    # u_ep = (u[2: x_nums + 1, 2: y_nums + 2] + u[3: x_nums + 2, 2: y_nums + 2]) / 2.0
    # u_ep_numpy = u_ep.cpu().numpy()
    #
    # p_ew = (p[3: x_nums + 2, 2: y_nums + 2] - p[1: x_nums, 2: y_nums + 2]) / (4.0 * a_p[2: x_nums + 1, 2: y_nums + 2])
    # p_ew_numpy = p_ew.cpu().numpy()
    #
    # p_eep = (p[4: x_nums + 3, 2: y_nums + 2] - p[2: x_nums + 1, 2: y_nums + 2]) / (4.0 * a_p[3: x_nums + 2, 2: y_nums + 2])
    # p_eep_numpy = p_eep.cpu().numpy()
    #
    # a_pe = 0.5 * (1.0 / a_p[2: x_nums + 1, 2: y_nums + 2] + 1.0 / a_p[3: x_nums + 2, 2: y_nums + 2]) * (p[2: x_nums + 1, 2: y_nums + 2] - p[3: x_nums + 2, 2: y_nums + 2])
    # a_pe_numpy = a_pe.cpu().numpy()
    #
    # a = (p_ew + p_eep + a_pe) * delta / (density * inlet_velocity ** 2)
    # a_numpy = a.cpu().numpy()
    # u_e_s = u_ep + uv_alpha * (p_ew + p_eep + a_pe) * delta / (density * inlet_velocity ** 2)
    # u_e_s_numpy = u_e_s.cpu().numpy()

    u_e[2: x_nums + 1, 2: y_nums + 2] = ((u[2: x_nums + 1, 2: y_nums + 2] + u[3: x_nums + 2, 2: y_nums + 2]) / 2.0
                                         + uv_alpha * ((p[3: x_nums + 2, 2: y_nums + 2] - p[1: x_nums, 2: y_nums + 2])
                                                       / (4.0 * a_p[2: x_nums + 1, 2: y_nums + 2])
                                                       + (p[4: x_nums + 3, 2: y_nums + 2]
                                                          - p[2: x_nums + 1, 2: y_nums + 2])
                                                       / (4.0 * a_p[3: x_nums + 2, 2: y_nums + 2])
                                                       + 0.5 * (1.0 / a_p[2: x_nums + 1, 2: y_nums + 2]
                                                                + 1.0 / a_p[3: x_nums + 2, 2: y_nums + 2])
                                                       * (p[2: x_nums + 1, 2: y_nums + 2]
                                                          - p[3: x_nums + 2, 2: y_nums + 2])) * delta
                                         / (density * inlet_velocity ** 2))

    u_e[2: x_nums + 1, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 1, 2: y_nums + 2].eq(0)
                                                    | flow_regions[3: y_nums + 2, 2: y_nums + 2].eq(0), 0.0,
                                                    u_e[2: x_nums + 1, 2: y_nums + 2])
    u_e_numpy = u_e.cpu().numpy()

    # v_np = (v[2: x_nums + 2, 1: y_nums + 2] + v[2: x_nums + 2, 2: y_nums + 3]) / 2.0
    # v_np_numpy = v_np.cpu().numpy()
    #
    # p_ns = (p[2: x_nums + 2, 2: y_nums + 3] - p[2: x_nums + 2, 0: y_nums + 1]) / (4.0 * a_p[2: x_nums + 2, 1: y_nums + 2])
    # p_ns_numpy = p_ns.cpu().numpy()
    #
    # p_nnp = (p[2: x_nums + 2, 3: y_nums + 4] - p[2: x_nums + 2, 1: y_nums + 2]) / (4.0 * a_p[2: x_nums + 2, 2: y_nums + 3])
    # p_nnp_numpy = p_nnp.cpu().numpy()
    #
    # a_pn = 0.5 * (1.0 / a_p[2: x_nums + 2, 1: y_nums + 2] + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 3]) * (p[2: x_nums + 2, 1: y_nums + 2] - p[2: x_nums + 2, 2: y_nums + 3])
    # a_pn_numpy = a_pn.cpu().numpy()
    #
    # a = uv_alpha * (p_ns + p_nnp + a_pn) * delta / (density * inlet_velocity ** 2)
    # a_numpy = a.cpu().numpy()
    #
    # v_n_s = v_np + uv_alpha * (p_ns + p_nnp + a_pn) * delta / (density * inlet_velocity ** 2)
    # v_n_s_numpy = v_n_s.cpu().numpy()

    v_n[2: x_nums + 2, 1: y_nums + 2] = ((v[2: x_nums + 2, 1: y_nums + 2] + v[2: x_nums + 2, 2: y_nums + 3]) / 2.0
                                         + uv_alpha * ((p[2: x_nums + 2, 2: y_nums + 3]
                                                        - p[2: x_nums + 2, : y_nums + 1])
                                                       / (4.0 * a_p[2: x_nums + 2, 1: y_nums + 2])
                                                       + (p[2: x_nums + 2, 3: y_nums + 4]
                                                          - p[2: x_nums + 2, 1: y_nums + 2])
                                                       / (4.0 * a_p[2: x_nums + 2, 2: y_nums + 3])
                                                       + 0.5 * (1.0 / a_p[2: x_nums + 2, 1: y_nums + 2]
                                                                + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 3])
                                                       * (p[2: x_nums + 2, 1: y_nums + 2]
                                                          - p[2: x_nums + 2, 2: y_nums + 3])) * delta
                                         / (density * inlet_velocity ** 2))

    v_n[2: x_nums + 2, 1: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 1: y_nums + 2].eq(0)
                                                    | flow_regions[2: x_nums + 2, 2: y_nums + 3].eq(0), 0.0,
                                                    v_n[2: x_nums + 2, 1: y_nums + 2])

    u_e_numpy = u_e.cpu().numpy()
    v_n_numpy = v_n.cpu().numpy()

    if decoupled == "SIMPLE_C":
        a_p[2: x_nums + 2, 2: y_nums + 2] = a_p[2: x_nums + 2, 2: y_nums + 2] / uv_alpha - (a_e + a_ee + a_w + a_ww + a_n + a_nn + a_s + a_ss)
        a_p[2: x_nums + 2, -1], a_p[2: x_nums + 2, -2] = a_p[2: x_nums + 2, -3], a_p[2: x_nums + 2, -3]
        a_p[2: x_nums + 2, 0], a_p[2: x_nums + 2, 1] = a_p[2: x_nums + 2, 2], a_p[2: x_nums + 2, 2]
    else:
        pass

    a_p_numpy = a_p.cpu().numpy()

    return u, v, u_e, v_n, a_p


def correct_velocities(u, v, u_e, v_n, p_prime, a_p, flow_regions, x_nums, y_nums, delta, uv_alpha, density, inlet_velocity):
    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()
    p_prime_numpy = p_prime.cpu().numpy()

    u[2: x_nums + 2, 2: y_nums + 2] += (uv_alpha * (p_prime[1: x_nums + 1, 2: y_nums + 2] - p_prime[3: x_nums + 3, 2: y_nums + 2]) * delta
                                        / (2.0 * density * inlet_velocity ** 2 * a_p[2: x_nums + 2, 2: y_nums + 2]))
    v[2: x_nums + 2, 2: y_nums + 2] += (uv_alpha * (p_prime[2: x_nums + 2, 1: y_nums + 1] - p_prime[2: x_nums + 2, 3: y_nums + 3]) * delta
                                        / (2.0 * density * inlet_velocity ** 2 * a_p[2: x_nums + 2, 2: y_nums + 2]))

    p_prime_numpy = p_prime.cpu().numpy()
    a_p_numpy = a_p.cpu().numpy()
    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()

    u[2: x_nums + 2, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 0.0,
                                                  u[2: x_nums + 2, 2: y_nums + 2])
    v[2: x_nums + 2, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 2: y_nums + 2].eq(0), 0.0,
                                                  v[2: x_nums + 2, 2: y_nums + 2])

    u_e[2: x_nums + 1, 2: y_nums + 2] += (uv_alpha * (1.0 / a_p[2: x_nums + 1, 2: y_nums + 2]
                                                      + 1.0 / a_p[3: x_nums + 2, 2: y_nums + 2])
                                          * (p_prime[2: x_nums + 1, 2: y_nums + 2]
                                             - p_prime[3: x_nums + 2, 2: y_nums + 2]) * delta
                                          / (2.0 * density * inlet_velocity ** 2))
    v_n[2: x_nums + 2, 1: y_nums + 2] += (uv_alpha * (1.0 / a_p[2: x_nums + 2, 1: y_nums + 2]
                                                      + 1.0 / a_p[2: x_nums + 2, 2: y_nums + 3])
                                          * (p_prime[2: x_nums + 2, 1: y_nums + 2]
                                             - p_prime[2: x_nums + 2, 2: y_nums + 3]) * delta
                                          / (2.0 * density * inlet_velocity ** 2))

    u_e[2: x_nums + 1, 2: y_nums + 2] = torch.where(flow_regions[2: x_nums + 1, 2: y_nums + 2].eq(0)
                                                    | flow_regions[3: y_nums + 2, 2: y_nums + 2].eq(0), 0.0,
                                                    u_e[2: x_nums + 1, 2: y_nums + 2])

    v_n[2: x_nums + 2, 1: y_nums + 2] = torch.where(flow_regions[2: x_nums + 2, 1: y_nums + 2].eq(0)
                                                    | flow_regions[2: x_nums + 2, 2: y_nums + 3].eq(0), 0.0,
                                                    v_n[2: x_nums + 2, 1: y_nums + 2])

    u_e_numpy = u_e.cpu().numpy()
    v_n_numpy = v_n.cpu().numpy()

    return u, v, u_e, v_n


def fvm_solver(scheme, decoupled, u, v, p, u0, v0, p_prime, u_e, v_n, re, a_p, flow_regions, hydraulic_diameter, x_nums, y_nums, delta, delta_time, density, inlet_velocity, inner_epochs, outer_epochs, uv_alpha, p_alpha):
    for epoch in range(outer_epochs):
        u_old, v_old, p_old = u.clone(), v.clone(), p.clone()
        u, v, u_e, v_n, a_p = velocity_solver(scheme, decoupled, u, v, p, u0, v0, u_e, v_n, re, a_p, flow_regions, hydraulic_diameter, x_nums, y_nums, delta, delta_time, density, inlet_velocity, inner_epochs, uv_alpha)
        u_numpy, v_numpy, a_p_numpy = u.cpu().numpy(), v.cpu().numpy(), a_p.cpu().numpy()
        u_e_numpy, v_n_numpy = u_e.cpu().numpy(), v_n.cpu().numpy()

        p, p_prime = simple_solver(p, p_prime, u_e, v_n, a_p, flow_regions, x_nums, y_nums, delta, density, inlet_velocity, inner_epochs, uv_alpha, p_alpha)
        p_numpy = p.cpu().numpy()
        p_prime_numpy = p_prime.cpu().numpy()

        u, v, u_e, v_n = correct_velocities(u, v, u_e, v_n, p_prime, a_p, flow_regions, x_nums, y_nums, delta, uv_alpha, density, inlet_velocity)
        u_numpy, v_numpy = u.cpu().numpy(), v.cpu().numpy()
        u_e_numpy, v_n_numpy = u_e.cpu().numpy(), v_n.cpu().numpy()

        u_error = (torch.max(torch.abs(u_old - u) * 100 / inlet_velocity).cpu().numpy()).item()
        v_error = (torch.max(torch.abs(v_old - v) * 100 / inlet_velocity).cpu().numpy()).item()
        p_error = (torch.max(torch.abs(p_old - p_prime)).cpu().numpy()).item()
        print("epoch = " + str(epoch) + ", uError = " + str(u_error) + ", vError = " + str(v_error) + ", pError = " + str(p_error))
        if u_error < 1.0e-6 and v_error < 1.0e-6:
            break

    u_numpy = u.cpu().numpy()
    v_numpy = v.cpu().numpy()
    p_numpy = p.cpu().numpy()
    p_prime_numpy = p_prime.cpu().numpy()

    return u, v, p
