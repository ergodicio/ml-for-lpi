from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import os


def calc_tpd_threshold_intensity(Te: float, Ln: float, w0: float) -> float:
    """
    Calculate the TPD threshold intensity

    :param Te:
    :return: intensity
    """

    c = 2.99792458e10
    me_keV = 510.998946  # keV/c^2
    me_cgs = 9.10938291e-28
    e = 4.8032068e-10

    vte = np.sqrt(Te / me_keV) * c
    I_threshold = 4 * 4.134 * 1 / (8 * np.pi) * (me_cgs * c / e) ** 2 * w0 * vte**2 / (Ln / 100) * 1e-7

    return I_threshold


def calc_coherence(lpse_module, used_driver, density):
    def _calc_e0_(t, y, light_wave):
        return lpse_module.diffeqsolve_quants["terms"].vector_field.light.calc_ey_at_one_point(t, y, light_wave)

    calc_e0 = vmap(_calc_e0_, in_axes=(0, None, None))
    t0 = 2 * np.pi / (lpse_module.diffeqsolve_quants["terms"].vector_field.light.w0)

    Ntau = 128

    tau = np.linspace(-150 * t0, 150 * t0, Ntau)
    ey = np.zeros((Ntau, 2), dtype=np.complex64)

    for it, ttau in enumerate(tau):
        tt = np.random.uniform(0, 1e3, int(1e5))
        e0_tt = calc_e0(tt, density, used_driver["E0"])
        e0_tt_tau = calc_e0(tt + ttau, density, used_driver["E0"])
        ey[it, 0] = np.mean(np.abs(e0_tt) ** 2.0)
        ey[it, 1] = np.mean(e0_tt_tau * np.conjugate(e0_tt))

    gtau = ey[:, 1] / ey[:, 0]

    return tau, gtau


def plot_coherence(lpse_module, used_driver, td, density):
    tau, gtau = calc_coherence(lpse_module, used_driver, density)
    tau_0 = 2 * np.pi / (lpse_module.diffeqsolve_quants["terms"].vector_field.light.w0)

    metrics = {"tau_cf": np.trapz(np.abs(gtau) ** 2.0, tau)}

    integrand = np.abs(gtau) ** 2.0
    half_integrand = integrand[len(integrand) // 2 :]
    decreasing_order_args = np.argsort(half_integrand)[::-1]

    for i in range(1, 4):
        slc = slice(len(integrand) // 2 - decreasing_order_args[-i], len(integrand) // 2 + decreasing_order_args[-i])
        _tau, int = tau[slc], integrand[slc]
        metrics[f"tau_c{str(i)}"] = np.trapz(int, _tau)
        metrics[f"bound_{str(i)}"] = _tau[-1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    ax.plot(tau / tau_0, np.abs(gtau))
    ax.set_xlabel(r"Time delay (in units of $\tau_0$)")
    ax.set_ylabel("Coherence")
    ax.grid()
    ax.set_title(rf"$\tau_c = {round(metrics['tau_cf'] * 1000, 2)}$ fs")
    fig.savefig(os.path.join(td, "driver", "coherence.png"), bbox_inches="tight")

    plt.close()

    return metrics


def plot_bandwidth(e0, td):
    dw_over_w = e0["delta_omega"]  # / cfg["units"]["derived"]["w0"] - 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 5), tight_layout=True)
    ax[0].plot(dw_over_w, e0["intensities"], "o")
    ax[0].grid()
    ax[0].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[0].set_ylabel("$|E|$", fontsize=14)
    ax[1].semilogy(dw_over_w, e0["intensities"], "o")
    ax[1].grid()
    ax[1].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[1].set_ylabel("$|E|$", fontsize=14)
    ax[2].plot(dw_over_w, e0["phases"], "o")
    ax[2].grid()
    ax[2].set_xlabel(r"$\Delta \omega / \omega_0$", fontsize=14)
    ax[2].set_ylabel(r"$\angle E$", fontsize=14)
    plt.savefig(os.path.join(td, "driver", "driver_that_was_used.png"), bbox_inches="tight")
    plt.close()


def postprocess_bandwidth(used_driver, lpse_module, td, density):
    import pickle

    os.makedirs(os.path.join(td, "driver"), exist_ok=True)
    with open(os.path.join(td, "driver", "used_driver.pkl"), "wb") as fi:
        pickle.dump(used_driver, fi)

    plot_bandwidth(used_driver["E0"], td)
    return plot_coherence(lpse_module, used_driver, td, density)
