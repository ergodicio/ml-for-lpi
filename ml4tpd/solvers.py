from typing import Dict
import numpy as np
from astropy.units import Quantity as _Q

from jax import numpy as jnp
from equinox import combine, tree_at, filter_value_and_grad, filter_jit
from jax.debug import print as jax_print
import optimistix as optmx


from adept.lpse2d import BaseLPSE2D, ArbitraryDriver
from adept._lpse2d.modules import driver

from .modules.drivers import GenerativeDriver, TPDLearner, ZeroLiner
from .helpers import postprocess_bandwidth, calc_tpd_threshold_intensity, calc_tpd_broadband_threshold_intensity


class TPDModule(BaseLPSE2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def init_modules(self) -> Dict:
        self.metric_timesteps = np.argmin(
            np.abs(self.cfg["save"]["default"]["t"]["ax"] - self.cfg["opt"]["metric_time_in_ps"])
        )
        self.metric_dt = self.cfg["save"]["default"]["t"]["ax"][1] - self.cfg["save"]["default"]["t"]["ax"][0]
        try:
            return super().init_modules()
        except NotImplementedError:
            modules = {}
            if "generative" == self.cfg["drivers"]["E0"]["shape"].casefold():
                modules = {"laser": GenerativeDriver(self.cfg)}
            elif "learner" == self.cfg["drivers"]["E0"]["shape"].casefold():
                modules = {"laser": TPDLearner(self.cfg)}
            elif "random_phaser" == self.cfg["drivers"]["E0"]["shape"].casefold():
                laser_module = driver.load(self.cfg, ArbitraryDriver)
                laser_module = tree_at(
                    lambda tree: tree.phases,
                    laser_module,
                    replace=jnp.array(np.random.uniform(-1, 1, self.cfg["drivers"]["E0"]["num_colors"])),
                )
                modules = {"laser": laser_module}

            elif "zero_lines" == self.cfg["drivers"]["E0"]["shape"].casefold():
                modules = {"laser": ZeroLiner(self.cfg)}

            else:
                raise NotImplementedError("Only generative model is supported in this repo")

        return modules

    def write_units(self):
        units_dict = super().write_units()
        L = _Q(self.cfg["density"]["gradient scale length"]).to("um").value
        lambda0 = _Q(self.cfg["units"]["laser_wavelength"]).to("um").value

        tau0_over_tauc = self.cfg["drivers"]["E0"]["delta_omega_max"]
        Te = _Q(self.cfg["units"]["reference electron temperature"]).to("keV").value
        gradient_scale_length = _Q(self.cfg["density"]["gradient scale length"]).to("um").value
        I_thresh = calc_tpd_threshold_intensity(Te, Ln=gradient_scale_length, w0=self.cfg["units"]["derived"]["w0"])
        I_thresh_broadband = calc_tpd_broadband_threshold_intensity(Te, gradient_scale_length, lambda0, tau0_over_tauc)
        
        units_dict["monochromatic threshold"] = float(I_thresh)
        units_dict["broadband threshold"] = float(I_thresh_broadband)
        self.cfg["units"]["derived"]["broadband threshold"] = units_dict["broadband threshold"]
        self.cfg["units"]["derived"]["monochromatic threshold"] = units_dict["monochromatic threshold"]
        return units_dict

    def __call__(self, trainable_modules, args=None):
        if args is not None:
            if "static_modules" in args:
                trainable_modules["laser"] = combine(trainable_modules["laser"], args["static_modules"]["laser"])

        out_dict = super().__call__(trainable_modules, args)
        e_sq = jnp.mean(out_dict["solver result"].ys["default"]["e_sq"][self.metric_timesteps :])  # * self.metric_dt
        log10e_sq = jnp.log10(e_sq)
        return log10e_sq, out_dict

    def vg(self, trainable_modules, args=None):
        return filter_value_and_grad(self.__call__, has_aux=True)(trainable_modules, args)

    def post_process(self, run_output: Dict, td: str) -> Dict:
        metrics = {}
        if isinstance(run_output, tuple):
            val, run_output = run_output
            metrics["loss"] = float(val)

        ppo = super().post_process(run_output, td)
        bw_metrics = postprocess_bandwidth(
            run_output["args"]["drivers"], self, td, ppo["x"]["background_density"].data[0]
        )
        fields = ppo["x"]
        dx = fields.coords["x (um)"].data[1] - fields.coords["x (um)"].data[0]
        dy = fields.coords["y (um)"].data[1] - fields.coords["y (um)"].data[0]
        dt = fields.coords["t (ps)"].data[1] - fields.coords["t (ps)"].data[0]

        tint = self.cfg["opt"]["metric_time_in_ps"]
        it = int(tint / dt)
        total_esq = np.abs(fields["ex"][it:].data) ** 2 + np.abs(fields["ey"][it:].data ** 2) * dx * dy * dt
        bw_metrics[f"mean_e_sq_{tint}_ps_to_end".replace(".", "p")] = float(np.mean(total_esq))
        bw_metrics[f"log10_mean_e_sq_{tint}_ps_to_end".replace(".", "p")] = float(
            np.log10(bw_metrics[f"mean_e_sq_{tint}_ps_to_end".replace(".", "p")])
        )
        bw_metrics[f"growth_rate_{tint}_ps_to_end".replace(".", "p")] = float(
            np.mean(np.gradient(np.log(total_esq), dt))
        )

        metrics.update(bw_metrics)

        return {"k": ppo["k"], "x": fields, "metrics": metrics}


class ArbitrarywIntensityDriver(ArbitraryDriver):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, state, args):
        state, args = super().__call__(state, args)
        args["drivers"]["E0"]["intensities"] *= args["laser_intensity_factor"]
        return state, args


class TPDThresholdModule(TPDModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_modules(self):
        self.metric_timesteps = np.argmin(
            np.abs(self.cfg["save"]["default"]["t"]["ax"] - self.cfg["opt"]["metric_time_in_ps"])
        )
        self.metric_dt = self.cfg["save"]["default"]["t"]["ax"][1] - self.cfg["save"]["default"]["t"]["ax"][0]
        modules = {}
        if "arbitrarywintensity" == self.cfg["drivers"]["E0"]["shape"].casefold():
            modules = {"laser": ArbitrarywIntensityDriver(self.cfg)}
        else:
            raise NotImplementedError
        self.grad_call = False
        return modules

    def last_sim(self, intensity, args):
        print("running a sim")
        trainable_modules = args["diff_modules"]
        args["laser_intensity_factor"] = intensity / float(self.cfg["units"]["laser intensity"][:-7])
        loss_val, aux_dict = super().__call__(trainable_modules, args)
        distance_to_target = loss_val - 2.0
        return distance_to_target, aux_dict

    def one_sim(self, intensity, args):
        distance_to_target, _ = self.last_sim(intensity, args)
        jax_print("intensity = {i}, distance = {x}", i=intensity, x=distance_to_target)
        return distance_to_target

    def __call__(self, trainable_modules, args=None):
        """
        Finds the threshold and returns the negative of it

        :param trainable_modules: Description
        :param args: Description
        """
        # self.grad_call = False
        args["diff_modules"] = trainable_modules
        optmx_sol = optmx.root_find(
            self.one_sim,
            solver=optmx.Bisection(rtol=0.01, atol=0.05),
            # solver=optmx.Newton(rtol=0.01, atol=0.05),
            # solver=optmx.Chord(rtol=0.01, atol=0.05),
            y0=self.cfg["units"]["derived"]["broadband threshold"] * 1e14,
            args=args,
            options={
                "lower": self.cfg["units"]["derived"]["I_thresh"] * 1e14,
                "upper": 10 * self.cfg["units"]["derived"]["I_thresh"] * 1e14,
            },
            has_aux=False,
        )
        threshold_intensity = optmx_sol.value
        _, aux = self.last_sim(threshold_intensity, args)
        aux["steps"] = optmx_sol.stats["num_steps"]
        aux["threshold_intensity_1e14"] = threshold_intensity / 1e14
        return -threshold_intensity, aux

    def vg(self, trainable_modules, args=None):
        self.grad_call = True
        return filter_jit(filter_value_and_grad(self.__call__, has_aux=True))(trainable_modules, args)

    def post_process(self, run_output, td):
        out_dict = super().post_process(run_output, td)
        if not self.grad_call:
            _, run_output = run_output

        out_dict["metrics"]["threshold_intensity_1e14"] = float(run_output["threshold_intensity_1e14"])
        out_dict["metrics"]["num_steps"] = run_output["steps"]

        return out_dict
