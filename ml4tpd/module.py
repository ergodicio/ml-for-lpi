from typing import Dict
import numpy as np
from astropy.units import Quantity as _Q
from jax import numpy as jnp
from jax.lax import stop_gradient
from jax.random import normal, PRNGKey

from equinox import filter_value_and_grad, Module

from adept.lpse2d import BaseLPSE2D, UniformDriver

from . import nn
from .postprocess import postprocess_bandwidth


class GenerativeDriver(UniformDriver):
    input_width: int
    model: Module
    amp_output: str
    phase_output: str

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self.input_width = cfg["drivers"]["E0"]["params"]["input_width"]
        cfg["drivers"]["E0"]["params"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        self.model = nn.GenerativeModel(**cfg["drivers"]["E0"]["params"])
        self.amp_output = cfg["drivers"]["E0"]["output"]["amp"]
        self.phase_output = cfg["drivers"]["E0"]["output"]["phase"]

    def __call__(self, state: Dict, args: Dict) -> tuple:
        inputs = normal(PRNGKey(seed=np.random.randint(2**20)), shape=(self.input_width,))
        ints_and_phases = self.model(inputs)
        ints, phases = self.scale_ints_and_phases(ints_and_phases["amps"], ints_and_phases["phases"])
        args["drivers"]["E0"] = {
            "delta_omega": stop_gradient(self.delta_omega),
            "initial_phase": phases,
            "intensities": ints,
        } | {k: stop_gradient(v) for k, v in self.envelope.items()}
        return state, args


class TPDModule(BaseLPSE2D):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def init_modules(self) -> Dict:
        try:
            return super().init_modules()
        except NotImplementedError:
            # check if shape is etamodel
            modules = {}

            if "generative" == self.cfg["drivers"]["E0"]["shape"].casefold():
                modules = {"laser": GenerativeDriver(self.cfg)}

            else:
                raise NotImplementedError("Only generative model is supported in this repo")

        return modules

    def write_units(self):
        units_dict = super().write_units()
        L = _Q(self.cfg["density"]["gradient scale length"]).to("um").value
        lambda0 = _Q(self.cfg["units"]["laser_wavelength"]).to("um").value

        tau0_over_tauc = self.cfg["drivers"]["E0"]["delta_omega_max"]

        units_dict["broadband threshold"] = float(
            232
            * _Q(self.cfg["units"]["reference electron temperature"]).to("keV").value ** 0.75
            / L ** (2 / 3)
            / lambda0 ** (4 / 3)
            * (tau0_over_tauc) ** 0.5
        )

        self.cfg["units"]["derived"]["broadband threshold"] = units_dict["broadband threshold"]

        return units_dict

    def __call__(self, trainable_modules, args=None):
        out_dict = super().__call__(trainable_modules, args)
        phi_xy = out_dict["solver result"].ys["fields"]["epw"][-4:]
        phi_k = jnp.fft.fft2(phi_xy.view(jnp.complex128), axes=(-2, -1))
        ex_k = -1j * self.cfg["save"]["fields"]["kx"][None, :, None] * phi_k
        ey_k = -1j * self.cfg["save"]["fields"]["ky"][None, None, :] * phi_k
        log10e_sq = jnp.log10(jnp.sum(jnp.abs(ex_k) ** 2 + jnp.abs(ey_k) ** 2))
        return log10e_sq, out_dict

    def vg(self, trainable_modules, args=None):
        return filter_value_and_grad(self.__call__, has_aux=True)(trainable_modules, args)

    def post_process(self, run_output: Dict, td: str) -> Dict:
        if isinstance(run_output, tuple):
            val, run_output = run_output

        ppo = super().post_process(run_output, td)
        metrics = postprocess_bandwidth(run_output["args"]["drivers"], self, td, ppo["x"]["background_density"].data[0])
        fields = ppo["x"]
        dx = fields.coords["x (um)"].data[1] - fields.coords["x (um)"].data[0]
        dy = fields.coords["y (um)"].data[1] - fields.coords["y (um)"].data[0]
        dt = fields.coords["t (ps)"].data[1] - fields.coords["t (ps)"].data[0]

        tint = 5.0  # last tint ps
        it = int(tint / dt)
        total_esq = np.abs(fields["ex"][-it:].data) ** 2 + np.abs(fields["ey"][-it:].data ** 2) * dx * dy * dt
        metrics[f"total_e_sq_last_{tint}_ps".replace(".", "p")] = float(np.sum(total_esq))
        metrics[f"log10_total_e_sq_last_{tint}_ps".replace(".", "p")] = float(
            np.log10(metrics[f"total_e_sq_last_{tint}_ps".replace(".", "p")])
        )
        metrics[f"growth_rate_last_{tint}_ps".replace(".", "p")] = float(np.mean(np.gradient(np.log(total_esq), dt)))

        return {"k": ppo["k"], "x": fields, "metrics": metrics}
