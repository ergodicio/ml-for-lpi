from typing import Dict
import numpy as np
from astropy.units import Quantity as _Q

from jax import numpy as jnp, tree_util as jtu, nn as jnn
from equinox import combine, tree_at, filter_value_and_grad, Module, is_array, nn as eqx_nn
from jax.random import normal, PRNGKey, uniform, split

from adept.lpse2d import BaseLPSE2D, ArbitraryDriver
from adept._lpse2d.modules import driver

from . import nn
from .postprocess import postprocess_bandwidth


class ZeroLiner(ArbitraryDriver):
    threshold: float

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        old_driver = driver.load(cfg, ArbitraryDriver)
        self.intensities = old_driver.intensities
        self.phases = old_driver.phases
        self.threshold = cfg["drivers"]["E0"]["intensity_threshold"]

    def __call__(self, state: Dict, args: Dict) -> tuple:
        intensities = self.scale_intensities(self.intensities)
        intensities = intensities / jnp.sum(intensities)
        intensities = jnp.where(intensities < self.threshold, 0.0, intensities)
        intensities = intensities / jnp.sum(intensities)

        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope
        return state, args


class LearnedSpacer(ArbitraryDriver):
    first_delta_omega: float
    dw_spacing: float
    delta_omega_max: float

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.first_delta_omega = np.random.uniform(
            cfg["drivers"]["E0"]["delta_omega_min"], cfg["drivers"]["E0"]["delta_omega_max"]
        )
        self.dw_spacing = np.random.uniform(
            0, cfg["drivers"]["E0"]["delta_omega_max"] - cfg["drivers"]["E0"]["delta_omega_min"]
        )
        self.delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]

    def __call__(self, state: Dict, args: Dict) -> tuple:

        delta_omega = jnp.arange(self.first_delta_omega, self.delta_omega_max, self.dw_spacing)
        intensities = jnp.ones_like(delta_omega)
        intensities = intensities / jnp.sum(intensities)

        args["drivers"]["E0"] = {
            "delta_omega": delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope
        return state, args


def reinitialize_nns(model: Module, key: PRNGKey) -> Module:

    weight_keys = split(key, num=len(model.layers))
    bias_keys = split(key, num=len(model.layers))

    for i in range(len(model.layers)):
        model = tree_at(
            lambda tree: tree.layers[i].weight,
            model,
            replace=uniform(weight_keys[i], model.layers[i].weight.shape, minval=-1, maxval=1),
        )

        model = tree_at(
            lambda tree: tree.layers[i].bias,
            model,
            replace=uniform(bias_keys[i], model.layers[i].bias.shape, minval=-1, maxval=1),
        )

    return model


class TPDLearner(ArbitraryDriver):

    amp_model: Module
    phase_model: Module
    inputs: np.array

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        cfg["drivers"]["E0"]["params"]["nn"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        # self.model = nn.GenerativeModel(**cfg["drivers"]["E0"]["params"]["nn"])
        if cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "relu":
            act_fun = jnn.relu
        elif cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "tanh":
            act_fun = jnp.tanh
        elif cfg["drivers"]["E0"]["params"]["nn"]["activation"] == "sigmoid":
            act_fun = jnn.sigmoid
        else:
            # print("Using default activation function")
            # act_fun = lambda x: x
            raise NotImplementedError(
                f"Activation function {cfg['drivers']['E0']['params']['nn']['activation']} not supported"
            )

        nn_params = {
            "in_size": cfg["drivers"]["E0"]["params"]["nn"]["in_size"],
            "out_size": cfg["drivers"]["E0"]["num_colors"],
            "width_size": cfg["drivers"]["E0"]["params"]["nn"]["width_size"],
            "depth": cfg["drivers"]["E0"]["params"]["nn"]["depth"],
            "key": PRNGKey(seed=np.random.randint(2**20)),
            "activation": act_fun,
        }

        self.amp_model = eqx_nn.MLP(**nn_params)

        reinitialize_nns(self.amp_model, key=PRNGKey(seed=np.random.randint(2**20)))

        nn_params["key"] = PRNGKey(seed=np.random.randint(2**20))
        self.phase_model = eqx_nn.MLP(**nn_params)

        reinitialize_nns(self.phase_model, key=PRNGKey(seed=np.random.randint(2**20)))

        Te = _Q(cfg["units"]["reference electron temperature"]).to("keV").value
        Ln = _Q(cfg["density"]["gradient scale length"]).to("um").value
        I0 = _Q(cfg["units"]["laser intensity"]).to("W/cm^2").value

        Te = (Te - 3.0) / 1.0
        Ln = (Ln - 400.0) / 200
        I0 = (np.log10(I0) - 14.5) / 0.5
        self.inputs = np.array((Te, Ln, I0))

    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        amp_model_filter_spec = jtu.tree_map(lambda _: False, self.amp_model)
        for i in range(len(self.amp_model.layers)):
            amp_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                amp_model_filter_spec,
                replace=(True, True),
            )

        phase_model_filter_spec = jtu.tree_map(lambda _: False, self.phase_model)
        for i in range(len(self.phase_model.layers)):
            phase_model_filter_spec = tree_at(
                lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                phase_model_filter_spec,
                replace=(True, True),
            )

        filter_spec = tree_at(lambda tree: tree.phase_model, filter_spec, replace=phase_model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.amp_model, filter_spec, replace=amp_model_filter_spec)

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:

        ints_and_phases = {
            "amps": self.amp_model(jnp.array(self.inputs)),
            "phases": self.phase_model(jnp.array(self.inputs)),
        }
        ints, phases = self.process_amplitudes_phases(ints_and_phases)
        args["drivers"]["E0"] = {"delta_omega": self.delta_omega, "phases": phases, "intensities": ints} | self.envelope
        return state, args

    def process_amplitudes_phases(self, ints_and_phases):
        if self.model_cfg["amplitudes"]["learned"]:
            ints = ints_and_phases["amps"]
        else:
            ints = self.intensities

        ints *= 3
        ints = self.scale_intensities(ints)
        ints = ints / jnp.sum(ints)

        if self.model_cfg["phases"]["learned"]:
            _phases_ = ints_and_phases["phases"]
        else:
            _phases_ = self.phases

        phases = jnp.pi * jnp.tanh(_phases_)
        phases *= 3.0

        return ints, phases


class GenerativeDriver(ArbitraryDriver):
    input_width: int
    model: Module

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self.input_width = cfg["drivers"]["E0"]["params"]["nn"]["input_width"]
        cfg["drivers"]["E0"]["params"]["nn"]["output_width"] = cfg["drivers"]["E0"]["num_colors"]
        self.model = nn.GenerativeModel(**cfg["drivers"]["E0"]["params"]["nn"])

    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        model_filter_spec = jtu.tree_map(lambda _: False, self.model)
        if self.model_cfg["amplitudes"]["learned"]:
            amp_model_filter_spec = jtu.tree_map(lambda _: False, self.model.amp_decoder)
            for i in range(len(self.model.amp_decoder.layers)):
                amp_model_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    amp_model_filter_spec,
                    replace=(True, True),
                )

        if self.model_cfg["phases"]["learned"]:
            phase_model_filter_spec = jtu.tree_map(lambda _: False, self.model.phase_decoder)
            for i in range(len(self.model.phase_decoder.layers)):
                phase_model_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    phase_model_filter_spec,
                    replace=(True, True),
                )

        filter_spec = tree_at(lambda tree: tree.model, filter_spec, replace=model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.model.amp_decoder, filter_spec, replace=amp_model_filter_spec)
        filter_spec = tree_at(lambda tree: tree.model.phase_decoder, filter_spec, replace=phase_model_filter_spec)

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:
        inputs = normal(PRNGKey(seed=np.random.randint(2**20)), shape=(self.input_width,))
        ints_and_phases = self.model(inputs)
        ints, phases = self.process_amplitudes_phases(ints_and_phases)
        args["drivers"]["E0"] = {"delta_omega": self.delta_omega, "phases": phases, "intensities": ints} | self.envelope
        return state, args

    def process_amplitudes_phases(self, ints_and_phases):
        if self.model_cfg["amplitudes"]["learned"]:
            ints = ints_and_phases["amps"]
        else:
            ints = self.intensities

        ints = self.scale_intensities(ints)
        ints = ints / jnp.sum(ints)

        if self.model_cfg["phases"]["learned"]:
            _phases_ = ints_and_phases["phases"]
        else:
            _phases_ = self.phases

        phases = jnp.pi * jnp.tanh(_phases_)
        return ints, phases


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

        if args is not None:
            if "static_modules" in args:
                trainable_modules["laser"] = combine(trainable_modules["laser"], args["static_modules"]["laser"])

        out_dict = super().__call__(trainable_modules, args)
        e_sq = jnp.sum(out_dict["solver result"].ys["default"]["e_sq"][self.metric_timesteps :]) * self.metric_dt
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

        tint = 5.0  # last tint ps
        it = int(tint / dt)
        total_esq = np.abs(fields["ex"][-it:].data) ** 2 + np.abs(fields["ey"][-it:].data ** 2) * dx * dy * dt
        bw_metrics[f"total_e_sq_last_{tint}_ps".replace(".", "p")] = float(np.sum(total_esq))
        bw_metrics[f"log10_total_e_sq_last_{tint}_ps".replace(".", "p")] = float(
            np.log10(bw_metrics[f"total_e_sq_last_{tint}_ps".replace(".", "p")])
        )
        bw_metrics[f"growth_rate_last_{tint}_ps".replace(".", "p")] = float(np.mean(np.gradient(np.log(total_esq), dt)))

        metrics.update(bw_metrics)

        return {"k": ppo["k"], "x": fields, "metrics": metrics}
