from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple
import logging, os, dill as pickle
import parsl
from parsl import python_app
from itertools import product
import optax

import yaml, mlflow, tempfile, os
import numpy as np, equinox as eqx
import jax
from jax.flatten_util import ravel_pytree


from ml4tpd.parsl_utils import setup_parsl
from adept import ergoExo, utils as adept_utils
from ml4tpd import TPDModule
from ml4tpd.helpers import calc_tpd_broadband_threshold_intensity
from ml4tpd.runners import run_one_val_and_grad, run_adept_fwd

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def train_model(_cfg_path, parsl_provider="gpu", num_nodes=4):
    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    with open(f"{_cfg_path}", "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
    exo = ergoExo()
    modules = exo.setup(orig_cfg, adept_module=TPDModule)
    diff_params, _ = eqx.partition(modules["laser"], modules["laser"].get_partition_spec())

    lr_sched = optax.cosine_decay_schedule(
        init_value=orig_cfg["opt"]["learning_rate"], decay_steps=orig_cfg["opt"]["decay_steps"]
    )

    with mlflow.start_run(run_id=exo.mlflow_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=lr_sched)
        opt_state = opt.init(eqx.filter(diff_params, eqx.is_array))  # initialize the optimizer state

        epoch_loss_mean = _train_(0, orig_cfg, parsl_provider, num_nodes, exo.mlflow_run_id, modules, opt, opt_state)

    return epoch_loss_mean


def initialize_training_data(cfg):
    """Generate (Te, GSL, baseline intensity) tuples from config-defined ranges."""
    nt = cfg["training data"]["num_temperatures"]
    ngsl = cfg["training data"]["num_gradient_scale_lengths"]
    bandwidth = cfg["drivers"]["E0"]["delta_omega_max"] * 2
    temperatures = np.round(np.linspace(2000, 4000, nt), 2)
    gradient_scale_lengths = np.round(np.linspace(200, 600, ngsl), 2)
    all_hps = []

    for te, gsl in product(temperatures, gradient_scale_lengths):
        all_hps.append(
            (te, gsl, round(calc_tpd_broadband_threshold_intensity(te / 1000, gsl, 0.351, bandwidth) * 1e14, 2))
        )
    return all_hps


@dataclass
class BatchResult:
    mean_loss: float
    grad_norm: float
    total_samples: int
    unstable_samples: int


@dataclass
class EpochResult:
    loss_mean: float
    grad_norm_mean: float
    total_samples: int
    unstable_samples: int

    @property
    def unstable_fraction(self) -> float:
        return float(self.unstable_samples / self.total_samples) if self.total_samples else 0.0


class TrainingLoop:
    """Controller that runs training epochs, handles validation, and updates intensity limits."""
    def __init__(
        self,
        *,
        start_epoch: int,
        orig_cfg: dict,
        base_cfg: dict,
        modules,
        diff_params,
        static_params,
        opt,
        opt_state,
        parent_run_id: str,
        parsl_run_one_val_and_grad,
        parsl_run_fwd,
        all_hps,
        num_nodes: int,
    ):
        self.start_epoch = start_epoch
        self.orig_cfg = orig_cfg
        self.base_cfg = base_cfg
        self.modules = modules
        self.diff_params = diff_params
        self.static_params = static_params
        self.opt = opt
        self.opt_state = opt_state
        self.parent_run_id = parent_run_id
        self.parsl_run_one_val_and_grad = parsl_run_one_val_and_grad
        self.parsl_run_fwd = parsl_run_fwd
        self.all_hps = all_hps
        self.num_batches = 1  # len(all_hps) // batch_size
        self.batch_size = num_nodes * 4
        self.max_epochs = 200
        self.validation_interval = 3

        self.rng = np.random.default_rng()

        intensity_schedule_cfg = orig_cfg.get("intensity_schedule", {})
        self.factor_cap_min = float(intensity_schedule_cfg.get("cap_min", 1.0))
        self.factor_cap_max = float(intensity_schedule_cfg.get("cap_max", 2.0))
        initial_cap = float(intensity_schedule_cfg.get("initial_cap", 1.1))
        self.factor_cap = float(np.clip(initial_cap, self.factor_cap_min, self.factor_cap_max))
        self.factor_cap_growth = float(intensity_schedule_cfg.get("growth_rate", 1.05))
        self.factor_cap_decay = float(intensity_schedule_cfg.get("decay_rate", 0.9))
        self.bracket_tolerance = float(intensity_schedule_cfg.get("bracket_tolerance", 0.02))

        self.grad_clip_norm = float(orig_cfg.get("opt", {}).get("grad_clip_norm", 10000.0))
        self.hp_states = {
            (hp[0], hp[1]): {"safe": self.factor_cap_min, "unsafe": self.factor_cap_max} for hp in all_hps
        }

    def run(self, workdir: str) -> float:
        """Execute the full training curriculum and return the final epoch loss."""
        epoch_loss_mean = 0.0
        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_result = self._run_epoch(epoch, workdir)
            unstable_fraction = epoch_result.unstable_fraction
            self._update_factor_cap(epoch_result, unstable_fraction)
            val_loss = self._maybe_run_validation(epoch, workdir)
            self._log_epoch_metrics(epoch, epoch_result, unstable_fraction, val_loss)
            epoch_loss_mean = epoch_result.loss_mean
        return epoch_loss_mean

    def _run_epoch(self, epoch: int, workdir: str) -> EpochResult:
        """Run one epoch of batched jobs and aggregate losses, gradients, and stability data."""
        epoch_losses = []
        epoch_gradnorms = []
        epoch_total_samples = 0
        epoch_unstable_samples = 0

        self.rng.shuffle(self.all_hps)

        for batch_idx in range(self.num_batches):
            training_data = self.all_hps[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
            current_batch_size = len(training_data)
            if current_batch_size == 0:
                continue

            module_path = self._save_batch_state(epoch, batch_idx, workdir)
            batch_result = self._run_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                training_data=training_data,
                module_path=module_path,
            )
            epoch_losses.append(batch_result.mean_loss)
            epoch_gradnorms.append(batch_result.grad_norm)
            epoch_total_samples += batch_result.total_samples
            epoch_unstable_samples += batch_result.unstable_samples

        loss_mean = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        grad_norm_mean = float(np.mean(epoch_gradnorms)) if epoch_gradnorms else 0.0
        return EpochResult(
            loss_mean=loss_mean,
            grad_norm_mean=grad_norm_mean,
            total_samples=epoch_total_samples,
            unstable_samples=epoch_unstable_samples,
        )

    def _save_batch_state(self, epoch: int, batch_idx: int, workdir: str) -> str:
        """Persist optimizer state and weights for reproducibility and later export."""
        opt_state_path = os.path.join(workdir, f"opt-state-epoch={epoch}-batch-{batch_idx}.pkl")
        with open(opt_state_path, "wb") as fi:
            pickle.dump(self.opt_state, fi)

        weights_dir = os.path.join(workdir, "weights-history")
        module_path = os.path.join(weights_dir, f"weights-e{epoch:02d}-b{batch_idx:02d}.eqx")
        self.modules["laser"].save(module_path)

        mlflow.log_artifact(opt_state_path)
        mlflow.log_artifact(module_path)
        return module_path

    def _run_batch(self, *, epoch: int, batch_idx: int, training_data, module_path: str) -> BatchResult:
        """Launch Parsl tasks for one batch, accumulate gradients, and apply an optimizer step."""
        self.orig_cfg["drivers"]["E0"]["file"] = module_path

        val_and_grads = []
        for sample_idx, training_example in enumerate(training_data):
            print(f"{epoch=}, {batch_idx=}, {sample_idx=} -- _Training Data: {training_example}")
            run_cfg_path, export = self._write_training_sample_config(
                epoch=epoch,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                training_example=training_example,
                module_path=module_path,
            )
            val_and_grads.append(
                self.parsl_run_one_val_and_grad(
                    parent_run_id=self.parent_run_id,
                    _run_cfg_path=run_cfg_path,
                    export=export,
                )
            )

        vgs = [vg.result() for vg in val_and_grads]
        raw_validation_losses = np.array([v for v, _ in vgs])
        validation_losses = np.nan_to_num(raw_validation_losses, nan=30.0, posinf=30.0, neginf=30.0)
        validation_losses = np.where(validation_losses > 30.0, 30, validation_losses)
        mean_loss = float(np.mean(validation_losses))

        unstable_samples = int(np.sum(~(np.isfinite(raw_validation_losses) & (raw_validation_losses <= 0.0))))

        valid_grad_entries = [
            grad for loss, grad in vgs if np.isfinite(loss) and self._tree_all_finite(grad["laser"])
        ]
        dropped_grads = len(vgs) - len(valid_grad_entries)
        if valid_grad_entries:
            avg_grad = adept_utils.all_reduce_gradients(valid_grad_entries, len(valid_grad_entries))
        else:
            zero_grad = jax.tree_map(lambda x: np.zeros_like(x), self.diff_params)
            avg_grad = {"laser": zero_grad}

        grad_norm_unclipped = optax.global_norm(avg_grad["laser"])
        if grad_norm_unclipped > self.grad_clip_norm:
            scale = self.grad_clip_norm / (grad_norm_unclipped + 1e-8)
            avg_grad["laser"] = jax.tree_map(lambda g: g * scale, avg_grad["laser"])

        flat_grad, _ = ravel_pytree(avg_grad["laser"])
        mlflow.log_metrics(
            {
                "batch grad norm": float(np.linalg.norm(flat_grad)),
                "batch loss": mean_loss,
                "batch grad norm unclipped": float(grad_norm_unclipped),
            },
            step=epoch * self.num_batches + batch_idx,
        )
        if dropped_grads:
            mlflow.log_metric(
                "batch dropped grad count",
                dropped_grads,
                step=epoch * self.num_batches + batch_idx,
            )

        updates, self.opt_state = self.opt.update(avg_grad["laser"], self.opt_state, self.diff_params)
        self.diff_params = eqx.apply_updates(self.diff_params, updates)
        self.modules["laser"] = eqx.combine(self.diff_params, self.static_params)

        return BatchResult(
            mean_loss=mean_loss,
            grad_norm=float(np.linalg.norm(flat_grad)),
            total_samples=len(training_data),
            unstable_samples=unstable_samples,
        )

    def _write_training_sample_config(
        self,
        *,
        epoch: int,
        batch_idx: int,
        sample_idx: int,
        training_example,
        module_path: str,
    ) -> Tuple[str, bool]:
        """Write a single training run configuration and return its path plus export flag."""
        tt, gsl, base_intensity = training_example
        export = bool(self.rng.choice([True, False], p=[0.25, 0.75]))

        factor = float(self.rng.uniform(self.factor_cap_min, self.factor_cap))
        intensity = factor * base_intensity

        self.orig_cfg["units"]["reference electron temperature"] = f"{tt:.3f} eV"
        self.orig_cfg["density"]["gradient scale length"] = f"{gsl:.3f} um"
        self.orig_cfg["units"]["intensity factor"] = f"{factor:.3f}"
        self.orig_cfg["units"]["laser intensity"] = f"{intensity:.2e} W/cm^2"
        self.orig_cfg["grid"]["dt"] = f"{self.rng.uniform(6, 8):.3f} fs"
        self.orig_cfg["mlflow"]["run"] = (
            f"epoch-{epoch}-batch-{batch_idx}-temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
        )
        self.orig_cfg["mlflow"]["export"] = str(export)

        config_dir = os.path.dirname(os.path.dirname(module_path))
        run_cfg_path = os.path.join(
            config_dir,
            f"config-epoch={epoch}-batch={batch_idx}-sample={sample_idx}.yaml",
        )
        with open(run_cfg_path, "w") as fi:
            yaml.dump(self.orig_cfg, fi)

        return run_cfg_path, export

    def _update_factor_cap(self, epoch_result: EpochResult, unstable_fraction: float) -> None:
        """Adjust the intensity cap based on epoch loss and instability frequency."""
        if epoch_result.loss_mean <= 0 and unstable_fraction == 0.0:
            self.factor_cap = min(self.factor_cap * self.factor_cap_growth, self.factor_cap_max)
        elif epoch_result.loss_mean > 0 or unstable_fraction > 0.2:
            self.factor_cap = max(self.factor_cap * self.factor_cap_decay, self.factor_cap_min)
        self.factor_cap = float(np.clip(self.factor_cap, self.factor_cap_min, self.factor_cap_max))

    def _maybe_run_validation(self, epoch: int, workdir: str):
        """Optionally run validation and refine brackets according to the schedule."""
        if epoch % self.validation_interval != 0:
            return None

        latest_weights_path = os.path.join(workdir, "weights-history", f"weights-e{epoch:02d}-latest.eqx")
        self.modules["laser"].save(latest_weights_path)

        validation_tasks = []
        for idx, (tt, gsl, base_intensity) in enumerate(self.all_hps):
            hp_key = (tt, gsl)
            state = self.hp_states[hp_key]
            bracket_width = state["unsafe"] - state["safe"]
            factor = (
                state["safe"] if bracket_width <= self.bracket_tolerance else 0.5 * (state["safe"] + state["unsafe"])
            )
            factor = float(np.clip(factor, self.factor_cap_min, self.factor_cap_max))

            validation_cfg_path = self._write_validation_config(
                epoch=epoch,
                hp_index=idx,
                tt=tt,
                gsl=gsl,
                base_intensity=base_intensity,
                factor=factor,
                weights_path=latest_weights_path,
                workdir=workdir,
            )

            validation_tasks.append(
                (
                    self.parsl_run_fwd(validation_cfg_path, parent_run_id=self.parent_run_id),
                    hp_key,
                    factor,
                )
            )

        validation_losses = []
        for future, hp_key, factor in validation_tasks:
            raw_loss = future.result()
            validation_losses.append(raw_loss)
            self._update_bracket(hp_key, factor, raw_loss)

        if not validation_losses:
            return None

        vals = np.array(validation_losses)
        vals = np.nan_to_num(vals, nan=30.0, posinf=30.0, neginf=30.0)
        vals = np.where(vals > 30.0, 30, vals)
        return float(np.mean(vals))

    def _write_validation_config(
        self,
        *,
        epoch: int,
        hp_index: int,
        tt: float,
        gsl: float,
        base_intensity: float,
        factor: float,
        weights_path: str,
        workdir: str,
    ) -> str:
        """Emit a validation config to probe the candidate factor for a given condition."""
        validation_cfg = deepcopy(self.base_cfg)
        intensity = factor * base_intensity

        validation_cfg["save"]["fields"]["t"]["dt"] = "0.25 ps"
        validation_cfg["grid"]["dt"] = f"{self.rng.uniform(1, 3):.3f} fs"
        validation_cfg["drivers"]["E0"]["file"] = weights_path
        validation_cfg["units"]["reference electron temperature"] = f"{tt:.3f} eV"
        validation_cfg["density"]["gradient scale length"] = f"{gsl:.3f} um"
        validation_cfg["units"]["intensity factor"] = f"{factor:.3f}"
        validation_cfg["units"]["laser intensity"] = f"{intensity:.2e} W/cm^2"
        validation_cfg["mlflow"]["run"] = f"epoch-{epoch}-validation-temperature={tt:.1f}-gsl={gsl:.1f}"
        validation_cfg["mlflow"]["export"] = "True"

        validation_cfg_path = os.path.join(
            workdir,
            f"validation-config-epoch={epoch}-hp={hp_index}.yaml",
        )
        with open(validation_cfg_path, "w") as fi:
            yaml.dump(validation_cfg, fi)

        return validation_cfg_path

    def _update_bracket(self, hp_key, factor: float, raw_loss: float) -> None:
        """Update the safe/unsafe bounds for one (Te, GSL) pair."""
        state = self.hp_states[hp_key]
        is_stable = np.isfinite(raw_loss) and raw_loss <= 0.0
        if is_stable:
            state["safe"] = max(state["safe"], factor)
        else:
            state["unsafe"] = min(state["unsafe"], max(factor, state["safe"] + 1e-3))
            state["unsafe"] = max(state["unsafe"], state["safe"] + 1e-3)

    @staticmethod
    def _tree_all_finite(tree) -> bool:
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)

    def _log_epoch_metrics(
        self, epoch: int, epoch_result: EpochResult, unstable_fraction: float, val_loss: float
    ) -> None:
        """Log epoch aggregates and optional validation loss to MLflow."""
        safe_values = [state["safe"] for state in self.hp_states.values()] if self.hp_states else []
        safe_mean = float(np.mean(safe_values)) if safe_values else 0.0
        safe_min = float(np.min(safe_values)) if safe_values else 0.0

        metrics = {
            "epoch loss": epoch_result.loss_mean,
            "epoch grad norm": epoch_result.grad_norm_mean,
            "epoch unstable fraction": unstable_fraction,
            "factor cap": self.factor_cap,
            "safe factor mean": safe_mean,
            "safe factor min": safe_min,
        }
        if val_loss is not None:
            metrics["val loss"] = val_loss

        mlflow.log_metrics(metrics, step=epoch)


def _train_(
    start_epoch,
    orig_cfg,
    parsl_provider,
    num_nodes,
    parent_run_id,
    modules,
    opt,
    opt_state,
):
    """Spin up Parsl, create a TrainingLoop, and run the configured number of epochs."""
    parsl_config = setup_parsl(parsl_provider, 4, nodes=num_nodes, walltime="8:00:00")
    parsl_run_one_val_and_grad = python_app(run_one_val_and_grad)
    parsl_run_fwd = python_app(run_adept_fwd)
    diff_params, static_params = eqx.partition(modules["laser"], modules["laser"].get_partition_spec())
    all_hps = initialize_training_data(cfg=orig_cfg)
    base_cfg = deepcopy(orig_cfg)
    trainer = TrainingLoop(
        start_epoch=start_epoch,
        orig_cfg=orig_cfg,
        base_cfg=base_cfg,
        modules=modules,
        diff_params=diff_params,
        static_params=static_params,
        opt=opt,
        opt_state=opt_state,
        parent_run_id=parent_run_id,
        parsl_run_one_val_and_grad=parsl_run_one_val_and_grad,
        parsl_run_fwd=parsl_run_fwd,
        all_hps=all_hps,
        num_nodes=num_nodes,
    )
    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
        os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)
        with parsl.load(parsl_config):
            epoch_loss_mean = trainer.run(td)
    return epoch_loss_mean


def initialize_resume(run_id: str, tmpdir: str) -> str:
    """
    - Download config using mlflow download artifact
    - find latest epoch and batch number by checking the logged metrics
    - download weights and opt state
    - continue training

    :param run_id: Description
    :type run_id: str
    :return: Description
    :rtype: str
    """

    # Download the config file
    cfg_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml", dst_path=tmpdir)

    # Find latest epoch and batch number
    all_artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    all_weights = []
    for artifact in all_artifacts:
        if artifact.path.startswith("weights-e"):
            all_weights.append(artifact.path)

    # the weights will have a name like weights-e02-b05.eqx
    # we want to find e02-b05.eqx in that case
    # sort the weights to find the latest epoch and latest batch

    epochs = set()
    for weight in all_weights:
        epoch_str = weight.split("-")[1]  # e02
        epoch_num = int(epoch_str[1:])  # 02 -> 2
        epochs.add(epoch_num)
    latest_epoch = max(epochs)

    all_weights = []
    for artifact in all_artifacts:
        if artifact.path.startswith(f"weights-e{latest_epoch:02d}-b"):
            all_weights.append(artifact.path)

    batches = set()
    for weight in all_weights:
        batch_str = weight.split("-")[2]  # b05.eqx
        batch_num = int(batch_str[1 : batch_str.index(".")])  # 05 -> 5
        batches.add(batch_num)
    latest_batch = max(batches)

    latest_weights = f"weights-e{latest_epoch:02d}-b{latest_batch:02d}.eqx"

    # download weights
    weights_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=latest_weights, dst_path=tmpdir)
    opt_state_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"opt-state-epoch={latest_epoch}-batch-0.pkl", dst_path=tmpdir
    )

    print(f"Resuming from epoch {latest_epoch} with weights {weights_path} and opt state {opt_state_path}")

    return cfg_path, weights_path, opt_state_path, latest_epoch


def resume_train_model(cfg_path, run_id, start_epoch, weights_path, opt_state_path, parsl_provider="gpu", num_nodes=4):
    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    with open(f"{cfg_path}", "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    orig_cfg["drivers"]["E0"]["file"] = str(weights_path)

    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])

    copy_cfg = deepcopy(orig_cfg)
    exo = ergoExo(mlflow_run_id=run_id)
    modules = exo._setup_(copy_cfg, td=os.path.dirname(weights_path), adept_module=TPDModule, log=False)

    lr_sched = optax.cosine_decay_schedule(
        init_value=orig_cfg["opt"]["learning_rate"], decay_steps=orig_cfg["opt"]["decay_steps"]
    )

    with mlflow.start_run(run_id=exo.mlflow_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=lr_sched)
        with open(opt_state_path, "rb") as fi:
            opt_state = pickle.load(fi)

        epoch_loss_mean = _train_(
            start_epoch,
            orig_cfg,
            parsl_provider,
            num_nodes,
            exo.mlflow_run_id,
            modules,
            opt,
            opt_state,
        )

    return epoch_loss_mean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    parser.add_argument("--run_id", type=str, default=None, help="The MLflow run ID to use for resuming")
    parser.add_argument("--provider", type=str, default="gpu", help="The Parsl provider to use")
    parser.add_argument("--nodes", type=int, default=4, help="The number of nodes to use")

    args = parser.parse_args()
    cfg_path = args.config
    parsl_provider = args.provider
    num_nodes = args.nodes
    resume_run_id = args.run_id

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if resume_run_id is not None:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            cfg_path, weights_path, opt_state_path, latest_epoch = initialize_resume(resume_run_id, td)
            resume_train_model(
                cfg_path,
                resume_run_id,
                latest_epoch,
                weights_path,
                opt_state_path,
                parsl_provider=parsl_provider,
                num_nodes=num_nodes,
            )
    else:
        train_model(cfg_path, parsl_provider=parsl_provider, num_nodes=num_nodes)
