import logging, os
from ml4tpd.parsl_utils import setup_parsl
from ml4tpd.matlab import run_matlab

import parsl
from parsl import python_app
from itertools import product
import uuid
import numpy as np
import yaml
import mlflow
import tempfile
from ml4tpd.helpers import calc_tpd_threshold_intensity
from tpd_opt import run_opt_with_retry

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_adept_fwd(_cfg_path, num_seeds=8):
    import yaml, mlflow

    from adept import ergoExo, utils as adept_utils
    import numpy as np
    from jax import config

    config.update("jax_enable_x64", True)

    from ml4tpd import TPDModule

    with open(_cfg_path, "r") as fi:
        _cfg = yaml.safe_load(fi)

    with mlflow.start_run(run_name=_cfg["mlflow"]["run"]) as parent_run:
        adept_utils.log_params(_cfg)
        vals = []
        for i in range(num_seeds):
            _cfg["drivers"]["E0"]["params"]["phases"]["seed"] = int(np.random.randint(0, 2**10))
            _cfg["mlflow"]["run"] = f"seed-{i}"
            exo = ergoExo(parent_run_id=parent_run.info.run_id, mlflow_nested=True)
            modules = exo.setup(_cfg, adept_module=TPDModule)
            run_output, ppo, _ = exo(modules)
            val = run_output[0]
            vals.append(val)
        mlflow.log_metric("loss", np.mean(vals))


def scan_loop(_cfg_path, shape="uniform", solver="adept", parsl_provider="gpu", num_nodes=4):
    temperatures = np.round(np.linspace(2000, 4000, 5), 0)
    gradient_scale_lengths = np.round(np.linspace(200, 600, 5), 0)

    # intensities = np.round(np.linspace(2.0e14, 7.0e14, 13), 3)
    intensity_factors = np.linspace(1.4, 4.0, 16)

    all_hps = list(product(temperatures, gradient_scale_lengths, intensity_factors))
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    orig_cfg["parsl"]["provider"] = parsl_provider
    orig_cfg["parsl"]["nodes"] = num_nodes
    parsl_config = setup_parsl(parsl_provider, 4, nodes=num_nodes, walltime="8:00:00")
    parsl_run_adept_fwd = python_app(run_adept_fwd)
    parsl_run_opt = python_app(run_opt_with_retry)
    orig_cfg["mlflow"]["experiment"] = f"{solver}-{shape}-tpd-100ps"
    all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])

    # find and pop completed runs from all_hps
    completed_runs = set()
    if not all_runs.empty:
        for run_name in all_runs["tags.mlflow.runName"].values:
            if run_name.startswith("temperature="):
                completed_runs.add(run_name)
    all_hps = [
        hp
        for hp in all_hps
        if f"temperature={hp[0]:.1f}-gsl={hp[1]:.1f}-intensity={1e14 * calc_tpd_threshold_intensity(hp[0] / 1000, hp[1]) * hp[2]:.2e}"
        not in completed_runs
    ][::-1]
    print(f"Found {len(completed_runs)} completed runs, {len(all_hps)} remaining.")

    with parsl.load(parsl_config):
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
            batch_size = orig_cfg["parsl"]["nodes"] * 4
            offset = 0
            num_batches = int(np.ceil((len(all_hps) - offset) / batch_size))
            print(f"Running {num_batches} batches of {batch_size} hyperparameter combinations each")
            print(f"Total combinations: {len(all_hps)}")
            for i in range(num_batches):
                vals = {}
                hp_slice = slice(batch_size * i + offset, batch_size * (i + 1) + offset)
                for tt, gsl, intensity_factor in all_hps[hp_slice]:
                    intensity = intensity_factor * calc_tpd_threshold_intensity(tt / 1000, gsl) * 1e14
                    run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
                    orig_cfg["mlflow"]["run"] = run_name
                    # check if run name exists by first searching all runs and then checking if the run name exists
                    # all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])
                    if all_runs.empty or run_name not in all_runs["tags.mlflow.runName"].values:
                        # Run does not exist, proceed to run
                        mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])

                        if shape == "mono":
                            orig_cfg["drivers"]["E0"]["num_colors"] = 1
                            orig_cfg["drivers"]["E0"]["shape"] = "uniform"
                        elif shape in ["uniform", "arbitrary"]:
                            orig_cfg["drivers"]["E0"]["num_colors"] = 64
                            orig_cfg["drivers"]["E0"]["shape"] = shape
                            if shape == "arbitrary":
                                amp_init = "uniform"
                                orig_cfg["drivers"]["E0"]["params"]["amplitudes"]["init"] = amp_init

                        orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
                        orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
                        orig_cfg["density"]["gradient scale length"] = f"{gsl} um"

                        if shape == "random_phaser":
                            orig_cfg = retrieve_latest_child_run(orig_cfg, run_name)

                        with open(
                            new_cfg_path := os.path.join(_td, f"config-{str(uuid.uuid4())[-6:]}.yaml"), "w"
                        ) as fi:
                            yaml.dump(orig_cfg, fi)

                        if solver == "adept":
                            if shape in ["uniform", "random_phaser", "mono"]:
                                vals[tt, gsl, intensity] = parsl_run_adept_fwd(
                                    _cfg_path=new_cfg_path, num_seeds=1 if shape == "mono" else 4
                                )
                            elif shape == "arbitrary":
                                vals[tt, gsl, intensity] = parsl_run_opt(new_cfg_path)
                            else:
                                raise NotImplementedError(f"Shape {shape} not implemented for adept.")

                        elif solver == "matlab":
                            if shape == "uniform":
                                vals[tt, gsl, intensity] = run_matlab(new_cfg_path, bandwidth=True)
                            elif shape == "mono":
                                vals[tt, gsl, intensity] = run_matlab(new_cfg_path, bandwidth=False)
                            else:
                                raise NotImplementedError(f"Shape {shape} not implemented for matlab.")
                        else:
                            raise NotImplementedError(f"Solver {solver} not implemented.")

                    else:
                        print(f"Run {run_name} already exists.")

                if solver == "adept":
                    for (tt, gsl, intensity), v in vals.items():
                        val = v.result()


def retrieve_latest_child_run(mlflow, orig_cfg, run_name):
    opt_parent_run = mlflow.search_runs(
        experiment_names=["arbitrary-64lines-more"],
        filter_string=f"attributes.run_name LIKE '{run_name}'",
    )["run_id"].values[0]
    child_runs = mlflow.search_runs(
        experiment_names=["arbitrary-64lines-more"],
        filter_string=f"tags.mlflow.parentRunId = '{opt_parent_run}'",
    )
    child_run = child_runs.sort_values("start_time", ascending=False).iloc[0]
    orig_cfg["drivers"]["E0"]["file"] = f"s3://public-ergodic-continuum/188470/{child_run.run_id}/artifacts/laser.eqx"

    return orig_cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    parser.add_argument(
        "--shape", type=str, default="uniform", help="The laser shape: uniform, random_phaser, mono, arbitrary"
    )
    parser.add_argument("--solver", type=str, default="adept", help="The solver to use: adept or matlab")
    parser.add_argument("--provider", type=str, default="gpu", help="The Parsl provider to use")
    parser.add_argument("--nodes", type=int, default=4, help="The number of nodes to use")

    args = parser.parse_args()
    cfg_path = args.config
    shape = args.shape
    solver = args.solver
    parsl_provider = args.provider
    num_nodes = args.nodes

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    scan_loop(cfg_path, shape=shape, solver=solver, parsl_provider=parsl_provider, num_nodes=num_nodes)
