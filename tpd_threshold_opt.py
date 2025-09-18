import logging, os
import parsl
from parsl import python_app
from itertools import product

from ml4tpd.parsl_utils import setup_parsl

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

def _maximize_threshold(_cfg_path: str):
    """
    Sets up and runs the parent run which is the optimization loop

    Args:
        _cfg_path: str: Path to the config file

    """
    from copy import deepcopy
    from adept import ergoExo
    from ml4tpd import TPDThresholdModule, threshold_fns

    import yaml, mlflow, tempfile, os

    if "BASE_TEMPDIR" in os.environ:
        BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
    else:
        BASE_TEMPDIR = None

    with open(_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    if cfg["opt"]["method"] == "optax":
        optimization_loop = threshold_fns.optax_loop
    elif cfg["opt"]["method"] == "scipy":
        optimization_loop = threshold_fns.scipy_loop
    else:
        raise NotImplementedError(f"Optimization method {cfg['opt']['method']} not implemented.")

    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        parent_run_id = mlflow_run.info.run_id
        orig_cfg = deepcopy(cfg)

    exo = ergoExo(mlflow_run_id=parent_run_id, mlflow_nested=False)
    modules = exo.setup(cfg, adept_module=TPDThresholdModule)

    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        optimization_loop(orig_cfg, modules)

    # Final cleanup after entire optimization run
    logger.info("Performing final XLA cache cleanup after optimization run")
    threshold_fns.clear_xla_cache()

    return mlflow_run

def scan_loop(_cfg_path):
    import uuid
    import tempfile
    import yaml
    import numpy as np
    temperatures = np.round(np.linspace(2000, 4000, 8), 0)
    gradient_scale_lengths = np.round(np.linspace(200, 800, 8), 0)

    all_hps = list(product(temperatures, gradient_scale_lengths))
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    parsl_config = setup_parsl(orig_cfg["parsl"]["provider"], 4, nodes=orig_cfg["parsl"]["nodes"], walltime="16:00:00")
    maximize_threshold = python_app(_maximize_threshold)
    orig_cfg["mlflow"]["experiment"] = "maximize_threshold"
    all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])

    with parsl.load(parsl_config):
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
            num_nodes = orig_cfg["parsl"]["nodes"]
            batch_size = num_nodes * 4
            offset = 0
            num_batches = int(np.ceil((len(all_hps) - offset) / batch_size))
            print(f"Running {num_batches} batches of {batch_size} hyperparameter combinations each")
            print(f"Total combinations: {len(all_hps)}")
            for i in range(num_batches):
                vals = {}
                hp_slice = slice(batch_size * i + offset, batch_size * (i + 1) + offset)
                for tt, gsl in all_hps[hp_slice]:
                    run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}"
                    orig_cfg["mlflow"]["run"] = run_name
                    # check if run name exists by first searching all runs and then checking if the run name exists
                    # all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])
                    # if run_name in all_runs["tags.mlflow.runName"].values:
                        # print(f"Run {run_name} already exists.")
                    # else:
                    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
                    orig_cfg["drivers"]["E0"]["shape"] = "arbitrarywintensity"
                    orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
                    orig_cfg["density"]["gradient scale length"] = f"{gsl} um"
                    with open(
                        new_cfg_path := os.path.join(_td, f"config-{str(uuid.uuid4())[-6:]}.yaml"), "w"
                    ) as fi:
                        yaml.dump(orig_cfg, fi)

                    vals[tt, gsl] = maximize_threshold(new_cfg_path)

                for (tt, gsl), v in vals.items():
                    val = v.result()


if __name__ == "__main__":
    import argparse, mlflow, yaml, os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" # second gpu
    import equinox as eqx
    from adept import ergoExo
    from ml4tpd import TPDThresholdModule


    parser = argparse.ArgumentParser(description="Run TPD training.")
    parser.add_argument("--config", type=str, help="The config file")
    args = parser.parse_args()
    cfg_path = args.config

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# 
    mlflow_run = scan_loop(cfg_path)

    # with open(cfg_path, "r") as fi:
    #     cfg = yaml.safe_load(fi)
    
    # cfg["mlflow"]["experiment"] = "maximize_threshold"
    # cfg["drivers"]["E0"]["shape"] = "arbitrarywintensity"

    # exo = ergoExo()
    # modules = exo.setup(cfg, adept_module=TPDThresholdModule)
    # diff_modules, static_modules = {}, {}
    # diff_modules["laser"], static_modules["laser"] = eqx.partition(
    #     modules["laser"], modules["laser"].get_partition_spec()
    # )
    
    # run_output, ppo, _ = exo(diff_modules, args={"static_modules": static_modules})

    # val, grad, (sol, ppo, _) = exo.val_and_grad(diff_modules, args={"static_modules": static_modules})

    # val = run_output[0]


