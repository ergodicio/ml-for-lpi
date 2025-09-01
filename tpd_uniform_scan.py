import logging, os
from utils import setup_parsl


import parsl
from parsl import python_app
from itertools import product

from tpd_opt import run_opt_with_retry

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

def run_all_seeds(_cfg_path):
    import yaml, mlflow

    from adept import ergoExo, utils as adept_utils
    import numpy as np

    from ml4tpd import TPDModule

    with open(_cfg_path, "r") as fi:
        _cfg = yaml.safe_load(fi)

    with mlflow.start_run(run_name=_cfg["mlflow"]["run"]) as parent_run:
        # mlflow.log_artifacts(_td)
        adept_utils.log_params(_cfg)
        vals = []
        for i in range(8):
            _cfg["drivers"]["E0"]["params"]["phases"]["seed"] = int(np.random.randint(0, 2**10))
            _cfg["mlflow"]["run"] = f"seed-{i}"
            exo = ergoExo(parent_run_id=parent_run.info.run_id, mlflow_nested=True)
            modules = exo.setup(_cfg, adept_module=TPDModule)
            run_output, ppo, _ = exo(modules)
            val = run_output[0]
            vals.append(val)
        
        mlflow.log_metric("loss", np.mean(vals))


def scan_loop(_cfg_path, shape="uniform"):
    import uuid
    temperatures = np.round(np.linspace(2000, 4000, 5), 0)
    gradient_scale_lengths = np.round(np.linspace(200, 400, 5), 0)
    intensities = np.round(np.linspace(1e14, 1e15, 10), 3)

    all_hps = list(product(temperatures, gradient_scale_lengths, intensities))
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    parsl_config = setup_parsl(
        orig_cfg["parsl"]["provider"], 4, nodes=orig_cfg["parsl"]["nodes"], walltime="16:00:00"
    )
    parsl_run_all_seeds = python_app(run_all_seeds)
    parsl_run_opt = python_app(run_opt_with_retry)
    orig_cfg["mlflow"]["experiment"] = f"{shape}-64lines-100ps-5x5x10"
    all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])
    
                
    with parsl.load(parsl_config):
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
            num_nodes = orig_cfg["parsl"]["nodes"]
            batch_size = num_nodes * 4
            offset = 0     
            num_batches = int(np.ceil((len(all_hps)-offset) / batch_size))
            print(f"Running {num_batches} batches of {batch_size} hyperparameter combinations each")
            print(f"Total combinations: {len(all_hps)}")
            for i in range(num_batches):
                vals = {}
                hp_slice = slice(batch_size*i + offset, batch_size*(i+1) + offset)
                for tt, gsl, intensity in all_hps[hp_slice]:
                    run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
                    orig_cfg["mlflow"]["run"] = run_name
                    # check if run name exists by first searching all runs and then checking if the run name exists
                    # all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])
                    if all_runs.empty or run_name not in all_runs["tags.mlflow.runName"].values:
                        # Run does not exist, proceed to run
                        mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
                        orig_cfg["drivers"]["E0"]["shape"] = shape
                        orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
                        orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
                        orig_cfg["density"]["gradient scale length"] = f"{gsl} um"

                        if shape == "random_phaser":
                            opt_parent_run = mlflow.search_runs(
                                experiment_names=["arbitrary-64lines-more"],
                                filter_string=f"attributes.run_name LIKE '{run_name}'",
                            )["run_id"].values[0]
                            child_runs = mlflow.search_runs(
                                experiment_names=["arbitrary-64lines-more"],
                                filter_string=f"tags.mlflow.parentRunId = '{opt_parent_run}'",
                            )
                            child_run = child_runs.sort_values("start_time", ascending=False).iloc[0]
                            orig_cfg["drivers"]["E0"][
                                "file"
                            ] = f"s3://public-ergodic-continuum/188470/{child_run.run_id}/artifacts/laser.eqx"


                        with open(new_cfg_path := os.path.join(_td, f"config-{str(uuid.uuid4())[-6:]}.yaml"), "w") as fi:
                            yaml.dump(orig_cfg, fi)
                        if shape in ["uniform", "random_phaser"]:
                            vals[tt, gsl, intensity] = parsl_run_all_seeds(_cfg_path=new_cfg_path)
                        elif shape == "arbitrary":
                            vals[tt, gsl, intensity] = parsl_run_opt(new_cfg_path)
                    else:
                        print(f"Run {run_name} already exists.")


                for (tt, gsl, intensity), v in vals.items():
                    val = v.result()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    args = parser.parse_args()
    cfg_path = args.config

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # from adept import ergoExo
    from adept import utils as adept_utils

    # from ml4tpd import TPDModule

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx

    # with open(f"./{cfg_path}", "r") as fi:
    #     cfg = yaml.safe_load(fi)

    # mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
    #     with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
    #         with open(os.path.join(td, "config.yaml"), "w") as fi:
    #             yaml.dump(cfg, fi)
    #         mlflow.log_artifacts(td)
    #     adept_utils.log_params(cfg)

    # parent_run_id = mlflow_run.info.run_id
    # adept_utils.export_run(parent_run_id)

    # with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
    scan_loop(cfg_path, shape="arbitrary")
