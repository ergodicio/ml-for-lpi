import logging, os, uuid
from utils import setup_parsl


import parsl
from parsl import python_app
from itertools import product

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def queue_runs(_cfg_path):

    # open remaining_hyperparameters.txt
    # each row is a hyperparameter set
    # temp, gsl, intensity, nc = [float(x) for x in line.split()]
    # read line by line
    with open("notebooks/remaining_hyperparameters.txt", "r") as fi:
        lines = fi.readlines()

    # parse lines
    all_hps = []
    for line in lines:
        temp, gsl, intensity, nc = [float(x) for x in line.split()]
        all_hps.append((temp, gsl, intensity, int(nc)))

    # print(temperatures, gradient_scale_lengths, intensities, num_colors)

    # raise ValueError
    # temperatures = np.linspace(2000, 4000, 5)
    # gradient_scale_lengths = np.linspace(200, 600, 4)
    # intensities = np.linspace(1e14, 1e15, 4)
    # num_colors = np.linspace(16, 128, 15)

    # all_hps = list(product(temperatures, gradient_scale_lengths, intensities, num_colors))
    # all_hps = all_hps[18:600]
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    orig_cfg["mlflow"]["experiment"] = "num-color-scan"

    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])

    run_ids, cfg_paths, parent_run_ids = {}, {}, {}

    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
        for tt, gsl, intensity, nc in all_hps:
            run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}-nc={int(nc)}"

            run_ids[run_name] = []
            orig_cfg["drivers"]["E0"]["shape"] = "uniform"
            orig_cfg["drivers"]["E0"]["num_colors"] = int(nc)
            orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
            orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
            orig_cfg["density"]["gradient scale length"] = f"{gsl} um"

            with open(
                new_cfg_path := os.path.join(BASE_TEMPDIR, "queue-configs", f"{str(uuid.uuid4())}.yaml"), "w"
            ) as fi:
                yaml.dump(orig_cfg, fi)

            cfg_paths[run_name] = new_cfg_path
            with mlflow.start_run(run_name=run_name) as parent_run:
                mlflow.set_tag("ensemble_status", "queued")

                with open(os.path.join(_td, "config.yaml"), "w") as fi:
                    yaml.dump(orig_cfg, fi)
                mlflow.log_artifacts(_td)
                adept_utils.log_params(orig_cfg)

                parent_run_ids[run_name] = parent_run.info.run_id

                for i in range(8):
                    with mlflow.start_run(nested=True, run_name=f"phase-{i}") as nested_run:
                        pass
                        mlflow.set_tag("ensemble_status", "queued")
                    run_ids[run_name].append(nested_run.info.run_id)

    return run_ids, cfg_paths, parent_run_ids


def run_parsl(run_ids, cfg_paths, parent_run_ids):
    def run_once(run_id, _cfg_path_):
        import yaml

        from adept import ergoExo

        from ml4tpd import TPDModule

        with open(_cfg_path_, "r") as fi:
            _cfg = yaml.safe_load(fi)

        exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
        modules = exo.setup(_cfg, adept_module=TPDModule)
        run_output, ppo, _ = exo(modules)
        val = run_output[0]

        return val

    parsl_config = setup_parsl("local", 4, nodes=os.environ["SLURM_NNODES"])
    run_once = python_app(run_once)

    vals = {}
    with parsl.load(parsl_config):
        for run_name in run_ids.keys():
            vals[run_name].append(run_once(run_id=run_ids[run_name], _cfg_path=cfg_paths[run_name]))

        for run_name in vals.keys():
            val = np.mean([v.result() for v in vals[run_name]])
            mlflow.log_metrics({"loss": float(val)}, step=0, run_id=parent_run_ids[run_name])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    args = parser.parse_args()
    cfg_path = args.config

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from adept import utils as adept_utils
    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx

    queue_config_dir = os.path.join(BASE_TEMPDIR, "queue-configs")
    os.makedirs(queue_config_dir, exist_ok=True)
    run_ids, cfg_paths, parent_run_ids = queue_runs(cfg_path)
    queued_runs_dir = os.path.join(os.getcwd(), "queued_runs")
    os.makedirs(queued_runs_dir, exist_ok=True)

    with open(os.path.join(queued_runs_dir, "run_ids.yaml"), "w") as fi:
        yaml.dump(run_ids, fi)
    with open(os.path.join(queued_runs_dir, "cfg_paths.yaml"), "w") as fi:
        yaml.dump(cfg_paths, fi)
    with open(os.path.join(queued_runs_dir, "parent_run_ids.yaml"), "w") as fi:
        yaml.dump(parent_run_ids, fi)
