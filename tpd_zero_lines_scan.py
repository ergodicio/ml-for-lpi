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


def scan_loop(_cfg_path):

    def run_once(_cfg_path):
        import yaml
        from adept import ergoExo
        from ml4tpd import TPDModule

        with open(_cfg_path, "r") as fi:
            _cfg = yaml.safe_load(fi)

        exo = ergoExo()
        modules = exo.setup(_cfg, adept_module=TPDModule)
        run_output, ppo, _ = exo(modules)
        val = run_output[0]

        return val

    temperatures = np.linspace(2000, 4000, 5)
    gradient_scale_lengths = np.linspace(200, 600, 4)
    intensities = np.linspace(1e14, 1e15, 4)
    thresholds = np.linspace(1e-2, 1e-1, 11)

    all_hps = list(product(temperatures, gradient_scale_lengths, intensities, thresholds))

    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)
    orig_cfg["mlflow"]["experiment"] = "zerolines-64lines"
    opt_runs = mlflow.search_runs(
        experiment_names=["opt-64lines"], filter_string="attribute.run_name LIKE '%temperature%'"
    )
    orig_cfg["parsl"]["nodes"] = 22

    parsl_config = setup_parsl(orig_cfg["parsl"]["provider"], 4, nodes=orig_cfg["parsl"]["nodes"])
    run_once = python_app(run_once)

    with parsl.load(parsl_config):
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
            vals = []
            for tt, gsl, intensity, threshold in all_hps:
                child_run_run_id = find_latest_child_run(tt, gsl, intensity)
                run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}-threshold={threshold:.1e}"
                orig_cfg["mlflow"]["run"] = run_name

                mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
                orig_cfg["drivers"]["E0"]["shape"] = "zero_lines"
                orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
                orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
                orig_cfg["density"]["gradient scale length"] = f"{gsl} um"
                orig_cfg["drivers"]["E0"]["intensity_threshold"] = float(threshold)

                orig_cfg["drivers"]["E0"][
                    "file"
                ] = f"s3://public-ergodic-continuum/181417/{child_run_run_id}/artifacts/laser.eqx"
                with open(new_cfg_path := os.path.join(_td, f"{str(uuid.uuid4())}.yaml"), "w") as fi:
                    yaml.dump(orig_cfg, fi)

                vals.append(run_once(_cfg_path=new_cfg_path))

            val = np.mean([v.result() for v in vals])


def find_latest_child_run(tt, gsl, intensity):
    orig_run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
    parent_runs = mlflow.search_runs(
        experiment_names=["opt-64lines"], filter_string=f"attributes.run_name LIKE '{orig_run_name}'"
    )
    not_already_found = True
    for parent_run_id in parent_runs["run_id"].values:
        if not_already_found:
            child_runs = mlflow.search_runs(
                experiment_names=["opt-64lines"], filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
            )

            if len(child_runs) == 0:
                print("--------------------")
                print("Temperature: ", tt, "GSL: ", gsl, "Intensity: ", intensity)
                print(f"Parent run {parent_run_id} has no child runs")
                print("--------------------")

            else:
                not_already_found = False

    child_run = child_runs.sort_values("start_time", ascending=False).iloc[0]
    return child_run.run_id  # get the mean of the loss values


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

    scan_loop(cfg_path)
