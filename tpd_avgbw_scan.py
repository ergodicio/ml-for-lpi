import logging, os
from utils import setup_parsl


import parsl
from parsl import python_app
from itertools import product
from uuid import uuid4

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def scan_loop(_cfg_path):

    def run_once(_cfg_path_):
        import yaml

        from adept import ergoExo

        from ml4tpd import TPDModule

        with open(_cfg_path_, "r") as fi:
            _cfg = yaml.safe_load(fi)

        exo = ergoExo()
        modules = exo.setup(_cfg, adept_module=TPDModule)
        run_output, ppo, _ = exo(modules)
        val = run_output[0]
        mlflow.log_metrics({"loss": float(val)}, step=0, run_id=exo.mlflow_run_id)
        return val

    temperatures = np.linspace(2000, 4000, 5)
    gradient_scale_lengths = np.linspace(200, 600, 4)
    intensities = np.linspace(1e14, 1e15, 4)

    all_hps = product(temperatures, gradient_scale_lengths, intensities)
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)
    orig_cfg["mlflow"]["experiment"] = "optavg-64lines"
    parsl_config = setup_parsl(orig_cfg["parsl"]["provider"], 4, nodes=orig_cfg["parsl"]["nodes"])
    run_once = python_app(run_once)

    vals = []
    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
        with parsl.load(parsl_config):
            for tt, gsl, intensity in list(all_hps):
                run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
                orig_cfg["mlflow"]["run"] = run_name

                mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
                orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
                orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
                orig_cfg["density"]["gradient scale length"] = f"{gsl} um"
                orig_cfg["drivers"]["E0"]["shape"] = "arbitrary"
                orig_cfg["drivers"]["E0"][
                    "file"
                ] = f"s3://public-ergodic-continuum/181417/f2215c835d3848adbd582cc58f2ccbd4/artifacts/weights-e14-b07.eqx"

                with open(new_cfg_path := os.path.join(td, f"config-{str(uuid4())}.yaml"), "w") as fi:
                    yaml.dump(orig_cfg, fi)

                vals.append(run_once(_cfg_path_=new_cfg_path))

            val = np.mean([v.result() for v in vals])  # get the mean of the loss values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    args = parser.parse_args()
    cfg_path = args.config

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx

    scan_loop(cfg_path)
