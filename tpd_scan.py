import logging, os
from uuid import uuid4
from ml4tpd.parsl_utils import setup_parsl


import parsl
from parsl import python_app

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def scan_loop(cfg):
    intensities = np.linspace(3.7e14, 4.6e14, 16)
    parsl_config = setup_parsl("local", 4, nodes=4, walltime="01:00:00")

    def run_once(_cfg_path):
        import yaml

        from adept import ergoExo
        from ml4tpd import TPDModule

        with open(_cfg_path, "r") as fi:
            _cfg = yaml.safe_load(fi)

        exo = ergoExo()
        modules = exo.setup(_cfg, adept_module=TPDModule)
        # diff_modules, static_modules = {}, {}
        # diff_modules["laser"], static_modules["laser"] = eqx.partition(
        #     modules["laser"], modules["laser"].get_partition_spec()
        # )
        run_output, ppo, _ = exo(modules)
        val = run_output[0]

        return val

    run_once = python_app(run_once)
    runs = []
    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
        with parsl.load(parsl_config):
            for intensity in intensities:
                # with open(_cfg_path, "r") as fi:
                #     orig_cfg = yaml.safe_load(fi)

                # orig_cfg["drivers"]["E0"]["num_colors"] = int(2**i)
                cfg["units"]["laser intensity"] = f"{intensity:.2e} W/cm^2"
                cfg["mlflow"]["run"] = f"param2-intensity={intensity:.2e}"

                with open(new_cfg_path := os.path.join(_td, f"{str(uuid4())}.yaml"), "w") as fi:
                    yaml.dump(cfg, fi)

                # with mlflow.start_run(nested=True, run_name=f"{intensity:.1e}") as nested_run:
                #     pass

                runs.append(run_once(_cfg_path=new_cfg_path))

                # mlflow.log_artifacts(_td, run_id=nested_run.info.run_id)
                # loss = float(val)
                # mlflow.log_metrics({"loss": loss}, step=i, run_id=parent_run_id)

            _ = [r.result() for r in runs]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    args = parser.parse_args()
    cfg_path = args.config

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx

    with open(f"./{cfg_path}", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["mlflow"]["experiment"] = "threshold-scan"
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    temperatures = 4000.0
    gradient_scale_lengths = 200.0

    cfg["drivers"]["E0"]["num_colors"] = 1
    cfg["units"]["reference electron temperature"] = f"{temperatures} eV"
    cfg["density"]["gradient scale length"] = f"{gradient_scale_lengths} um"

    scan_loop(cfg)
