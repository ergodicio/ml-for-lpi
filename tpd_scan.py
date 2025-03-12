import logging, os
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_once(run_id, _cfg_path):
    import yaml

    from adept import ergoExo

    # from adept.utils import export_run
    from ml4tpd import TPDModule

    with open(_cfg_path, "r") as fi:
        _cfg = yaml.safe_load(fi)

    exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
    modules = exo.setup(_cfg, adept_module=TPDModule)
    # diff_modules, static_modules = {}, {}
    # diff_modules["laser"], static_modules["laser"] = eqx.partition(
    #     modules["laser"], modules["laser"].get_partition_spec()
    # )
    run_output, ppo, _ = exo(modules)
    val = run_output[0]

    return val


def scan_loop(parent_run_id, _cfg_path):

    base_dt = np.random.uniform(1, 3)
    for i in range(7):  # 1000 epochs
        from adept.utils import export_run

        with open(_cfg_path, "r") as fi:
            orig_cfg = yaml.safe_load(fi)

        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:

            # dt = np.random.uniform(0.97, 1.03) * base_dt / 2**i
            # orig_cfg["grid"]["dt"] = f"{dt:.3f}fs"
            # print(orig_cfg["grid"]["dt"])

            # orig_cfg["drivers"]["E0"]["num_colors"] = int(2**i)

            with open(new_cfg_path := os.path.join(_td, "config.yaml"), "w") as fi:
                yaml.dump(orig_cfg, fi)

            with mlflow.start_run(nested=True, run_name=f"phase-{i}") as nested_run:
                pass

            # Capture the output into a file
            output_file = os.path.join(_td, f"stdout_stderr.txt")
            with open(output_file, "w") as f:
                with redirect_stdout(f), redirect_stderr(f):
                    val = run_once(run_id=nested_run.info.run_id, _cfg_path=new_cfg_path)

            mlflow.log_artifacts(_td, run_id=nested_run.info.run_id)
            # export_run(nested_run.info.run_id)
            loss = float(val)
            mlflow.log_metrics({"loss": loss}, step=i, run_id=parent_run_id)
            # adept_utils.export_run(parent_run_id, prefix="parent", step=i)

    return loss


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

    with open(f"./{cfg_path}", "r") as fi:
        cfg = yaml.safe_load(fi)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        adept_utils.log_params(cfg)

        parent_run_id = mlflow_run.info.run_id
        # adept_utils.export_run(parent_run_id)

        # with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        scan_loop(parent_run_id, cfg_path)
