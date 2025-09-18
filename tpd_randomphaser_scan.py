import logging, os
from ml4tpd.parsl_utils import setup_parsl


import parsl
from parsl import python_app
from itertools import product

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def scan_loop(_cfg_path):

    def run_once(run_id, _cfg_path):
        import yaml, mlflow

        from adept import ergoExo

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

    temperatures = np.linspace(2000, 4000, 5)
    gradient_scale_lengths = np.linspace(200, 600, 4)
    intensities = np.linspace(1e14, 1e15, 4)

    all_hps = product(temperatures, gradient_scale_lengths, intensities)
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)
    orig_cfg["mlflow"]["experiment"] = "randomphaser-64lines"
    opt_runs = mlflow.search_runs(
        experiment_names=["opt-64lines"], filter_string="attribute.run_name LIKE '%temperature%'"
    )

    parsl_config = setup_parsl(orig_cfg["parsl"]["provider"], 4, nodes=orig_cfg["parsl"]["nodes"])
    run_once = python_app(run_once)

    with parsl.load(parsl_config):
        for tt, gsl, intensity in list(all_hps)[61:62]:
            run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
            orig_cfg["mlflow"]["run"] = run_name
            print(f"{run_name=}")

            mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])
            orig_cfg["drivers"]["E0"]["shape"] = "uniform"
            orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
            orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
            orig_cfg["density"]["gradient scale length"] = f"{gsl} um"

            # run = opt_runs[
            #     (opt_runs["params.units.reference electron temperature"] == f"{tt:.1f} eV")
            #     & (opt_runs["params.units.laser intensity"] == f"{intensity:.2e} W/cm^2")
            #     & (opt_runs["params.density.gradient scale length"] == f"{gsl:.1f} um")
            # ]
            parent_runs = mlflow.search_runs(
                experiment_names=["opt-64lines"],
                # filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                # experiment_names=["opt-64lines"],
                filter_string=f"attributes.run_name LIKE '{run_name}'",
            )
            not_already_found = True
            for parent_run_id in parent_runs["run_id"].values:
                if not_already_found:
                    child_runs = mlflow.search_runs(
                        experiment_names=["opt-64lines"],
                        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                        # experiment_names=["opt-64lines"],
                        # filter_string=f"attributes.run_name LIKE '{run_name}'",
                    )

                    if len(child_runs) == 0:
                        print("--------------------")
                        print("Temperature: ", tt, "GSL: ", gsl, "Intensity: ", intensity)
                        print(f"Parent run {parent_run_id} has no child runs")
                        print("--------------------")

                    else:
                        not_already_found = False

            child_run = child_runs.sort_values("start_time", ascending=False).iloc[0]
            with mlflow.start_run(run_name=run_name) as parent_run:
                with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
                    with open(os.path.join(_td, "config.yaml"), "w") as fi:
                        yaml.dump(orig_cfg, fi)
                    mlflow.log_artifacts(_td)
                    adept_utils.log_params(orig_cfg)

                    vals, run_ids = [], []
                    for i in range(8):
                        orig_cfg["drivers"]["E0"]["shape"] = "random_phaser"
                        orig_cfg["drivers"]["E0"][
                            "file"
                        ] = f"s3://public-ergodic-continuum/181417/{child_run.run_id}/artifacts/laser.eqx"

                        with open(new_cfg_path := os.path.join(_td, f"config-{i}.yaml"), "w") as fi:
                            yaml.dump(orig_cfg, fi)

                        with mlflow.start_run(nested=True, run_name=f"phase-{i}") as nested_run:
                            pass

                        vals.append(run_once(run_id=nested_run.info.run_id, _cfg_path=new_cfg_path))
                        run_ids.append(nested_run.info.run_id)

                    val = np.mean([v.result() for v in vals])  # get the mean of the loss values

                    mlflow.log_metrics({"loss": float(val)}, step=0)


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
    scan_loop(cfg_path)
