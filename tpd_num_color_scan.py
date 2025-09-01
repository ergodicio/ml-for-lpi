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

    def run_once(run_id, _cfg_path):
        import yaml

        from adept import ergoExo

        from ml4tpd import TPDModule

        with open(_cfg_path, "r") as fi:
            _cfg = yaml.safe_load(fi)

        exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
        modules = exo.setup(_cfg, adept_module=TPDModule)
        run_output, ppo, _ = exo(modules)
        val = run_output[0]

        return val

    temperatures = np.linspace(2000, 4000, 5)[-1:]
    gradient_scale_lengths = np.linspace(200, 600, 4)[-3:-2]
    intensities = np.linspace(3e14, 7e14, 5)
    num_colors = 2 ** np.linspace(1, 8, 8)

    all_hps = list(product(temperatures, gradient_scale_lengths, intensities, num_colors))
    # all_hps = all_hps[10:1000]
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    orig_cfg["mlflow"]["experiment"] = "num-color-and-intensity-scan-100ps"
    orig_cfg["parsl"]["nodes"] = 4  # int(os.environ["SLURM_NNODES"]) if "SLURM_NNODES" in os.environ else 1
    orig_cfg["parsl"]["provider"] = "local"

    parsl_config = setup_parsl(orig_cfg["parsl"]["provider"], 4, nodes=orig_cfg["parsl"]["nodes"], walltime="01:00:00")
    run_once = python_app(run_once)

    mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])

    parent_runs = {}

    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
        for tt, gsl, intensity, nc in all_hps:
            run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}-nc={int(nc)}"
            orig_cfg["mlflow"]["run"] = run_name
            orig_cfg["drivers"]["E0"]["shape"] = "uniform"
            orig_cfg["drivers"]["E0"]["num_colors"] = int(nc)
            orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
            orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
            orig_cfg["density"]["gradient scale length"] = f"{gsl} um"

            with mlflow.start_run(run_name=run_name) as parent_run:
                with open(os.path.join(_td, "config.yaml"), "w") as fi:
                    yaml.dump(orig_cfg, fi)
                mlflow.log_artifacts(_td)
                adept_utils.log_params(orig_cfg)

            parent_runs[run_name] = parent_run.info.run_id

        vals = {}
        with parsl.load(parsl_config):
            for tt, gsl, intensity, nc in all_hps:
                run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}-nc={int(nc)}"
                orig_cfg["drivers"]["E0"]["shape"] = "uniform"
                orig_cfg["drivers"]["E0"]["num_colors"] = int(nc)
                orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
                orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
                orig_cfg["density"]["gradient scale length"] = f"{gsl} um"
                vals[run_name] = []
                for i in range(1):
                    with open(new_cfg_path := os.path.join(_td, f"{str(uuid.uuid4())}.yaml"), "w") as fi:
                        yaml.dump(orig_cfg, fi)

                    with mlflow.start_run(
                        nested=True, run_name=f"phase-{i}", parent_run_id=parent_runs[run_name]
                    ) as nested_run:
                        pass

                    vals[run_name].append(run_once(run_id=nested_run.info.run_id, _cfg_path=new_cfg_path))

            for tt, gsl, intensity, nc in all_hps:
                run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}-nc={int(nc)}"
                val = np.mean([v.result() for v in vals[run_name]])
                mlflow.log_metrics({"loss": float(val)}, step=0, run_id=parent_runs[run_name])


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


# import logging, os, uuid
# from utils import setup_parsl


# import parsl
# from parsl import python_app
# from itertools import product

# logger = logging.getLogger(__name__)

# if "BASE_TEMPDIR" in os.environ:
#     BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
# else:
#     BASE_TEMPDIR = None


# def run_parsl(run_ids, cfg_paths, parent_run_ids):
#     def run_once(run_id, _cfg_path_):
#         import yaml

#         from adept import ergoExo

#         from ml4tpd import TPDModule

#         with open(_cfg_path_, "r") as fi:
#             _cfg = yaml.safe_load(fi)

#         exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
#         modules = exo.setup(_cfg, adept_module=TPDModule)
#         run_output, ppo, _ = exo(modules)
#         val = run_output[0]

#         return val

#     parsl_config = setup_parsl("local", 4, nodes=os.environ["SLURM_NNODES"])
#     run_once = python_app(run_once)

#     vals = {}
#     with parsl.load(parsl_config):

#         for run_name in run_ids.keys():
#             vals[run_name] = []
#             for run_id in run_ids[run_name]:
#                 vals[run_name].append(run_once(run_id=run_id, _cfg_path_=cfg_paths[run_name]))

#         for run_name in vals.keys():
#             val = np.mean([v.result() for v in vals[run_name]])
#             mlflow.log_metrics({"loss": float(val)}, step=0, run_id=parent_run_ids[run_name])


# if __name__ == "__main__":

#     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#     import yaml, mlflow
#     import numpy as np

#     queued_runs_dir = os.path.join(os.getcwd(), "queued_runs")

#     with open(os.path.join(queued_runs_dir, "run_ids.yaml"), "r") as fi:
#         run_ids = yaml.safe_load(fi)
#     with open(os.path.join(queued_runs_dir, "cfg_paths.yaml"), "r") as fi:
#         cfg_paths = yaml.safe_load(fi)
#     with open(os.path.join(queued_runs_dir, "parent_run_ids.yaml"), "r") as fi:
#         parent_run_ids = yaml.safe_load(fi)

#     run_parsl(run_ids, cfg_paths, parent_run_ids)
