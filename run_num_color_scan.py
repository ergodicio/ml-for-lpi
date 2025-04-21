import logging, os
from utils import setup_parsl


import parsl
from parsl import python_app

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_parsl(run_ids, cfg_paths, parent_run_ids):
    def run_once(run_id, _cfg_path_):
        import yaml, mlflow

        from adept import ergoExo

        from ml4tpd import TPDModule

        with open(_cfg_path_, "r") as fi:
            _cfg = yaml.safe_load(fi)

        exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
        modules = exo.setup(_cfg, adept_module=TPDModule)
        run_output, ppo, _ = exo(modules)
        val = run_output[0]
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("ensemble_status", "completed")

        return val

    parsl_config = setup_parsl("local", 4, nodes=int(os.environ["SLURM_NNODES"]))
    run_once = python_app(run_once)

    vals = {}
    with parsl.load(parsl_config):

        for run_name in run_ids.keys():
            vals[run_name] = []
            for run_id in run_ids[run_name]:
                vals[run_name].append(run_once(run_id=run_id, _cfg_path_=cfg_paths[run_name]))

        for run_name in vals.keys():
            val = np.mean([v.result() for v in vals[run_name]])
            mlflow.log_metrics({"loss": float(val)}, step=0, run_id=parent_run_ids[run_name])
            mlflow.set_tag("ensemble_status", "completed")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run the parsl jobs")
    parser.add_argument("--num_ensembles", type=int, help="The number of ensembles to run")
    parser.add_argument("--start", type=int, help="The starting index for the ensembles")
    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import yaml, mlflow
    import numpy as np

    queued_runs_dir = os.path.join(os.getcwd(), "queued_runs")

    with open(os.path.join(queued_runs_dir, "run_ids.yaml"), "r") as fi:
        run_ids = yaml.safe_load(fi)
    with open(os.path.join(queued_runs_dir, "cfg_paths.yaml"), "r") as fi:
        cfg_paths = yaml.safe_load(fi)
    with open(os.path.join(queued_runs_dir, "parent_run_ids.yaml"), "r") as fi:
        parent_run_ids = yaml.safe_load(fi)

    # num_ensembles = 30
    # start = 12

    # needs deleting and rerunning
    # temperature=2000.0-gsl=200.0-intensity=4.00e+14-nc=48
    # temperature=2000.0-gsl=200.0-intensity=4.00e+14-nc=56

    # num_ensembles = 200
    # start = 200
    num_ensembles = args.num_ensembles
    start = args.start

    indices = slice(start, start + num_ensembles)

    _parent_run_ids, _cfg_paths, _run_ids = {}, {}, {}
    for k in list(parent_run_ids.keys())[indices]:
        _parent_run_ids[k] = parent_run_ids[k]
        _cfg_paths[k] = cfg_paths[k]
        _run_ids[k] = run_ids[k]

    run_parsl(_run_ids, _cfg_paths, _parent_run_ids)
