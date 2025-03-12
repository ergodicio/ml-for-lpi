from itertools import product
import argparse, os, time, yaml, mlflow, tempfile, numpy as np
from uuid import uuid4

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None

QUEUE_DIR = "/pscratch/sd/a/archis/queue-configs"


def load_and_make_folders(base_cfg_path: str, _tt, _gsl, _intensity) -> str:
    """
    This is used to queue runs on NERSC

    Args:
        cfg_path:

    Returns:

    """

    with open(base_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    experiment = cfg["mlflow"]["experiment"]
    run_name = f"temperature={_tt:.1f}-gsl={_gsl:.1f}-intensity={_intensity:.2e}"
    cfg["mlflow"]["run"] = run_name

    mlflow.set_experiment(experiment)

    cfg["units"]["reference electron temperature"] = f"{_tt} eV"
    cfg["units"]["laser intensity"] = f"{_intensity} W/cm^2"
    cfg["density"]["gradient scale length"] = f"{_gsl} um"

    _cfg_path = os.path.join(QUEUE_DIR, f"{str(uuid4())}.yaml")
    with open(_cfg_path, "w") as fi:
        yaml.dump(cfg, fi)

    return _cfg_path


def _queue_run_(_cfg_path):

    with open("nersc-gpu-shared-base.sh", "r") as fh:
        base_job = fh.read()

    with open(os.path.join(os.getcwd(), "new_job.sh"), "w") as job_file:
        job_file.write(base_job + "\n")
        job_file.writelines(f"python3 tpd_opt.py --config {_cfg_path}")

    os.system(f"sbatch new_job.sh")
    time.sleep(0.25)


if __name__ == "__main__":

    temperatures = np.linspace(2000, 4000, 5)
    gradient_scale_lengths = np.linspace(200, 600, 4)
    intensities = np.linspace(1e14, 1e15, 4)

    all_hps = product(temperatures, gradient_scale_lengths, intensities)
    all_hps = list(all_hps)
    # np.random.shuffle(all_hps)

    for tt, gsl, intensity in all_hps:
        cfg_path = load_and_make_folders("configs/tpd-opt.yaml", tt, gsl, intensity)
        _queue_run_(cfg_path)
    os.system("sqs")
    os.system("rm new_job.sh")
