from itertools import product
import argparse, os, time, yaml, mlflow, tempfile, numpy as np


if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def load_and_make_folders(cfg_path: str, _hparams) -> str:
    """
    This is used to queue runs on NERSC

    Args:
        cfg_path:

    Returns:

    """

    with open(cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    experiment = cfg["mlflow"]["experiment"]
    run_name = f"{_hparams['decoder_width']}x{_hparams['decoder_depth']}_{_hparams['activation']}_{_hparams['learning_rate']:0.4f}"
    cfg["mlflow"]["run"] = run_name

    mlflow.set_experiment(experiment)

    cfg["opt"]["learning_rate"] = float(_hparams["learning_rate"])
    cfg["drivers"]["E0"]["params"]["nn"]["decoder_width"] = int(_hparams["decoder_width"])
    cfg["drivers"]["E0"]["params"]["nn"]["decoder_depth"] = int(_hparams["decoder_depth"])
    cfg["drivers"]["E0"]["params"]["nn"]["activation"] = str(_hparams["activation"])
    cfg["drivers"]["E0"]["params"]["nn"]["key"] = int(np.random.randint(0, 2**10))

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, f"config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            mlflow.log_artifacts(td)

    return mlflow_run.info.run_id


def _queue_run_(run_id):

    with open("nersc-gpu-base.sh", "r") as fh:
        base_job = fh.read()

    with open(os.path.join(os.getcwd(), "new_job.sh"), "w") as job_file:
        job_file.write(base_job + "\n")
        job_file.writelines(f"python3 tpd_learn.py --run_id {run_id}")

    os.system(f"sbatch new_job.sh")
    time.sleep(0.25)
    os.system("sqs")
    os.system("rm new_job.sh")


if __name__ == "__main__":

    learning_rates = 10 ** np.random.uniform(-2.5, -1, 3)  # (-4, -2, 3)
    decoder_widths = [16, 64, 128]
    decoder_depths = [3, 4, 5]
    activations = ["tanh", "leaky_relu"]

    all_hps = product(learning_rates, decoder_widths, decoder_depths, activations)
    all_hps = list(all_hps)
    np.random.shuffle(all_hps)

    for lr, dw, dd, act in all_hps[::3]:
        hparams = {"learning_rate": lr, "decoder_width": dw, "decoder_depth": dd, "activation": act}
        run_id = load_and_make_folders("configs/tpd-learn.yaml", hparams)
        _queue_run_(run_id)
