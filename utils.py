from typing import Tuple
import pickle, shutil, time, os
from optax import OptState


def robust_rmtree(directory, retries=5, delay=5):
    for attempt in range(retries):
        try:
            shutil.rmtree(directory)
            print(f"Successfully removed {directory}")
            break
        except OSError as e:
            # if e.errno == 5:  # Input/output error
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)  # Wait before retrying

    else:
        print(f"Failed to remove {directory} after {retries} attempts.")


def setup_parsl(parsl_provider="local", num_gpus=4, nodes=1, walltime="00:30:00"):
    from parsl.config import Config
    from parsl.providers import SlurmProvider, LocalProvider
    from parsl.launchers import SrunLauncher
    from parsl.executors import HighThroughputExecutor

    if parsl_provider == "local":
        if nodes == 1:
            this_provider = LocalProvider
            provider_args = dict(
                worker_init="source /pscratch/sd/a/archis/venvs/ml-for-lpi/bin/activate; \
                        export PYTHONPATH=$PYTHONPATH:/global/homes/a/archis/ml-for-lpi; \
                        export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                        export MLFLOW_TRACKING_URI='https://continuum.ergodic.io/experiments/'",
                init_blocks=1,
                max_blocks=1,
                nodes_per_block=1,
            )
            htex = HighThroughputExecutor(
                available_accelerators=num_gpus,
                label="tpd",
                provider=this_provider(**provider_args),
                cpu_affinity="block",
            )
            print(f"{htex.workers_per_node=}")
        else:
            this_provider = LocalProvider
            provider_args = dict(
                worker_init="source /pscratch/sd/a/archis/venvs/ml-for-lpi/bin/activate; \
                        export PYTHONPATH=$PYTHONPATH:/global/homes/a/archis/ml-for-lpi; \
                        export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                        export MLFLOW_TRACKING_URI='https://continuum.ergodic.io/experiments/'",
                nodes_per_block=nodes,
                launcher=SrunLauncher(overrides="-c 32 --gpus-per-node 4"),
                cmd_timeout=120,
                init_blocks=1,
                max_blocks=1,
            )

            htex = HighThroughputExecutor(
                available_accelerators=num_gpus * nodes,
                label="tpd",
                provider=this_provider(**provider_args),
                max_workers_per_node=4,
                cpu_affinity="block",
            )
            print(f"{htex.workers_per_node=}")

    elif parsl_provider == "gpu":

        this_provider = SlurmProvider
        sched_args = ["#SBATCH -C gpu", "#SBATCH --qos=regular"]
        provider_args = dict(
            partition=None,
            account="m4490_g",
            scheduler_options="\n".join(sched_args),
            worker_init="export SLURM_CPU_BIND='cores';\
                    export PYTHONPATH=$PYTHONPATH:/global/homes/a/archis/ml-for-lpi; \
                    source /pscratch/sd/a/archis/venvs/ml-for-lpi/bin/activate; \
                    export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'; \
                    export MLFLOW_TRACKING_URI='https://continuum.ergodic.io/experiments/'",
            launcher=SrunLauncher(overrides="--gpus-per-node 4 -c 128"),
            walltime=walltime,
            cmd_timeout=120,
            nodes_per_block=1,
            # init_blocks=1,
            max_blocks=nodes,
        )

        htex = HighThroughputExecutor(
            available_accelerators=4, label="tpd-learn", provider=this_provider(**provider_args), cpu_affinity="block"
        )
        print(f"{htex.workers_per_node=}")

    return Config(executors=[htex], retries=4)


def get_checkpoint(resume_dict) -> Tuple[str, OptState]:
    import pickle, tempfile
    from adept import utils as adept_utils

    base_path = (
        "s3://public-ergodic-continuum/"
        + str(resume_dict["experiment_id"])
        + "/"
        + resume_dict["run_id"]
        + "/artifacts/"
    )
    resume_epoch = resume_dict["epoch"]
    with tempfile.TemporaryDirectory() as td:
        target_path = td + "/opt_state.pkl"
        opt_state = adept_utils.download_from_s3(base_path + f"opt_state-e{resume_epoch:02d}-b{0:02d}.pkl", target_path)

        opt_state = renamed_load(open(opt_state, "rb"))

    weights_path = base_path + f"weights-e{resume_epoch:02d}-b{0:02d}.eqx"

    return weights_path, opt_state


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "adept.lpse2d.modules.driver":
            renamed_module = "adept._lpse2d.modules.driver"
        elif module == "adept.lpse2d.modules.nn.driver":
            renamed_module = "adept._lpse2d.modules.nn.driver"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
