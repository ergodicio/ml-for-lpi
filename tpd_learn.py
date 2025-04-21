import logging, os, dill as pickle


logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def train_model(_cfg_path):
    def run_one_val_and_grad(parent_run_id, _run_cfg_path, export=False):
        import os, yaml, mlflow
        from equinox import partition

        if "BASE_TEMPDIR" in os.environ:
            BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
        else:
            BASE_TEMPDIR = None

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        from adept import ergoExo
        from ml4tpd import TPDModule

        with open(_run_cfg_path, "r") as fi:
            _run_cfg = yaml.safe_load(fi)

        exo = ergoExo(parent_run_id=parent_run_id, mlflow_nested=True)
        modules = exo.setup(_run_cfg, adept_module=TPDModule)
        diff_modules, static_modules = {}, {}
        diff_modules["laser"], static_modules["laser"] = partition(
            modules["laser"], modules["laser"].get_partition_spec()
        )
        val, grad, (sol, ppo, _) = exo.val_and_grad(
            diff_modules, args={"static_modules": static_modules}, export=export
        )

        return val, grad

    from utils import setup_parsl
    from adept import ergoExo, utils as adept_utils

    from ml4tpd import TPDModule
    import parsl
    from parsl import python_app
    from itertools import product

    import jax
    from jax.flatten_util import ravel_pytree

    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    import optax

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx

    with open(f"{_cfg_path}", "r") as fi:
        cfg = yaml.safe_load(fi)

    with open(f"{_cfg_path}", "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    parsl_config = setup_parsl(
        cfg["parsl"]["provider"], 4, nodes=cfg["parsl"]["nodes"], walltime=cfg["parsl"]["walltime"]
    )
    run_one_val_and_grad = python_app(run_one_val_and_grad)

    # cfg["drivers"]["E0"]["params"]["key"] = np.random.randint(0, 2**10)
    # cfg["mlflow"]["run"] = f"{cfg['mlflow']['run']}-{cfg['drivers']['E0']['params']['key']}"
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    exo = ergoExo()
    modules = exo.setup(cfg, adept_module=TPDModule)
    diff_params, static_params = eqx.partition(modules["laser"], modules["laser"].get_partition_spec())

    lr_sched = optax.cosine_decay_schedule(
        init_value=cfg["opt"]["learning_rate"], decay_steps=cfg["opt"]["decay_steps"]
    )

    temperatures = np.linspace(2000, 4000, 5)
    gradient_scale_lengths = np.linspace(200, 600, 5)
    intensities = np.linspace(1e14, 1e15, 8)

    all_training_data = list(product(temperatures, gradient_scale_lengths, intensities))
    num_batches = len(all_training_data) // cfg["opt"]["batch_size"]

    with mlflow.start_run(run_id=exo.mlflow_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=lr_sched)
        opt_state = opt.init(eqx.filter(diff_params, eqx.is_array))  # initialize the optimizer state

        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)

            os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)  # create a directory for model history
            with parsl.load(parsl_config):
                for i in range(200):  # 1000 epochs
                    np.random.shuffle(all_training_data)
                    epoch_loss = []
                    epoch_gradnorm = []
                    for j in range(num_batches):
                        with open(opt_state_path := os.path.join(td, f"opt-state-epoch={i}-batch-{j}.pkl"), "wb") as fi:
                            pickle.dump(opt_state, fi)

                        modules["laser"].save(
                            module_path := os.path.join(td, "weights-history", f"weights-e{i:02d}-b{j:02d}.eqx")
                        )

                        mlflow.log_artifact(opt_state_path)
                        mlflow.log_artifact(module_path)

                        step = i * num_batches + j
                        training_data = all_training_data[
                            j * cfg["opt"]["batch_size"] : (j + 1) * cfg["opt"]["batch_size"]
                        ]

                        orig_cfg["drivers"]["E0"]["file"] = module_path

                        val_and_grads = []
                        for k in range(cfg["opt"]["batch_size"]):
                            _training_data = training_data[k]
                            run_name = f"epoch-{i}-batch-{j}-sim-{k}"
                            print(f"{i=}, {j=}, {k=} -- _Training Data: {_training_data}")
                            export = np.random.choice([True, False], p=[0.25, 0.75])  # if j % 1 == 0 else False
                            orig_cfg["units"]["reference electron temperature"] = f"{_training_data[0]:.3f} eV"
                            orig_cfg["density"]["gradient scale length"] = f"{_training_data[1]:.3f} um"
                            orig_cfg["units"]["laser intensity"] = f"{_training_data[2]:.2e} W/cm^2"
                            orig_cfg["grid"]["dt"] = f"{np.random.uniform(1, 3):.3f} fs"
                            orig_cfg["mlflow"]["run"] = run_name
                            orig_cfg["mlflow"]["export"] = str(export)

                            with open(run_cfg_path := os.path.join(td, f"config-{i=}-{j=}-{k=}.yaml"), "w") as fi:
                                yaml.dump(orig_cfg, fi)

                            val_and_grads.append(
                                run_one_val_and_grad(
                                    parent_run_id=mlflow_run.info.run_id, _run_cfg_path=run_cfg_path, export=export
                                )
                            )

                        vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                        val = np.mean([v for v, _ in vgs])  # get the mean of the loss values

                        avg_grad = adept_utils.all_reduce_gradients([g for _, g in vgs], cfg["opt"]["batch_size"])

                        flat_grad, _ = ravel_pytree(avg_grad["laser"])
                        mlflow.log_metrics(
                            {"batch grad norm": float(np.linalg.norm(flat_grad)), "batch loss": float(val)}, step=step
                        )
                        updates, opt_state = opt.update(avg_grad["laser"], opt_state, diff_params)
                        diff_params = eqx.apply_updates(diff_params, updates)
                        modules["laser"] = eqx.combine(diff_params, static_params)
                        epoch_loss.append(val)
                        epoch_gradnorm.append(np.linalg.norm(flat_grad))

                    mlflow.log_metrics(
                        {
                            "epoch loss": (epoch_loss_mean := float(np.mean(epoch_loss))),
                            "epoch grad norm": float(np.mean(epoch_gradnorm)),
                        },
                        step=i,
                    )

    return epoch_loss_mean


# def train_once(params):


if __name__ == "__main__":
    import argparse, mlflow

    parser = argparse.ArgumentParser(description="Run TPD training.")
    parser.add_argument("--config", type=str, help="The config file")
    parser.add_argument("--run_id", type=str, help="An MLFlow RunID")
    args = parser.parse_args()

    if args.run_id is not None:
        run_id = args.run_id
        cfg_path = os.path.join(mlflow.get_run(run_id).info.artifact_uri, "config.yaml")
    else:
        cfg_path = args.config

    train_model(cfg_path)
