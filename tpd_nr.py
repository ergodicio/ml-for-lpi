import logging, os


logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_one_val_and_grad(run_id, cfg_path):
    import os, yaml

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from adept import ergoExo
    from adept.utils import export_run
    from ml4tpd import TPDModule

    with open(cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
    modules = exo.setup(cfg, adept_module=TPDModule)
    val, grad, (sol, ppo, _) = exo.val_and_grad(modules)
    export_run(run_id)

    return val, grad


if __name__ == "__main__":
    import uuid
    from copy import deepcopy
    from utils import setup_parsl
    from adept import ergoExo
    from adept import utils as adept_utils
    from ml4tpd import TPDModule
    import parsl
    from parsl import python_app

    logging.basicConfig(filename=f"runlog-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    import jax
    from jax.flatten_util import ravel_pytree

    jax.config.update("jax_platform_name", "cpu")

    import optax

    parsl_config = setup_parsl("local", num_gpus=4, nodes=1)
    run_one_val_and_grad = python_app(run_one_val_and_grad)

    import yaml, mlflow, tempfile, os
    import numpy as np, equinox as eqx

    with open(f"./configs/tpd-nr.yaml", "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg["drivers"]["E0"]["params"]["key"] = np.random.randint(0, 2**10)
    cfg["mlflow"]["run"] = f"{cfg['mlflow']['run']}-{cfg['drivers']['E0']['params']['key']}"
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        adept_utils.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    adept_utils.export_run(parent_run_id)

    orig_cfg = deepcopy(cfg)

    exo = ergoExo()
    modules = exo.setup(cfg, adept_module=TPDModule)

    lr_sched = optax.cosine_decay_schedule(
        init_value=cfg["opt"]["learning_rate"], decay_steps=cfg["opt"]["decay_steps"]
    )
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=lr_sched)
        opt_state = opt.init(eqx.filter(modules["laser"], eqx.is_array))  # initialize the optimizer state

        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)  # create a directory for model history
            with parsl.load(parsl_config):
                for i in range(64):  # 1000 epochs
                    module_path = os.path.join(td, "weights-history", f"weights-e{i:02d}.eqx")
                    modules["laser"].save(module_path)
                    orig_cfg["drivers"]["E0"]["file"] = module_path

                    with open(cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
                        yaml.dump(orig_cfg, fi)

                    if cfg["opt"]["batch_size"] == 1:
                        with mlflow.start_run(nested=True, run_name=f"epoch-{i}") as nested_run:
                            pass
                        val, avg_grad = run_one_val_and_grad(run_id=nested_run.info.run_id, cfg_path=cfg_path)
                    else:
                        val_and_grads = []
                        for j in range(cfg["opt"]["batch_size"]):
                            with mlflow.start_run(nested=True, run_name=f"epoch-{i}-sim-{j}") as nested_run:
                                mlflow.log_artifact(module_path)
                                # val, grad = run_one_val_and_grad(cfg, run_id=nested_run.info.run_id).result()
                                val_and_grads.append(
                                    run_one_val_and_grad(run_id=nested_run.info.run_id, cfg_path=cfg_path)
                                )

                        vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                        val = np.mean([v for v, _ in vgs])  # get the mean of the loss values

                        avg_grad = adept_utils.all_reduce_gradients([g for _, g in vgs], cfg["opt"]["batch_size"])

                    grad_bandwidth = avg_grad["laser"]
                    flat_grad, _ = ravel_pytree(grad_bandwidth)
                    mlflow.log_metrics({"grad norm": float(np.linalg.norm(flat_grad))}, step=i)
                    mlflow.log_metrics({"loss": float(val)}, step=i)
                    if i % 10 == 0:
                        mlflow.log_artifacts(td)
                    adept_utils.export_run(parent_run_id, prefix="parent", step=i)
                    updates, opt_state = opt.update(grad_bandwidth, opt_state, modules["laser"])
                    modules["laser"] = eqx.apply_updates(modules["laser"], updates)
