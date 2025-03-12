import logging, os


logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_opt(_cfg_path):
    def run_one_val_and_grad(run_id, _run_cfg_path, export=False):
        import os, yaml, tempfile
        from contextlib import redirect_stdout, redirect_stderr
        import mlflow
        from equinox import partition

        if "BASE_TEMPDIR" in os.environ:
            BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
        else:
            BASE_TEMPDIR = None

        from jax import config

        # config.update("jax_enable_x64", True)

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        from adept import ergoExo
        from adept.utils import export_run
        from ml4tpd import TPDModule

        with open(_run_cfg_path, "r") as fi:
            cfg = yaml.safe_load(fi)

        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
            output_file = os.path.join(_td, f"stdout_stderr.txt")
            with open(output_file, "w") as f:
                with redirect_stdout(f), redirect_stderr(f):
                    exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
                    modules = exo.setup(cfg, adept_module=TPDModule)
                    diff_modules, static_modules = {}, {}
                    diff_modules["laser"], static_modules["laser"] = partition(
                        modules["laser"], modules["laser"].get_partition_spec()
                    )
                    val, grad, (sol, ppo, _) = exo.val_and_grad(diff_modules, args={"static_modules": static_modules})

            mlflow.log_artifact(_td, run_id=run_id)
        if export:
            export_run(run_id)
        return val, grad

    from utils import setup_parsl
    from adept import ergoExo
    from adept import utils as adept_utils
    from ml4tpd import TPDModule
    import parsl
    from parsl import python_app

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

    parsl_config = setup_parsl(cfg["parsl"]["provider"], 4, nodes=cfg["parsl"]["nodes"])
    run_one_val_and_grad = python_app(run_one_val_and_grad)

    # cfg["drivers"]["E0"]["params"]["key"] = np.random.randint(0, 2**10)
    # cfg["mlflow"]["run"] = f"{cfg['mlflow']['run']}-{cfg['drivers']['E0']['params']['key']}"
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        adept_utils.log_params(cfg)

    parent_run_id = mlflow_run.info.run_id
    adept_utils.export_run(parent_run_id)

    exo = ergoExo()
    modules = exo.setup(cfg, adept_module=TPDModule)
    diff_params, static_params = eqx.partition(modules["laser"], modules["laser"].get_partition_spec())

    # lr_sched = optax.cosine_decay_schedule(
    #     init_value=cfg["opt"]["learning_rate"], decay_steps=cfg["opt"]["decay_steps"]
    # )
    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        opt = optax.adam(learning_rate=cfg["opt"]["learning_rate"])
        opt_state = opt.init(eqx.filter(diff_params, eqx.is_array))  # initialize the optimizer state
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            os.makedirs(os.path.join(td, "weights-history"), exist_ok=True)  # create a directory for model history
            with parsl.load(parsl_config):
                for i in range(200):  # 1000 epochs
                    old_layer = diff_params.model.amp_decoder.layers[0].weight
                    module_path = os.path.join(td, "weights-history", f"weights-e{i:02d}.eqx")
                    modules["laser"].save(module_path)
                    orig_cfg["drivers"]["E0"]["file"] = module_path

                    with open(run_cfg_path := os.path.join(td, "config.yaml"), "w") as fi:
                        yaml.dump(orig_cfg, fi)

                    if cfg["opt"]["batch_size"] == 1:
                        with mlflow.start_run(nested=True, run_name=f"epoch-{i}") as nested_run:
                            pass
                        val, avg_grad = run_one_val_and_grad(run_id=nested_run.info.run_id, cfg_path=run_cfg_path)
                    else:
                        val_and_grads = []
                        for j in range(cfg["opt"]["batch_size"]):
                            export = np.random.choice([True, False], p=[0.1, 0.9])  # if j % 1 == 0 else False
                            with mlflow.start_run(nested=True, run_name=f"epoch-{i}-sim-{j}") as nested_run:
                                mlflow.log_artifact(module_path)
                                val_and_grads.append(
                                    run_one_val_and_grad(
                                        run_id=nested_run.info.run_id, _run_cfg_path=run_cfg_path, export=export
                                    )
                                )

                        vgs = [vg.result() for vg in val_and_grads]  # get the results of the futures
                        val = np.mean([v for v, _ in vgs])  # get the mean of the loss values

                        avg_grad = adept_utils.all_reduce_gradients([g for _, g in vgs], cfg["opt"]["batch_size"])

                    flat_grad, _ = ravel_pytree(avg_grad["laser"])
                    mlflow.log_metrics({"grad norm": float(np.linalg.norm(flat_grad))}, step=i)
                    mlflow.log_metrics({"loss": float(val)}, step=i)
                    if i % 10 == 0:
                        mlflow.log_artifacts(td)
                    adept_utils.export_run(parent_run_id, prefix="parent", step=i)
                    updates, opt_state = opt.update(avg_grad["laser"], opt_state, diff_params)

                    diff_params = eqx.apply_updates(diff_params, updates)
                    new_layer = diff_params.model.amp_decoder.layers[0].weight
                    assert not np.allclose(old_layer, new_layer)
                    modules["laser"] = eqx.combine(diff_params, static_params)


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

    run_opt(cfg_path)
