import logging, os
from ml4tpd.parsl_utils import setup_parsl

import parsl
from parsl import python_app
from itertools import product

from tpd_opt import run_opt_with_retry

logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_adept_fwd(_cfg_path, num_seeds=8):
    import yaml, mlflow

    from adept import ergoExo, utils as adept_utils
    import numpy as np
    from jax import config

    config.update("jax_enable_x64", True)

    from ml4tpd import TPDModule

    with open(_cfg_path, "r") as fi:
        _cfg = yaml.safe_load(fi)

    with mlflow.start_run(run_name=_cfg["mlflow"]["run"]) as parent_run:
        adept_utils.log_params(_cfg)
        vals = []
        for i in range(num_seeds):
            _cfg["drivers"]["E0"]["params"]["phases"]["seed"] = int(np.random.randint(0, 2**10))
            _cfg["mlflow"]["run"] = f"seed-{i}"
            exo = ergoExo(parent_run_id=parent_run.info.run_id, mlflow_nested=True)
            modules = exo.setup(_cfg, adept_module=TPDModule)
            run_output, ppo, _ = exo(modules)
            val = run_output[0]
            vals.append(val)
        mlflow.log_metric("loss", np.mean(vals))


def run_matlab(_cfg_path, bandwidth=False):
    import os
    import tempfile

    os.environ["PATH"] += ":/global/common/software/nersc9/texlive/2024/bin/x86_64-linux"
    import yaml, mlflow, subprocess

    from adept import utils as adept_utils
    import numpy as np
    from scipy.io import loadmat
    from matplotlib import pyplot as plt
    import scienceplots
    import xarray as xr

    plt.style.use(["science", "grid"])

    with open(_cfg_path, "r") as fi:
        _cfg = yaml.safe_load(fi)

    with mlflow.start_run(run_name=_cfg["mlflow"]["run"]) as parent_run:
        adept_utils.log_params(_cfg)
        vals = []
        num_seeds = 4 if bandwidth else 1
        for i in range(num_seeds):
            _cfg["drivers"]["E0"]["params"]["phases"]["seed"] = int(np.random.randint(0, 2**10))
            _cfg["mlflow"]["run"] = f"seed-{i}"
            intensity = float(_cfg["units"]["laser intensity"].split(" ")[0])
            gsl = float(_cfg["density"]["gradient scale length"].split(" ")[0])
            seed = _cfg["drivers"]["E0"]["params"]["phases"]["seed"]

            with mlflow.start_run(run_name=f"seed-{i}", nested=True, log_system_metrics=True) as mlflow_run:
                with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
                    matlab_cmd = [
                        "matlab",
                        "-batch",
                        f"addpath('{os.path.abspath('/global/common/software/m4490/matlab-lpse/')}');"
                        + f"log_lpse({intensity}, {gsl}, {str(bandwidth).lower()}, {seed}, '{td}')",
                    ]
                    subprocess.run(matlab_cmd)

                    data = loadmat(os.path.join(td, "output.mat"), simplify_cells=True)["output"]
                    epwEnergy = np.squeeze(data["metrics"]["epwEnergy"])
                    divEmax = np.squeeze(data["metrics"]["max"]["divE"])
                    nelfmax = np.squeeze(data["metrics"]["max"]["Nelf"])
                    params = {"intensity": intensity, "Ln": gsl, "seed": seed}
                    metrics = {
                        "epw_energy": float(epwEnergy[-1]),
                        "max_phi": float(divEmax[-1]),
                        "log10_epw_energy": float(np.log10(epwEnergy[-1])),
                    }

                    tax = np.arange(epwEnergy.shape[0]) * np.squeeze(data["dt"])
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
                    ax.semilogy(tax, epwEnergy, label="EPW Energy")
                    ax.set_xlabel("Time (ps)")
                    ax.set_ylabel("EPW Energy")
                    fig.savefig(os.path.join(td, "epw_energy.png"))
                    plt.close()

                    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
                    ax.semilogy(tax, divEmax, label="EPW Energy")
                    ax.set_xlabel("Time (ps)")
                    ax.set_ylabel("Maximum Potential")
                    fig.savefig(os.path.join(td, "max_divE.png"))
                    plt.close()

                    os.makedirs(laser_dir := os.path.join(td, "laser"), exist_ok=True)
                    os.makedirs(epw_dir := os.path.join(td, "epw"), exist_ok=True)

                    # save fields to xarrays
                    x_matlab_y_adept = np.squeeze(data["x"])
                    y_matlab_x_adept = np.squeeze(data["y"])
                    t = np.squeeze(data["outputTimes"])
                    t_skip = int(t.size // 8)
                    t_skip = t_skip if t_skip > 1 else 1
                    tslice = slice(0, -1, t_skip)

                    # save laser fields
                    E0x = np.array([data["E0_save"][i]["x"] for i in range(len(data["E0_save"]))])
                    E0y = np.array([data["E0_save"][i]["y"] for i in range(len(data["E0_save"]))])

                    laser_ds = xr.Dataset(
                        {
                            "E0x": (("time", "x", "y"), E0x),
                            "E0y": (("time", "x", "y"), E0y),
                        },
                        coords={
                            "x": ("x", x_matlab_y_adept, {"units": "um"}),
                            "y": ("y", y_matlab_x_adept, {"units": "um"}),
                            "time": ("time", t, {"units": "ps"}),
                        },
                    )

                    np.abs(laser_ds["E0x"][tslice].T).plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(laser_dir, "E0x.png"))
                    plt.close()

                    np.abs(laser_ds["E0y"][tslice].T).plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(laser_dir, "E0y.png"))
                    plt.close()

                    # plot lineout of E0y
                    np.abs(laser_ds["E0y"][tslice].isel(y=E0y.shape[2] // 2)).plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(laser_dir, "E0y_lineout.png"))
                    plt.close()
                    laser_ds.to_netcdf(os.path.join(laser_dir, "laser_fields.nc"))
                    # save epw fields
                    phi = np.array([data["divE_save"][i] for i in range(len(data["divE_save"]))])
                    phi_da = xr.DataArray(
                        phi,
                        dims=("time", "x", "y"),
                        coords={
                            "x": ("x", x_matlab_y_adept, {"units": "um"}),
                            "y": ("y", y_matlab_x_adept, {"units": "um"}),
                            "time": ("time", t, {"units": "ps"}),
                        },
                        name="phi",
                    )

                    np.abs(phi_da[tslice].T).plot(col="time", col_wrap=4)
                    plt.savefig(os.path.join(epw_dir, "phi.png"))
                    phi_da.to_netcdf(os.path.join(epw_dir, "epw_fields.nc"))
                    plt.close()

                    os.makedirs(density_dir := os.path.join(td, "density"), exist_ok=True)
                    os.makedirs(nelf_dir := os.path.join(td, "nelf"), exist_ok=True)

                    background_density_da = xr.DataArray(
                        data["backgroundDensity"],
                        dims=("x", "y"),
                        coords={
                            "x": ("x", x_matlab_y_adept, {"units": "um"}),
                            "y": ("y", y_matlab_x_adept, {"units": "um"}),
                        },
                        name="background_density",
                    )
                    nelf_da = xr.DataArray(
                        data["Nelf"],
                        dims=("x", "y"),
                        coords={
                            "x": ("x", x_matlab_y_adept, {"units": "um"}),
                            "y": ("y", y_matlab_x_adept, {"units": "um"}),
                            # "time": ("time", t, {"units": "ps"}),
                        },
                        name="nelf",
                    )
                    background_density_da.T.plot()
                    plt.savefig(os.path.join(density_dir, "background_density.png"))
                    plt.close()
                    background_density_da.to_netcdf(os.path.join(density_dir, "background_density.nc"))

                    nelf_da.T.plot()
                    plt.savefig(os.path.join(nelf_dir, "nelf.png"))
                    plt.close()
                    nelf_da.to_netcdf(os.path.join(nelf_dir, "nelf.nc"))

                    mlflow.log_artifacts(td)

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

            vals.append(np.log10(metrics["epw_energy"]))

        mlflow.log_metric("loss", np.mean(vals))


def scan_loop(_cfg_path, shape="uniform", solver="adept"):
    import uuid
    import numpy as np
    import yaml
    import mlflow
    import tempfile
    from ml4tpd.helpers import calc_tpd_threshold_intensity

    temperatures = np.round(np.linspace(2000, 4000, 5), 0)[:1]
    gradient_scale_lengths = np.round(np.linspace(200, 600, 5), 0)[:1]

    # intensities = np.round(np.linspace(2.0e14, 7.0e14, 13), 3)
    intensity_factors = np.linspace(1.4, 4.0, 16)

    all_hps = list(product(temperatures, gradient_scale_lengths, intensity_factors))
    with open(_cfg_path, "r") as fi:
        orig_cfg = yaml.safe_load(fi)

    parsl_config = setup_parsl(orig_cfg["parsl"]["provider"], 4, nodes=orig_cfg["parsl"]["nodes"], walltime="24:00:00")
    parsl_run_adept_fwd = python_app(run_adept_fwd)
    parsl_run_opt = python_app(run_opt_with_retry)
    orig_cfg["mlflow"]["experiment"] = f"{solver}-{shape}-tpd-100ps"
    all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])

    # find and pop completed runs from all_hps
    completed_runs = set()
    if not all_runs.empty:
        for run_name in all_runs["tags.mlflow.runName"].values:
            if run_name.startswith("temperature="):
                completed_runs.add(run_name)
    all_hps = [
        hp
        for hp in all_hps
        if f"temperature={hp[0]:.1f}-gsl={hp[1]:.1f}-intensity={1e14 * calc_tpd_threshold_intensity(hp[0]/1000, hp[1]) * hp[2]:.2e}"
        not in completed_runs
    ]
    print(f"Found {len(completed_runs)} completed runs, {len(all_hps)} remaining.")

    with parsl.load(parsl_config):
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
            num_nodes = orig_cfg["parsl"]["nodes"]
            batch_size = num_nodes * 4
            offset = 0
            num_batches = int(np.ceil((len(all_hps) - offset) / batch_size))
            print(f"Running {num_batches} batches of {batch_size} hyperparameter combinations each")
            print(f"Total combinations: {len(all_hps)}")
            for i in range(num_batches):
                vals = {}
                hp_slice = slice(batch_size * i + offset, batch_size * (i + 1) + offset)
                for tt, gsl, intensity_factor in all_hps[hp_slice]:
                    intensity = intensity_factor * calc_tpd_threshold_intensity(tt/1000, gsl) * 1e14
                    run_name = f"temperature={tt:.1f}-gsl={gsl:.1f}-intensity={intensity:.2e}"
                    orig_cfg["mlflow"]["run"] = run_name
                    # check if run name exists by first searching all runs and then checking if the run name exists
                    # all_runs = mlflow.search_runs(experiment_names=[orig_cfg["mlflow"]["experiment"]])
                    if all_runs.empty or run_name not in all_runs["tags.mlflow.runName"].values:
                        # Run does not exist, proceed to run
                        mlflow.set_experiment(orig_cfg["mlflow"]["experiment"])

                        if shape == "mono":
                            orig_cfg["drivers"]["E0"]["num_colors"] = 1
                            orig_cfg["drivers"]["E0"]["shape"] = "uniform"
                        elif shape == "uniform":
                            orig_cfg["drivers"]["E0"]["num_colors"] = 64
                            orig_cfg["drivers"]["E0"]["shape"] = shape
                        orig_cfg["units"]["reference electron temperature"] = f"{tt} eV"
                        orig_cfg["units"]["laser intensity"] = f"{intensity} W/cm^2"
                        orig_cfg["density"]["gradient scale length"] = f"{gsl} um"

                        if shape == "random_phaser":
                            opt_parent_run = mlflow.search_runs(
                                experiment_names=["arbitrary-64lines-more"],
                                filter_string=f"attributes.run_name LIKE '{run_name}'",
                            )["run_id"].values[0]
                            child_runs = mlflow.search_runs(
                                experiment_names=["arbitrary-64lines-more"],
                                filter_string=f"tags.mlflow.parentRunId = '{opt_parent_run}'",
                            )
                            child_run = child_runs.sort_values("start_time", ascending=False).iloc[0]
                            orig_cfg["drivers"]["E0"]["file"] = (
                                f"s3://public-ergodic-continuum/188470/{child_run.run_id}/artifacts/laser.eqx"
                            )

                        with open(
                            new_cfg_path := os.path.join(_td, f"config-{str(uuid.uuid4())[-6:]}.yaml"), "w"
                        ) as fi:
                            yaml.dump(orig_cfg, fi)

                        if solver == "adept":
                            if shape in ["uniform", "random_phaser", "mono"]:
                                vals[tt, gsl, intensity] = parsl_run_adept_fwd(
                                    _cfg_path=new_cfg_path, num_seeds=1 if shape == "mono" else 4
                                )
                            elif shape == "arbitrary":
                                vals[tt, gsl, intensity] = parsl_run_opt(new_cfg_path)
                            else:
                                raise NotImplementedError(f"Shape {shape} not implemented for adept.")

                        elif solver == "matlab":
                            if shape == "uniform":
                                vals[tt, gsl, intensity] = run_matlab(new_cfg_path, bandwidth=True)
                            elif shape == "mono":
                                vals[tt, gsl, intensity] = run_matlab(new_cfg_path, bandwidth=False)
                            else:
                                raise NotImplementedError(f"Shape {shape} not implemented for matlab.")
                        else:
                            raise NotImplementedError(f"Solver {solver} not implemented.")

                    else:
                        print(f"Run {run_name} already exists.")

                if solver == "adept":
                    for (tt, gsl, intensity), v in vals.items():
                        val = v.result()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the TPD scan")
    parser.add_argument("--config", type=str, help="The config file")
    args = parser.parse_args()
    cfg_path = args.config

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    scan_loop(cfg_path, shape="uniform", solver="adept")
