#!/bin/bash
unset ${!SLURM_@};

export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="https://continuum.ergodic.io/experiments/"

# copy job stuff over
source /pscratch/sd/a/archis/venvs/ml-for-lpi/bin/activate

cd /global/u2/a/archis/ml-for-lpi/

python tpd_uniform_scan.py --config configs/tpd-opt.yaml