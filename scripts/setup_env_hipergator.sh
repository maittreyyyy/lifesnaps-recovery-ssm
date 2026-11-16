#!/bin/bash
# One time setup for the LifeSnaps project on HiPerGator.
# Usage: bash setup_env.sh <your_group_name>
# Run from a login node. Package installs are light; heavy work goes in a job.

set -euo pipefail

GROUP="${1:?Pass your HiPerGator group name, e.g. bash setup_env.sh mygroup}"
PROJECT="/blue/${GROUP}/${USER}/lifesnaps"
ENV_PATH="${PROJECT}/conda/envs/lifesnaps"

mkdir -p "${PROJECT}"/{data/raw,data/processed,notebooks,scripts,results,logs}

module load conda

# Keep conda packages on /blue instead of the small home directory
conda config --add pkgs_dirs "/blue/${GROUP}/${USER}/conda/pkgs"

if [ ! -d "${ENV_PATH}" ]; then
  mamba create -y -p "${ENV_PATH}" python=3.11 \
    numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
    jupyterlab ipykernel pyarrow tqdm
fi

conda activate "${ENV_PATH}"

# Deep learning and data tools
pip install torch pyro-ppl zenodo-get neurokit2

# Register the environment as a Jupyter kernel for Open OnDemand
python -m ipykernel install --user --name lifesnaps --display-name "LifeSnaps (py3.11)"

echo "Project folder: ${PROJECT}"
echo "Environment:    ${ENV_PATH}"
echo "Kernel 'LifeSnaps (py3.11)' is now available in Jupyter on ood.rc.ufl.edu"
