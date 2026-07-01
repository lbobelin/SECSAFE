#!/bin/bash
#SBATCH --job-name=iagrc_campaign_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/iagrc_campaign_merge_%j.out
#SBATCH --error=logs/iagrc_campaign_merge_%j.err

set -euo pipefail
mkdir -p logs results figures

module purge
module load miniforge/24.11

# Uncomment/adapt if you use a venv or conda environment:
source ~/venvs/agvmomdp/bin/activate

python merge_iagr_sep_stress_results.py \
  --out-dir results/iagrc_campaign_stress \
  --figures-dir figures/iagrc_campaign_stress
