#!/bin/bash
#SBATCH --job-name=iagrc_campaign_arr
#SBATCH --array=0-95%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/iagrc_campaign_%A_%a.out
#SBATCH --error=logs/iagrc_campaign_%A_%a.err

set -euo pipefail
mkdir -p logs results figures

module purge
module load miniforge/24.11

# Uncomment/adapt if you use a venv or conda environment:
source ~/venvs/agvmomdp/bin/activate

python run_iagr_sep_stress_array_task.py \
  --fleet-sizes 50,100,200,300 \
  --profiles convoy,sequential,shared_resource,coordination \
  --strengths 6,10,15 \
  --alphas 1,2 \
  --beta 2 \
  --policies IAGR_C,IAGR,SEP,II \
  --runs 30 \
  --horizon 120 \
  --out-dir results/iagrc_campaign_stress \
  --campaign-cost-model \
  --campaign-group-size 5 \
  --campaign-marginal-cost-factor 0.45
