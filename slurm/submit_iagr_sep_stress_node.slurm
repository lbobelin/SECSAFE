#!/bin/bash
#SBATCH --job-name=iagr_sep_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/iagr_sep_stress_%j.out
#SBATCH --error=logs/iagr_sep_stress_%j.err

set -euo pipefail
mkdir -p logs results figures

# Adjust these lines for your cluster if needed.
# module load python/3.11
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python run_iagr_sep_stress_parallel.py \
  --fleet-sizes 50,100,200,300 \
  --profiles convoy,sequential,shared_resource,coordination \
  --strengths 6,10,15 \
  --alphas 1,2 \
  --beta 2 \
  --policies IAGR,SEP,II \
  --runs 30 \
  --horizon 120 \
  --workers "${SLURM_CPUS_PER_TASK:-32}" \
  --out-dir results/iagr_sep_stress

python make_iagr_sep_stress_figures.py \
  --summary results/iagr_sep_stress/iagr_sep_stress_summary.csv \
  --ratios results/iagr_sep_stress/iagr_sep_stress_ratios.csv \
  --out-dir figures/iagr_sep_stress
