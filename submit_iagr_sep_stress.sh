#!/bin/bash
set -euo pipefail
mkdir -p logs results figures
sbatch slurm/submit_iagr_sep_stress_node.slurm
