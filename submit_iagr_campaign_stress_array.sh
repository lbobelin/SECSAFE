#!/bin/bash
set -euo pipefail
mkdir -p logs results figures

# Submit the campaign-aware array, then merge/generate figures after successful completion.
array_job=$(sbatch --parsable slurm/submit_iagr_sep_stress_array.slurm)
echo "Submitted campaign-aware IAGR-C stress-test array job: ${array_job}"
merge_job=$(sbatch --parsable --dependency=afterok:${array_job} slurm/merge_iagr_sep_stress.slurm)
echo "Submitted dependent merge/figure job: ${merge_job}"

echo "Monitor with: squeue -j ${array_job},${merge_job}"
