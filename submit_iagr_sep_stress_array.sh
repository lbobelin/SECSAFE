#!/bin/bash
set -euo pipefail
mkdir -p logs results figures

# Submit the array, then submit the merge/figure job after all array tasks finish successfully.
array_job=$(sbatch --parsable slurm/submit_iagr_sep_stress_array.slurm)
echo "Submitted campaign stress-test array job: ${array_job}"
merge_job=$(sbatch --parsable --dependency=afterok:${array_job} slurm/merge_iagr_sep_stress.slurm)
echo "Submitted dependent campaign merge/figure job: ${merge_job}"

echo "Monitor with: squeue -j ${array_job},${merge_job}"
