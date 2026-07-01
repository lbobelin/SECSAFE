#!/bin/bash
set -euo pipefail
mkdir -p logs
jid=$(sbatch --parsable submit_agv_campaign_v3_node.slurm)
echo "Submitted AGV campaign v3 as job $jid"
