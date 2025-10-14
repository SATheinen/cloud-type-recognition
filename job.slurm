#!/bin/bash
#SBATCH --job-name=jupyter-gpu
#SBATCH --output=jupyter.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=dgx*

source venv/bin/activate
module load cuda

PORT=8889
HOSTNAME=$(hostname)

echo "Starting jupyter on node $HOSTNAME port $PORT"

jupyter lab --no-browser --port=$PORT --ip=0.0.0.0
