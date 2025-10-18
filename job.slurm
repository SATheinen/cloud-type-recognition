#!/bin/bash
#SBATCH --job-name=jupyter-gpu
#SBATCH --output=jupyter.log
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --partition=dgx*

source venv/bin/activate
module load cuda

PORT=8889
HOSTNAME=$(hostname)

echo "Starting jupyter on node $HOSTNAME port $PORT"

jupyter lab --no-browser --port=$PORT --ip=0.0.0.0
