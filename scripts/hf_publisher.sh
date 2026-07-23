#!/bin/bash -l
# SLURM submission script template for hf_publisher.py.
# Submit from the repository root with: sbatch scripts/hf_publisher.sh
# Fill in the placeholder <...> values. Requires HF_TOKEN to be set/exported.
#SBATCH -J hf_publisher
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=i.vande.wouw@student.vu.nl

echo "== Starting run at $(date)"
module load shared 2024 Python/3.11.3-GCCcore-12.3.0 Python-bundle-PyPI/2023.06-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a
source "$HOME/.venv/bin/activate"

srun python scripts/hf_publisher.py \
    --hf_name <your_hf_id>/<model_name> \
    --model_ckpt <path_to_checkpoint>.ckpt \
    --mode <mlm|lm|add_head>
