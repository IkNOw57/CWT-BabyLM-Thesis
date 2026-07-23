#!/bin/bash -l
# SLURM submission script template for glue_finetuning.py.
# Submit from the repository root with: sbatch scripts/glue_finetuning.sh
# Fill in the placeholder <...> values and adjust #SBATCH directives /
# module versions to match your cluster before running.
#SBATCH -J glue_finetuning
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=i.vande.wouw@student.vu.nl

echo "== Starting run at $(date)"
module load shared 2024 Python/3.11.3-GCCcore-12.3.0 Python-bundle-PyPI/2023.06-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a
source "$HOME/.venv/bin/activate"

srun python scripts/glue_finetuning.py \
    --model_ckpt <path_to_checkpoint>.ckpt \
    --mode <mlm|lm> \
    --train_batch_size 16 \
    --run_name <run_name>
