#!/bin/bash -l
# SLURM submission script template for mlm_headless.py.
# Submit from the repository root with: sbatch scripts/mlm_headless.sh
# Fill in the placeholder <...> values and adjust #SBATCH directives /
# module versions to match your cluster before running.
#SBATCH -J mlm_headless
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=i.vande.wouw@student.vu.nl

echo "== Starting run at $(date)"
module load shared 2024 Python/3.11.3-GCCcore-12.3.0 Python-bundle-PyPI/2023.06-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a
source "$HOME/.venv/bin/activate"

srun python scripts/mlm_headless.py \
    --config configs/<your_config>.json \
    --num_nodes 1 \
    --global_bs <global_batch_size> \
    --gpu_bs <per_device_batch_size> \
    --dataset <preprocessed_dataset>.hf \
    --hf_tokenizer bert-base-uncased \
    --hf_path bert-base-uncased \
    --run_name <run_name> \
    --saved_ckpt_path <checkpoint_output_dir>
