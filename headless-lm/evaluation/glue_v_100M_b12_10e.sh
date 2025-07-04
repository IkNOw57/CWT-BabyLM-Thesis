#!/bin/bash -l
#SBATCH -J GLUE_evaluation_vanilla_100M_b12_10e
#SBATCH --output=evaluation/%x_output.log   # Auto-generated log file name (x = job-name)
#SBATCH --error=evaluation/%x_error.log 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=i.vande.wouw@student.vu.nl
echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
echo "== Scratch dir. : ${TMPDIR}"
echo "== Home dir. : ${HOME}"
# environment modules
module load shared 2024 Python/3.11.3-GCCcore-12.3.0 Python-bundle-PyPI/2023.06-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a


source $HOME/.venv/bin/activate

srun python $HOME/Thesis-Git/headless-lm/glue_finetuning_unpublished.py --model_ckpt $HOME/Thesis-Git/headless-lm/vanilla_MLM_b12_100M/vanilla_MLM_b12_100M/epoch=10-step=55000.ckpt --mode mlm --train_batch_size 64 --run_name vanilla_MLM-b12_100M

#srun python $HOME/Thesis-Git/headless-lm/glue_finetuning_unpublished.py --model_ckpt $HOME/Thesis-Git/headless-lm/vanilla_MLM_b12_100M/vanilla_MLM_b12_100M/epoch=10-step=55000.ckpt --mode mlm --train_batch_size 12 --run_name vanilla_MLM-b12_100M

#srun python $HOME/Thesis-Git/headless-lm/glue_finetuning_unpublished.py --model_ckpt $HOME/Thesis-Git/headless-lm/vanilla_MLM_b12_100M/vanilla_MLM_b12_100M/epoch=10-step=55000.ckpt --mode mlm --train_batch_size 64 --run_name vanilla_MLM-b12_100M
