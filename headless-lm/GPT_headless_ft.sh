#!/bin/bash -l
#SBATCH -J GPT_headless_ft_20e
#SBATCH --output=evaluation/evaluation/%x_output.log   # Auto-generated log file name (x = job-name)
#SBATCH --error=evaluation/evaluation/%x_error.log 
#SBATCH --time=0-06:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ino.vande.wouw@student.vu.nl
echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
echo "== Scratch dir. : ${TMPDIR}"
echo "== Home dir. : ${HOME}"


# environment modules
module load shared 2024 Python/3.11.3-GCCcore-12.3.0 Python-bundle-PyPI/2023.06-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a 

source $HOME/.venv/bin/activate
srun python ft_gpt_headless.py  --ckpt_path $HOME/Thesis-Git/headless-lm/HGPT-b12-70M/HGPT-b12-70M/HGPT-epoch=20-step=110000.ckpt  --config configs/gpt_vanilla_ft.json   --num_nodes 1     --global_bs 12     --gpu_bs 12 --dataset InoWouw/BabyLM-strict --run_name GPT_headless_ft_20e  --saved_ckpt HGPT-70M-ft