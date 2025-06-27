#!/bin/bash -l
#SBATCH -J HGPT-b12-70M
#SBATCH --output=output/%x_output.log   # Auto-generated log file name (x = job-name)
#SBATCH --error=output/%x_error.log 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
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
srun python gpt_headless.py --config configs/gpt_headless_70m.json --num_nodes 1 --global_bs 12 --gpu_bs 12 --dataset InoWouw/BabyLM-strict --hf_tokenizer bert-base-uncased --hf_path EleutherAI/pythia-70m --model_max_seq_len 128 --run_name HGPT-b12-70M --saved_ckpt_path HGPT-b12-70M --accelerator hf --ckpt_every 5000