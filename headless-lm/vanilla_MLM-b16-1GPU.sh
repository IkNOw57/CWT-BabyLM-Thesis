#!/bin/bash -l
#SBATCH -J vMLM
#SBATCH --output=output/%x_output.log   # Auto-generated log file name (x = job-name)
#SBATCH --error=output/%x_error.log 
#SBATCH --nodes=1
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
python mlm_headless.py --config configs/mlm_vanilla_baby_lm.json --num_nodes 1 --global_bs 16 --gpu_bs 16  --dataset InoWouw/BabyLM-strict-small   --hf_tokenizer bert-base-uncased --hf_path google-bert/bert-base-uncased --model_max_seq_len 2048 --run_name vanilla_MLM_b16 --saved_ckpt_path vanilla_MLM_b16 --ckpt_every 1000