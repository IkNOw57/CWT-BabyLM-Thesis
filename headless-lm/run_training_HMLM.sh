#!/bin/bash -l
#SBATCH -J run_training_vanilla_MLM
#SBATCH --output=%x_output.log   # Auto-generated log file name (x = job-name)
#SBATCH --error=%x_error.log 
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ino.vande.wouw@student.vu.nl
echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
echo "== Scratch dir. : ${TMPDIR}"
echo "== Home dir. : ${HOME}"
# environment modules
module load shared 2024 Python/3.11.3-GCCcore-12.3.0 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source $HOME/.venv/bin/activate
python mlm_headless.py --config configs\mlm_headless_baby_lm.json --num_nodes 4 --global_bs 256 --gpu_bs 64  --dataset dataset_storage\baby-lm-small.hf   --hf_tokenizer bert-base-uncased --hf_path google-bert/bert-base-uncased --model_max_seq_len 2048 --run_name HMLM --saved_ckpt_path HMLM