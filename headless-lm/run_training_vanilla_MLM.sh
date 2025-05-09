#!/bin/bash -l
#SBATCH -J amp_recipe
#SBATCH -o amp_recipe.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=1
#SBATCH --time=0-00:10:00
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
python mlm_headless.py --config configs\mlm_vanilla_baby_lm.json --num_nodes 1 --global_bs 256 --gpu_bs 256  --dataset dataset_storage\baby-lm-small.hf   --hf_tokenizer bert-base-uncased --hf_path google-bert/bert-base-uncased --model_max_seq_len 2048 --run_name vanilla_MLM --saved_ckpt_path vanilla_MLM