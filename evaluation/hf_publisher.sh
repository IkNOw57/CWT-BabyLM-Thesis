#!/bin/bash -l
#SBATCH -J Publisher_30000_HMLMstp
#SBATCH --output=evaluation/%x_output.log   # Auto-generated log file name (x = job-name)
#SBATCH --error=evaluation/%x_error.log 
#SBATCH --time=0-01:00:00
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
python  $HOME/Thesis-Git/headless-lm/hf_publisher.py --hf_name InoWouw/VGPT_20e --model_ckpt $HOME/Thesis-Git/headless-lm/HGPT-b12-70M/HGPT-b12-70M/VGPT-epoch=20-step=110000.ckpt --mode lm
