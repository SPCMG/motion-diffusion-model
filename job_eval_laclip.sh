#!/bin/bash
#SBATCH --nodes=1                # Use 1 node unless you need distributed computing across nodes
#SBATCH --ntasks=4               # Adjust tasks (parallel jobs) per node as needed
#SBATCH --cpus-per-task=4        # CPUs per task, adjust if needed
#SBATCH --account=vita
#SBATCH --mem=40G                # Increase memory if needed
#SBATCH --gres=gpu:4             # Use GPU if supported by the code
#SBATCH --time=72:00:00          # Set the job to 1 hour max
#SBATCH --partition=l40s         # h100
#SBATCH --output=/home/jiaxu/projects/sp-mdm/job_eval_output_laclip.log
#SBATCH --error=/home/jiaxu/projects/sp-mdm/job_eval_error_laclip.log
#SBATCH --mail-type=END,FAIL,DONE
#SBATCH --mail-user=jianan.xu@epfl.ch


echo "fidis $HOSTNAME"

# Load necessary modules
# module load gcc/11.3.0 python/3.10.4

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate ~/venvs/mdm

# Run the Python script
python -m eval.eval_humanml --model_path ./save/my_humanml_trans_enc_512/model000600154.pt