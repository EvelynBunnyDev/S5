#!/bin/bash
#
#SBATCH --job-name=s5_autoencoder
#SBATCH --time=7:59:59
#SBATCH -p swl1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --mail-type=END
#SBATCH --mail-user=evsong@stanford.edu
#SBATCH --mem=32GB

conda activate myenv

ml cuda/12.2.0
ml cudnn/8.9.0.131
ml py-jax/0.4.7_py39
ml py-jaxlib/0.4.7_py39
ml viz
ml py-matplotlib/3.7.1_py39

srun -N 1 -n 1 -o S5_train.out python3 -u autoencoder_real_data.py &
wait