#!/bin/bash

#SBATCH --job-name=gl_default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=47:59:59
#SBATCH --gres=gpu
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --export=NONE
#SBATCH --output="gerbil_default%j.out"


singularity exec --nv \
    --overlay /scratch/ci411/singularity_environments/gerbil_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
    /bin/bash -c\
    "init-s
    source /ext3/env.sh;
    cd /home/ci411/gerbil_roomsim;
    python rir_builder.py --save-path="/vast/ci411/gerbil_data/rir_datasets/default.h5" --save-yaml="rir_configs/default.yaml"
    python -m gerbilizer --config model_configs/base_config.json5 --data /vast/ci411/gerbil_data/rir_datasets/default.h5 --save-path /vast/ci411/gerbil_data/models/default"
