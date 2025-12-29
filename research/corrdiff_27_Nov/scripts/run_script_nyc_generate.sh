#!/bin/bash

#SBATCH --exclusive
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --qos=leap


CONTAINER=/home/lep-zerjs/leap/mike_testing/physicsnemo-2511-corrdiff.simg
#CONTAINER=/home/lep-qqttb/leap/physicsnemo-2506.simg
#CONTAINER=/home/lep-qqttb/leap/mike_testing/physicsnemo-pytorch-2405-corrdif-reqs/

#export HYDRA_FULL_ERROR=1
#export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1

# Set number of PyTorch (GPU) processes per node to be spawned by torchrun - NOTE: One for each GPU
NUM_PYTORCH_PROCESSES=2

# Define the compute node executing the batch script
RDZV_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Num Nodes: $SLURM_JOB_NUM_NODES"
echo "Num Procs: $NUM_PYTORCH_PROCESSES"

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

export RDZV_HOST
export RDZV_PORT=29400

echo "Rendezvous Node IP: $RDZV_HOST"

RUN_CMD="--nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$NUM_PYTORCH_PROCESSES \
        --rdzv_id=$RANDOM\
        --rdzv_backend=c10d \
        --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT"

#APP_CMD="/workspace/mike_testing/physicsnemo/examples/weather/corrdiff/train.py --config-name=config_training_hrrr_mini_regression.yaml"
APP_CMD="/leap/DB_scratch/physicsnemo/research/corrdiff_27_Nov/generate.py --config-name=config_generate_nyc_2.yaml"

srun apptainer exec --bind /home/lep-zerjs/leap:/leap --nv $CONTAINER torchrun $RUN_CMD $APP_CMD