#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=BERTrun
#SBATCH --nodes=2
#SBATCH --time=11:59:00
#SBATCH --exclusive
#SBATCH --mem=220G

#export MASTER_PORT=12340
#export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
#echo "NODELIST="${SLURM_NODELIST}
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#echo "MASTER_ADDR="$MASTER_ADDR

module load cuda
module load conda
conda activate deeplearning3

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun --nnodes 2 --nproc_per_node 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 train.py
