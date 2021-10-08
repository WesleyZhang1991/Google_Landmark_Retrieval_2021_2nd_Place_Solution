set -e

DATASET_NAME=GLDv2
BACKBONE=SER101ibn    # RXt101ibn, SER101ibn, ResNeSt101/ResNeSt269, (R50, EffNetB7, R101ibn)
IMAGE_SIZE=512_all

# export NCCL_DEBUG=INFO
# torch.distributed.run, ref: https://pytorch.org/docs/stable/elastic/run.html
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 55555 --max_restarts 0 train.py --config_file configs/${DATASET_NAME}/${BACKBONE}_${IMAGE_SIZE}.yml
