EPOCHS=20
BATCHES=128
N_GPU_PER_NODE=2
SEED=100
LR1=0.001

CUDA_VISIBLE_DEVICES=0,1 python train_DP.py \
    --epochs $EPOCHS \
    --batches $BATCHES \
    --lr $(($LR*$N_GPU_PER_NODE)) \
    --seeds $SEED


# 单卡batch64 共计128batch 共计74s