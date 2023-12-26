EPOCHS=20
BATCHES=64
SEED=100
LR=0.001

CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs $EPOCHS \
    --batches $BATCHES \
    --lr $LR \
    --seeds $SEED

# 共计67s