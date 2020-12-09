CODE_DIR=codes/
SAVE_DIR=checkpoints/

# Meta-training
NUM_SPLIT=10
SRC_LR=1e-3
SRC_DATA=tiny_imagenet
SRC_IMG_SIZE=32
SRC_STEPS=750
SRC_MODEL=resnet20
SRC_DETAIL=final
SRC_EXP_NAME=${SRC_DATA}_${SRC_MODEL}_${SRC_DETAIL}
GPUS=0,1,2,3,4,0,1,2,3,4

# Meta-testing
TGT_LR=1e-3
TGT_DATA=stanford_cars
TGT_IMG_SIZE=84
TGT_MODEL=resnet20
NOISE_COEFF=1
TGT_DETAIL=
TGT_EXP_NAME=${SRC_DATA}_${SRC_MODEL}_${SRC_STEPS}_to_${TGT_DATA}_${TGT_MODEL}_${TGT_DETAIL}

wandb on
if [ "$1" = "src" ]; then
python train_src.py \
  --num_run 5 \
  --code_dir $CODE_DIR \
  --save_dir $SAVE_DIR \
  --num_split $NUM_SPLIT \
  --lr $SRC_LR \
  --data $SRC_DATA \
  --img_size $SRC_IMG_SIZE \
  --model $SRC_MODEL \
  --train_steps $SRC_STEPS \
  --exp_name $SRC_EXP_NAME \
  --gpus $GPUS \
  --num_workers 2 
elif [ "$1" = "tgt" ]; then
python train_tgt.py \
  --code_dir $CODE_DIR \
  --save_dir $SAVE_DIR \
  --lr $TGT_LR \
  --data $TGT_DATA \
  --img_size $TGT_IMG_SIZE \
  --model $TGT_MODEL \
  --src_name $SRC_EXP_NAME \
  --src_steps $SRC_STEPS \
  --exp_name $TGT_EXP_NAME \
  --gpus $2
else
echo Wrong Argument
fi
