export CUDA_VISIBLE_DEVICES=0
BLD_TRAINING="/share/datasets/BlendedMVS/BlendedMVS_low_res"
BLD_TRAINLIST="lists/blendedmvs/train.txt"
BLD_TESTLIST="lists/blendedmvs/val.txt"
BLD_CKPT_FILE="./checkpoints/xxxx"  # dtu pretrained model

bldexp="train_bld"





PY_ARGS=${@:2}

BLD_LOG_DIR="./checkpoints/bld/"$bldexp
if [ ! -d $BLD_LOG_DIR ]; then
    mkdir -p $BLD_LOG_DIR
fi


python -u train_bld_mono.py --fpn_base_channel=8 --lr=0.001 --logdir $BLD_LOG_DIR --dataset=blendedmvs_aug_mono_cv2 --batch_size=1 --trainpath=$BLD_TRAINING --summary_freq 100 --loadckpt $BLD_CKPT_FILE\
        --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 8,8,4,4 --group_cor --inverse_depth --rt --attn_temp 2 --trainlist $BLD_TRAINLIST --testlist $BLD_TESTLIST  $PY_ARGS | tee -a $BLD_LOG_DIR/log.txt



