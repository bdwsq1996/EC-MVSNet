export CUDA_VISIBLE_DEVICES=0

DTU_TRAINING="/share/datasets/DTU/"
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"
exp="train_dtu"
LOAD_PATH="./checkpoints/DTU/finalmodel_9.ckpt"





PY_ARGS=${@:2}


DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi






python -u train_dtu_mono_aug.py --epochs=10 --logdir $DTU_LOG_DIR --dataset=dtu_yao4_aug_mono_cv2 --batch_size=1 --trainpath=$DTU_TRAINING --summary_freq 100 \
        --ndepths 32,16,8,4 --depth_inter_r 2,1,1,0.5 --group_cor_dim 8,8,4,4 --group_cor --inverse_depth --rt --attn_temp 2 --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt
