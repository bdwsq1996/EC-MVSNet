export CUDA_VISIBLE_DEVICES=0
DTU_TESTPATH="/share/datasets/DTU/dtu_testing/"
DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE='./checkpoints/DTU/finalmodel_9.ckpt' # dtu pretrained model




exp="test_dtu"
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_OUT_DIR="./output/"$exp

if [ ! -d $DTU_OUT_DIR ]; then
    mkdir -p $DTU_OUT_DIR
fi



python -u test_dtu_dypcd_mono.py --dataset=general_eval4_mono_cv2 --num_view=5 --fpn_base_channel=8 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST  --interval_scale 1.06 --outdir $DTU_OUT_DIR\
            --loadckpt $DTU_CKPT_FILE --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 8,8,4,4 --conf 0.5 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt


