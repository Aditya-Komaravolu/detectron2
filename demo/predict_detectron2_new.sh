FOLDER_NAME=$1
CONF=$2

python3 demo.py \
    --config-file /home/aditya/detectron2/configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py \
    --input /home/aditya/${FOLDER_NAME}/extracted_frames_200/*.jpg \
    --output /home/aditya/${FOLDER_NAME}_mask_rcnn_R_101_FPN_400ep_LSJ_conf_${CONF} \
    --min_size_test 720 \
    --max_size_test 1280\
    --confidence-threshold ${CONF} \
    --opts train.init_checkpoint=/home/aditya/detectron2_mask_rcnn_R_101_FPN_400ep_LSJ_training_d4_may8/model_0160999.pth \
    train.device='cuda'
