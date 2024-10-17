FOLDER_NAME=$1
CONF=$2

python3 demo.py \
    --config-file /home/aditya/detectron2_mask_rcnn_X_101_32x8d_FPN_3x_training_d4_may8/config.yaml \
    --input /home/aditya/${FOLDER_NAME}/extracted_frames_200/*.jpg \
    --output /home/aditya/${FOLDER_NAME}_mask_rcnn_X_101_32x8d_FPN_3x_model_0005999_conf_${CONF} \
    --confidence-threshold ${CONF} \
    --opts MODEL.WEIGHTS /home/aditya/detectron2_mask_rcnn_X_101_32x8d_FPN_3x_training_d4_may8/model_0005999.pth \
