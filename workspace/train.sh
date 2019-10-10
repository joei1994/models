#!/bin/sh
PIPELINE_CONFIG_PATH=/home/iapp/Desktop/projects/models/workspace/example/training/faster_rcnn_resnet50_cat/config/pipeline.config
MODEL_DIR=/home/iapp/Desktop/projects/models/workspace/example/training/faster_rcnn_resnet50_cat/
NUM_TRAIN_STEPS=300000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python /home/iapp/Desktop/projects/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
    