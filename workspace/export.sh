INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/iapp/Desktop/projects/models/workspace/lp_ocr/training/faster_rcnn_resnet50_gs_h150/config/pipeline.config
TRAINED_CKPT_PREFIX=/home/iapp/Desktop/projects/models/workspace/lp_ocr/training/faster_rcnn_resnet50_gs_h150/model.ckpt-30000
EXPORT_DIR=/home/iapp/Desktop/projects/models/workspace/lp_ocr/training/faster_rcnn_resnet50_gs_h150/export
python /notebooks/projects/object-detection/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}