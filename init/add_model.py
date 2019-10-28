import os
import argparse
from glob import glob
from shutil import copyfile
from pdb import set_trace
from utils.create_label_map import create_label_map
from utils.create_tf_record import generate_tf_records
from utils.constants import *

ap = argparse.ArgumentParser()
ap.add_argument('-pn', '--project_name', type=str, required=True, help='Project to add model to')
ap.add_argument('-mn', '--model_name', type=str, required=True, help='Model\'s name')
ap.add_argument('-pmn', '--pre_trained_model_name', type=str, required=True, help='Name of pre-trained model from model zoo')
ap.add_argument('-dsn', '--dataset_name', type=str, required=True, help='Name of data set')
ap.add_argument('-tc', '--target_classes', nargs='+', default=None, help='Target classes')
args = vars(ap.parse_args())    

def main():
    project_name = args['project_name']
    model_name = args['model_name']
    pre_trained_model_name = args['pre_trained_model_name']
    dataset_name = args['dataset_name']
    target_classes = args['target_classes']
    project_dir = os.path.join('..', WORKSPACE_DIR, project_name)
    if not os.path.exists(project_dir):
        raise Exception("Project not found")
        
    pre_trained_model_dir = os.path.join(project_dir, PROJECT_PRE_TRAINED_MODEL, pre_trained_model_name)
    if not os.path.exists(pre_trained_model_dir):
        raise  Exception("Pre trained model not found, Please download it from model zoo before proceed")
        
    dataset_dir = os.path.join(project_dir, PROJECT_DATA_DIR, PROJECT_DATA_TRAIN_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        raise Exception(f"Dataset not found: {dataset_dir}")
        
    model_dir = os.path.join(project_dir, PROJECT_TRAINING_DIR, f'{model_name}')    
    if not os.path.exists(model_dir):
        print("Create model dir")
        os.makedirs(model_dir)
        print("Create config dir")
        model_config_dir = os.path.join(model_dir, MODEL_PIPELINE_CONFIG_DIR)
        os.makedirs(model_config_dir)
        print("Create tf_records dir")
        model_data_dir = os.path.join(model_dir, PROJECT_DATA_DIR)
        os.makedirs(model_data_dir)
        print("Create label map dir")
        model_label_map_dir = os.path.join(model_dir, MODEL_LABEL_MAP_DIR)
        os.makedirs(model_label_map_dir)
        
        print("Copy pipeline.config")
        src_config_file = os.path.join(pre_trained_model_dir, MODEL_PIPELINE_CONFIG_FILE)
        des_config_file = os.path.join(model_config_dir, MODEL_PIPELINE_CONFIG_FILE)
        copyfile(src_config_file, des_config_file)
        
        print("Create label_map.pbtxt")
        model_label_map_file = os.path.join(model_label_map_dir, MODEL_LABEL_MAP_FILE)
        create_label_map(dataset_dir, model_label_map_file, target_classes) 
        
        print("Create tf records")
        generate_tf_records(
            dataset_dir = dataset_dir,
            label_map_file = model_label_map_file,
            tf_record_dir = os.path.join(model_dir, PROJECT_DATA_DIR),
            target_classes = target_classes
        )
    else:
        raise("This model already exists")
        
if __name__ == '__main__':
    main()