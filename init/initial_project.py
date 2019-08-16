import os
import argparse
from glob import glob
from utils.constants import WORKSPACE_DIR, PROJECT_DATA_DIR, PROJECT_DATA_TRAIN_DIR, PROJECT_DATA_TEST_DIR, PROJECT_PRE_TRAINED_MODEL, PROJECT_TRAINING_DIR

ap = argparse.ArgumentParser()
ap.add_argument('-pn', '--project_name', default='My Project', help='Project\'s name')
args = vars(ap.parse_args())

def get_last_dir_from_path(path):
    return path.split('/')[-2]

def main():
    project_name = args['project_name']
    project_dir = os.path.join('..', WORKSPACE_DIR, project_name)
    if not os.path.exists(project_dir):
        dirs_to_create = [
            os.path.join(project_dir, PROJECT_DATA_DIR, PROJECT_DATA_TRAIN_DIR),
            os.path.join(project_dir, PROJECT_DATA_DIR, PROJECT_DATA_TEST_DIR),
            os.path.join(project_dir, PROJECT_PRE_TRAINED_MODEL),
            os.path.join(project_dir, PROJECT_TRAINING_DIR)
        ]
        
        for d in dirs_to_create:
            os.makedirs(d)
        
    else:
        raise Exception("This project already exists")
    
if __name__ == "__main__":
    main()