import os 
import json

def read_json(data_dir):
    refer_list_file = os.path.join(data_dir, 'train_valid_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)
    return datastore