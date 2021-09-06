import os 
import json

def read_json(data_dir):
    refer_list_file = os.path.join(data_dir, 'train_valid_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)
    return datastore

def handle_eyecorner_rectangle(pts,size):
    w,h = size[0],size[1]
    x1,y1,x2,y2,x3,y3,x4,y4 = pts[36][0],pts[36][1],pts[39][0],pts[39][1],pts[42][0],pts[42][1],pts[45][0],pts[45][1]
    x1,x2,x3,x4 = x1 / w, x2 / w,x3 / w,x4 / w 
    y1,y2,y3,y4 = y1 / h, y2 / h,y3 / h,y4 / h
    return [x1,y1,x2,y2,x3,y3,x4,y4]