import os 
import json
import numpy as np 
import cv2 as cv
import torch 
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

def draw_point(pts,size = [192, 108]):
    img = np.zeros((size[1],size[0]))
    cv.rectangle(img, (0,0), (192,108), color = 1.0, thickness = 1)
    pts = [pts[0] * size[0], pts[1] * size[1]]
    try:
        pts = np.array(pts, dtype=np.int)
        img = cv.circle(img,pts,10,thickness = -1, color = 1.0)
    except:
        pass
    # print(img.shape)
    img = np.expand_dims(img, axis=0)
    return img


def draw_gaze(gt, pred = None, image_in = None, size = (224, 224, 3), thickness=2, color=(0, 0, 255), color_gt=(0, 255, 0)):
    if image_in is None:
        image_in = np.zeros(size)

    image_out = image_in
    # print("image_in.shape:", image_in.shape)
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 4.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(gt[1]) * np.cos(gt[0])
    dy = -length * np.sin(gt[0])
    cv.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    if pred is not None:
        dx = -length * np.sin(pred[1]) * np.cos(pred[0])
        dy = -length * np.sin(pred[0])
        cv.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color_gt,
                   thickness, cv.LINE_AA, tipLength=0.2)  
    image_out = image_out.transpose((2,0,1))
    return image_out


def get_rect(points , ratio = 1.0): # ratio = w:h
    x = points[:,0]
    y = points[:,1]

    x_expand = 0.1*(max(x)-min(x))
    y_expand = 0.1*(max(y)-min(y))
    
    
    x_max, x_min = max(x)+x_expand, min(x)-x_expand
    y_max, y_min = max(y)+y_expand, min(y)-y_expand

    #h:w=1:2
    if (y_max-y_min)*ratio < (x_max-x_min):
        h = (x_max-x_min)/ratio
        pad = (h-(y_max-y_min))/2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max-y_min)
        pad = (h*ratio-(x_max-x_min))/2
        x_max += pad
        x_min -= pad
    return int(x_min),int(x_max),int(y_min),int(y_max)

def get_eye_rect(face_points, eye_points, ratio = 1.0):
    x_min,x_max,y_min,y_max = get_rect(face_points)
    w = x_max - x_min
    eye_points_copy = eye_points.copy()
    eye_points_copy[:,0] = eye_points_copy[:,0] - x_min
    eye_points_copy[:,1] = eye_points_copy[:,1] - y_min
    eye_points_copy = eye_points_copy * 224 / w
    try:
        result = get_rect(eye_points_copy, ratio = ratio)
    except:
        # print("face_points:",face_points)
        result = 0, 1, 0, 1
    return result

def tensor2im(input_image, imtype=np.uint8):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    # print(image_tensor.shape)
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = (image_numpy - np.min(image_numpy)) / (np.max(image_numpy) - np.min(image_numpy))
        image_numpy = image_numpy * 2 - 1
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    for i in range(len(mean)):
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0.0, 255.0)
    return image_numpy.astype(imtype).copy()