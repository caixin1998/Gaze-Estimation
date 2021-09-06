from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import random
import os,h5py
import numpy as np
from util.data_util import read_json,handle_eyecorner_rectangle

class MPKUDataset(BaseDataset):
    def __init__(self, opt, split = "train"):
        self.split = split
        self.opt = opt
        self.hdfs = {}
        self.root = opt.dataroot
        max_dataset_size = opt.max_dataset_size
        self.key_to_use = read_json(self.root)[self.split]
        self.selected_keys = [k for k in self.key_to_use]

        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            # print(list(self.hdfs[num_i]))
            # t = self.hdfs[num_i]["frames"].shape[0]

            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        self.idx_to_kv = []
        for num_i in range(0, len(self.selected_keys)):
            n = self.hdfs[num_i]["frames"].shape[0]
            for i in range(n):
                if split != "test":
                    if self.hdfs[num_i]["gaze_valids"][i]:
                        self.idx_to_kv += [(num_i, i)]
                else:
                    if not self.hdfs[num_i]["gaze_valids"][i]:
                        self.idx_to_kv += [(num_i, i)]
    

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if max_dataset_size is not None:
            random.seed(max_dataset_size)
            self.idx_to_kv = random.sample(self.idx_to_kv, max_dataset_size)
            # print(self.idx_to_kv[:20])
            random.seed(time.time())
  
        self.transform = get_transform(opt)
        

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def cropImage(self, img, bbox):
        bbox = np.array(bbox, int)
        aSrc = np.maximum(bbox[:2], 0)
        bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))
        aDst = aSrc - bbox[:2]
        bDst = aDst + (bSrc - aSrc)
        res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)    
        res[aDst[1]:bDst[1],aDst[0]:bDst[0],:] = img[aSrc[1]:bSrc[1],aSrc[0]:bSrc[0],:]
        return res
    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]
        entry = {}
        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        # assert self.hdf.swmr_mode
        # Get face image
        face = self.hdf['frames'][idx,:][0]
        pts = self.hdf['landmarks'][idx, :][0]
        gaze_pt = self.hdf['gaze_points'][idx, :]
        gaze_pt[0] = gaze_pt[0] * 53.15
        gaze_pt[1] = gaze_pt[1] * 29.90

        size = self.hdf['shapes'][idx, :][0]
        # leye = self.eyehdf["leyes"][idx,:]
        # reye = self.eyehdf["reyes"][idx,:]
        ec = handle_eyecorner_rectangle(pts, size)
      
        # leye = leye[:, :, [2, 1, 0]].astype(np.uint8) 
        # reye = reye[:, :, [2, 1, 0]].astype(np.uint8) 
        
        face = face[:, :, [2, 1, 0]]  # from BGR to RGB
        #print(image.shape,image.dtype, leye.shape, leye.dtype)
       # print(pts)
        # x_min,x_max,y_min,y_max = get_rect(pts[42:47])
        # #print(x_max - x_min, y_max - y_min)
        # leye_image = self.cropImage(image, [x_min,y_min,x_max - x_min,y_max -y_min])
        # x_min,x_max,y_min,y_max = get_rect(pts[36:41])
        # reye_image = self.cropImage(image, [x_min,y_min,x_max - x_min,y_max -y_min])
        #print(image.shape)
        face = self.transform(face)
        # leye_image = self.transform(leye_image)
        # reye_image = self.transform(reye_image)
        
        # head_pose = self.hdf['face_head_pose'][idx, :]
        # head_pose = head_pose.astype('float')
        entry["face"] = image
        entry["ec"] = np.array(ec)
        if self.split != "test":
            entry["gaze_pt"] = gaze_pt
        # print(gaze_pt.type)
        return entry