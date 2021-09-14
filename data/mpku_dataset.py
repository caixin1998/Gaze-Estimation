from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import random
import os,h5py
import numpy as np
import torch
from util.data_util import read_json,handle_eyecorner_rectangle, draw_point, get_rect

class MPKUDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--interval', type=int, default=5, help='interval for video dataset.')
        parser.add_argument('--create_idx_file', action='store_true', default=False, help='create index file mapping from full-data index to key and person-specific index')
        parser.add_argument('--debug', action='store_true', default=False, help='debug using ec and gaze point position.')
        return parser
    def __init__(self, opt, split = "train"):
        self.split = split
        self.opt = opt
        self.hdfs = {}
        self.root = opt.dataroot
        max_dataset_size = opt.max_dataset_size
        self.key_to_use = read_json(self.root)[self.split]
        self.selected_keys = [k for k in self.key_to_use]
        # self.selected_keys = self.selected_keys[::8]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.root, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.root, self.selected_keys[num_i]))
            # print(list(self.hdfs[num_i]))
            # t = self.hdfs[num_i]["frames"].shape[0]

            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        self.idx_to_kv = []
        index_file = os.path.join(self.root, split + ".txt") 
        if not os.path.isfile(index_file) or opt.create_idx_file:
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["frames"].shape[0]
                for i in range(0,n,opt.interval):
                    if split != "test":
                        if self.hdfs[num_i]["gaze_valids"][i] and np.all(self.hdfs[num_i]["face_valids"][i]) :
                            self.idx_to_kv += [(num_i, i)]
                    else:
                        if not (self.hdfs[num_i] ["gaze_valids"][i]):
                        # and np.all(self.hdfs[num_i]["face_valids"]):
                            self.idx_to_kv += [(num_i, i)]

            np.savetxt(index_file, np.array(self.idx_to_kv,dtype = np.int),fmt = '%d')
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)
    
      
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
        hdf = h5py.File(os.path.join(self.root, self.selected_keys[key]), 'r', swmr=True)
        # assert hdf.swmr_mode
        # Get face image
        face = hdf['frames'][idx,:][0]
        pts = hdf['landmarks'][idx, :][0]
        gaze_pt = hdf['gaze_points'][idx, :]
        gaze_pt_ = hdf['gaze_points'][idx, :]

        gaze_pt[0] = gaze_pt[0] * 53.15
        gaze_pt[1] = gaze_pt[1] * 29.90

        size = hdf['shapes'][idx, :][0]

        hdf.close()
        # leye = self.eyehdf["leyes"][idx,:]
        # reye = self.eyehdf["reyes"][idx,:]
        ec = handle_eyecorner_rectangle(pts, size)
      
        # leye = leye[:, :, [2, 1, 0]].astype(np.uint8) 
        # reye = reye[:, :, [2, 1, 0]].astype(np.uint8) 
        
        face = face[:, :, [2, 1, 0]]  # from BGR to RGB
        #print(image.shape,image.dtype, leye.shape, leye.dtype)
       # print(pts)
        x_min,x_max,y_min,y_max = get_rect(pts[42:47])
        #print(x_max - x_min, y_max - y_min)
        leye_image = self.cropImage(face, [x_min,y_min,x_max - x_min,y_max -y_min])
        x_min,x_max,y_min,y_max = get_rect(pts[36:41])
        reye_image = self.cropImage(face, [x_min,y_min,x_max - x_min,y_max -y_min])
        #print(image.shape)
        face = self.transform(face)
        # leye_image = self.transform(leye_image)
        # reye_image = self.transform(reye_image)
        
        # head_pose = self.hdf['face_head_pose'][idx, :]
        # head_pose = head_pose.astype('float')
        entry["face"] = face
        entry["ec"] = np.array(ec)
        entry["ec"] = torch.FloatTensor(entry["ec"])
        if self.opt.debug:
            entry["ec_position"] = draw_point([(ec[0] + ec[2] + ec[4] + ec[6]) / 4,(ec[1] + ec[3] + ec[5] + ec[7]) / 4])
            entry["gt_position"] = draw_point(gaze_pt_)

   
        if self.split != "test":
            entry["gaze_pt"] = gaze_pt

        entry["gaze_pt"] = torch.FloatTensor(entry["gaze_pt"])
        entry["index"] = [os.path.join(self.root, self.selected_keys[key]), key, idx]
        # print(gaze_pt.type)
        return entry