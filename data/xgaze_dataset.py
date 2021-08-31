"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image

import os,h5py
import numpy as np

class XGazeDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
    
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
      
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        if is_train:
            parser.add_argument('--split', type=str, default="train", help='dataset split,eg: train, valid, test')
        else:
            parser.add_argument('--split', type=str, default="valid", help='dataset split,eg: train, valid, test')
        
        parser.add_argument('--index_file', type=str, default=None, help='mapping from full-data index to key and person-specific index')
        parser.add_argument('--cam_index', type=int, nargs='+', default=None, help='loading specific camera index for ethxgaze')

        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.key_to_use = read_json(self.root)
        # get the image paths of your dataset;

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.split, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode
    
        # Construct mapping from full-data index to key and person-specific index
        index_file = opt.index_file
        cam_list = opt.cam_list
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                if cam_list is None:
                    for i in range(0,n):
                        self.idx_to_kv += [(num_i, i)]
                else:
                    for cam in cam_list:
                        for i in range(cam, n, 18):
                            self.idx_to_kv += [(num_i, i)]
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __len__(self):
        return len(self.idx_to_kv)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        key, idx = self.idx_to_kv[index]
        self.hdf = h5py.File(os.path.join(self.path, self.split, self.selected_keys[key]), 'r', swmr=True)
        face = self.hdf['face_patch'][idx,:]
        face = face[:, :, [2, 1, 0]]
        gaze_label = self.hdf['face_gaze'][idx, :]

        return {'face': face, 'gaze': gaze_label}

