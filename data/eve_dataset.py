"""Copyright 2020 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import pickle
from typing import List
import time
import cv2 as cv
import h5py
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_transform

from util.data_util import read_json
from util.eve_util import predefined_splits, stimulus_type_from_folder_name
import simplejpeg as jpeg

source_to_fps = {
    'screen': 30,
    'basler': 60,
    'webcam_l': 30,
    'webcam_c': 30,
    'webcam_r': 30,
}

source_to_interval_ms = dict([
    (source, 1e3 / fps) for source, fps in source_to_fps.items()
])

img_segmentations = None
cache_pkl_path = './eve_segmentation_cache.pkl'

class EVEDataset(BaseDataset):

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

      
        parser.add_argument('--cameras_to_use', type=str, nargs='+', default=['basler'], help='data form those cameras will be used.')
        parser.add_argument('--types_of_stimuli', type=str, nargs='+', default=["image", "video", "wikipedia"], help='using specific types of stimuli from eve')

        parser.add_argument('--face_size', type=int, default=256, help='size of input face image.')

        parser.add_argument('--eyes_size', type=int, default=128, help='size of input eye image.')

        parser.add_argument('--interval', type=int, default=1, help=' the interval of sampling in videos.')

        parser.add_argument('--camera_frame_type', type=str, default= "eyes", help = 'the input of models')

        parser.add_argument('--h5s_path', type=str, default= "/home/caixin/GazeData/eve_h5", help = 'h5s file path')



        parser.set_defaults(max_dataset_size=None, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt, split):
        

        self.split = split
        self.opt = opt
        self.root = opt.dataroot

        self.key_to_use = predefined_splits[self.split]


        self.types_of_stimuli = opt.types_of_stimuli
    
        self.cameras_to_use = opt.cameras_to_use
        self.validation_data_cache = {}

        # Some sanity checks
        assert(len(self.key_to_use) > 0)
    

        # Load or calculate sequence segmentations (start/end indices)
        global cache_pkl_path, img_segmentations
        cache_pkl_path = (
            './segmentation_cache/base_interval=%d.pkl'%self.opt.interval
        )

        os.makedirs("./segmentation_cache", exist_ok=True)

        if img_segmentations is None:
            if not os.path.isfile(cache_pkl_path):
                self.build_segmentation_cache()
                assert(os.path.isfile(cache_pkl_path))
            with open(cache_pkl_path, 'rb') as f:
                img_segmentations = pickle.load(f)
        # Register entries
        self.select_imgs()
        print('Initialized %s dataset class for: %s' % (self.split, self.root))

    def build_segmentation_cache(self):
        """Create support data structure for knowing how to segment (cut up) time sequences."""
        all_folders = sorted([
            d for d in os.listdir(self.root) if os.path.isdir(self.root + '/' + d)
        ])
        output_to_cache = {}
        for folder_name in all_folders:
            participant_path = '%s/%s' % (self.root, folder_name)
            assert(os.path.isdir(participant_path))
            output_to_cache[folder_name] = {}

            subfolders = sorted([
                p for p in os.listdir(participant_path)
                if os.path.isdir(os.path.join(participant_path, p))
                and p.split('/')[-1].startswith('step')
                and 'eye_tracker_calibration' not in p
            ])
            for subfolder in subfolders:
                subfolder_path = '%s/%s' % (participant_path, subfolder)
                output_to_cache[folder_name][subfolder] = {}

                # NOTE: We assume that the videos are synchronized and have the same length in time.
                #       This should be the case for the publicly released EVE dataset.
                for source in ('basler', 'webcam_l', 'webcam_c', 'webcam_r'):
                    current_outputs = []
                    source_path_pre = '%s/%s' % (subfolder_path, source)
                    available_indices = np.loadtxt('%s.timestamps.txt' % source_path_pre)
                    for i in range(0, len(available_indices), self.opt.interval):
                        current_outputs.append(i)
                    # Store back indices
                    if len(current_outputs) > 0:
                        output_to_cache[folder_name][subfolder][source] = current_outputs
                        # print('%s: %d' % (source_path_pre, len(current_outputs)))
        # Do the caching
        with open(cache_pkl_path, 'wb') as f:
            pickle.dump(output_to_cache, f)

        print('> Stored indices of sequences to: %s' % cache_pkl_path)


    def select_imgs(self):
        self.all_imgs = []
        for key, values in img_segmentations.items():
            if key not in self.key_to_use:
                continue
            for stimulus_key, stimulus_values in values.items():
                current_stimulus_type = stimulus_type_from_folder_name(stimulus_key) 
                if current_stimulus_type not in self.types_of_stimuli:
                    continue
                for camera, all_indices in stimulus_values.items():
                    if camera not in self.cameras_to_use:
                        continue
                    
                    for i, index in enumerate(all_indices):
                        self.all_imgs.append({
                            'camera_name': camera,
                            'participant': key,
                            'subfolder': stimulus_key,
                            'partial_path': '%s/%s' % (key, stimulus_key),
                            'full_path': '%s/%s/%s' % (self.root, key, stimulus_key),
                            'h5_path': '%s/%s/%s' % (self.opt.h5s_path, key, stimulus_key),
                            'index': index,
                        })



    def __len__(self):
        return len(self.all_imgs)

    def handle_eye_corner(self, pts):
 
        w,h = 1920, 1080
        x1,y1,x2,y2,x3,y3,x4,y4 = pts[36][0],pts[36][1],pts[39][0],pts[39][1],pts[42][0],pts[42][1],pts[45][0],pts[45][1]
        x1,x2,x3,x4 = x1 / w, x2 / w,x3 / w,x4 / w 
        y1,y2,y3,y4 = y1 / h, y2 / h,y3 / h,y4 / h
        return [x1,y1,x2,y2,x3,y3,x4,y4]

    def preprocess_frame(self, frame):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frame = np.transpose(frame, [2, 0, 1])
        frame = frame.astype(np.float32)
        frame *= 2.0 / 255.0
        frame -= 1.0
        return frame

    def preprocess_screen_frames(self, frames):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frames = np.transpose(frames, [0, 3, 1, 2])
        frames = frames.astype(np.float32)
        frames *= 1.0 / 255.0
        return frames

    screen_frames_cache = {}

    def load_all_from_source(self, h5s_path, videos_path, source, index):
        assert(source in ('basler', 'webcam_l', 'webcam_c', 'webcam_r', 'screen'))

        # Read HDF
        subentry = {}  # to output
        start = time.time()
        if source != 'screen':
            with h5py.File('%s/%s.h5' % (h5s_path, source), 'r') as hdf:
                for k1, v1 in hdf.items():
                    if isinstance(v1, h5py.Group):
                        subentry[k1] = np.copy(v1['data'][index])
                        subentry[k1 + '_validity'] = np.copy(v1['validity'][index])
                    elif isinstance(v1, h5py.Dataset):
                        subentry[k1] = np.copy(v1)
                              

            # Compute rotation matrices from rvec values
            subentry['head_R'] = cv.Rodrigues(subentry['head_rvec'])[0]

        # print(time.time() - start)

        # Get frames
        video_path = '%s/%s' % (videos_path, source)
        output_size = None

        if self.opt.camera_frame_type == 'face':
            video_path += '_face'
            output_size = (self.opt.face_size, self.opt.face_size)
        elif self.opt.camera_frame_type == 'original_face':
            video_path += '_original_face'
            output_size = (self.opt.face_size, self.opt.face_size)
        elif self.opt.camera_frame_type == 'eyes':
            video_path += '_eyes'
            output_size = (2*self.opt.eyes_size, self.opt.eyes_size)
        else:
            raise ValueError('Unknown camera frame type: %s' %self.opt.camera_frame_type)

        start = time.time()

        frame_path = os.path.join(video_path, "%05d.png"%index)
        frame = cv.imread(frame_path)
        # with open(frame_path, "rb") as f:
        #     frame = jpeg.decode_jpeg(f.read(), colorspace='bgr')

            # frame = np.zeros((128,256,3), dtype='float32')
            # print("timestamps",timestamps)
            # print("frames",frames.shape)
        # print(time.time() - start)
        
        # Collect and return
        frame = self.preprocess_frame(frame)
        
        if self.opt.camera_frame_type == 'eyes':
            ew = self.opt.eyes_size
            subentry['leye'] = frame[:, :, ew:]
            subentry['reye'] = frame[:, :, :ew]
            subentry['eyes'] = frame[:, :, :]
        elif self.opt.camera_frame_type == 'original_face':
            subentry['face'] = frame[:, :, :]
            
            # print(frames.shape[0],subentry["facial_landmarks"].shape[0])
          
            eye_corner = self.handle_eye_corner(subentry["facial_landmarks"])
            subentry["eye_corner"] = eye_corner.astype(np.float32)

        # Pad as necessary with zero value and zero validity

        # print("hhhhhhhhhhhhh", time.time() - start)
        

        return subentry

    def __getitem__(self, idx):
        # Retrieve sub-folder specification
        spec = self.all_imgs[idx]
        path = spec['full_path']
        h5_path = spec['h5_path']
        source = spec['camera_name']
        index = spec['index']

        # Check cache if requested
        # NOTE: this only works with num_workers=0 as otherwise memory is not shared nor persisted.
    

        # Grab all data
        
        entry = self.load_all_from_source(h5_path, path, source, index)
        
        # Add meta data
        entry['participant'] = spec['participant']
        entry['subfolder'] = spec['subfolder']
        entry['camera'] = spec['camera_name']

        torch_entry = dict([
            (k, torch.from_numpy(a)) if isinstance(a, np.ndarray) else (k, a)
            for k, a in entry.items()
        ])
    
        return torch_entry


