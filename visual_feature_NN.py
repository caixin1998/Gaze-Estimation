from data import xgaze_dataset
import numpy as np
import random
import torch
from options.train_options import TrainOptions
import torchvision
import os
def KNN(selected_feature, features, k = 63):
    diffMat = np.tile(selected_feature,(features.shape[0]),1) - features
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    return sortedDistIndicies[:k]

output_path = "visual_feature_NN/xgaze"
os.makedirs(output_path, exist_ok=True)
opt = TrainOptions().parse()    
features = np.load("xgaze_features.npy")
xgaze = xgaze_dataset(opt)
for i in range(100):
    i = random.randint(0, len(features) - 1)
    knn_indicies = KNN(features[i], features)
    knn_indicies = [i] + knn_indicies
    image_tensor = torch.tensor(64, 3, 224,224)
    for k, idx in enumerate(knn_indicies):
        image_tensor[k] = xgaze[idx]["image"]
    torchvision.utils.save_image(os.path.join(output_path,"%05d.png"%i))    