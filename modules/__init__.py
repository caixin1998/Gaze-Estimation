from .resnet import resnet50, resnet18
from .botnet import botnet
from .resnet50_scratch_dag import Resnet50_scratch_dag
from .discriminator import Discriminator
from .completion_net import CompletionNetwork,LocalDis,GlobalDis,PredictionEyeNetwork
from .fsanet import FSANet
from .iTrackerModel import iTrackerECModel
from .eye_net import EyeNet

#from .fsanet_64 import FSANet
__all__ = [
    'resnet18',
    'resnet50',
    'botnet',
    'Resnet50_scratch_dag',
    'Discriminator',
    'CompletionNetwork',
    'LocalDis',
    'GlobalDis',
    'FSANet',
    'PredictionEyeNetwork',
    'iTrackerECModel',
    'EyeNet'
]
