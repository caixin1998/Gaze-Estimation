import torch.nn as nn
import torch
from modules import *

'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class ItrackerFaceImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self, backbone = "resnet50"):
        super(ItrackerFaceImageModel, self).__init__()
        if backbone == "resnet50":
            self.model = resnet50(pretrained=True)
        if backbone == "resnet18":
            self.model = resnet18(pretrained=True)
        if backbone == "dilated":
            self.model = resnet50(pretrained=True, replace_stride_with_dilation = [True, True, True])
        elif backbone == "botnet":
            self.model = botnet()
        elif backbone == "rednet":
            self.model = RedNet(depth = 50)

        #self.features = models.vgg16(pretrained = True).features
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x


class ItrackerEyeImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self, backbone = "resnet50", dilated = False):
        super(ItrackerEyeImageModel, self).__init__()
        if backbone == "resnet50":
            self.model = resnet50(pretrained=True, replace_stride_with_dilation = [dilated, dilated, dilated])
        elif backbone == "resnet18":
            self.model = resnet18(pretrained=True, replace_stride_with_dilation = [dilated, dilated, dilated])
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):
    
    def __init__(self,backbone = "resnet50"):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerFaceImageModel(backbone = backbone)
        if backbone == "resnet18":
            ngf = 512
        else:
            ngf = 2048
        self.fc = nn.Sequential(
            nn.Linear(ngf, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class iTrackerModel(nn.Module):


    def __init__(self,backbone = "resnet50"):
        super(iTrackerModel, self).__init__()
        self.leyeModel = ItrackerEyeImageModel()
        self.faceModel = FaceImageModel(backbone = backbone)
        self.reyeModel = ItrackerEyeImageModel()
     
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*2048, 128),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+128 , 128),
            nn.ReLU(inplace=True),
            nn.Linear(128 , 2),
            )
       

    def forward(self, faces, eyesLeft, eyesRight):
        # Eye nets
        xEyeL = self.leyeModel(eyesLeft)
        xEyeR = self.reyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)

        # Cat all
        x = torch.cat((xEyes, xFace), 1)
        features = torch.cat((xEyes, xFace), 1)
        x = self.fc(x)
        
        return x

class iTrackerMHSAModel(nn.Module):
    def __init__(self,backbone = "resnet50"):
        super(iTrackerMHSAModel, self).__init__()
        self.leyeModel = ItrackerEyeImageModel()
        self.faceModel = FaceImageModel(backbone = backbone)
        self.reyeModel = ItrackerEyeImageModel()
     
        # Joining both eyes
        self.leyesFC = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            )
        self.reyesFC = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            )
        self.mhsa = nn.MultiheadAttention(128, 4)
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(3 * 128 , 128),
            nn.ReLU(inplace=True),
            nn.Linear(128 , 2),
            )
       

    def forward(self, faces, eyesLeft, eyesRight):
        # Eye nets
        xEyeL = self.leyesFC(self.leyeModel(eyesLeft))[None]
        xEyeR = self.reyesFC(self.reyeModel(eyesRight))[None]
        # Cat and FC
        xFace = self.faceModel(faces)[None]
        features = torch.cat((xFace, xEyeL, xEyeR), 0)
        x,_ = self.mhsa(features,features,features)
        x = x.transpose(0,1).reshape(-1, 3 * 128)
        x = self.fc(x)
        return x

    
class EyeCornerModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(EyeCornerModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            )


    def forward(self,  x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class EyeImageModel(nn.Module):
    
    def __init__(self, backbone = "resnet50"):
        super(EyeImageModel, self).__init__()
        self.conv = ItrackerEyeImageModel(backbone=backbone)
        if backbone == "resnet18":
            ngf = 512
        else:
            ngf = 2048
        self.fc = nn.Sequential(
            nn.Linear(ngf, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class iTrackerECModel(nn.Module):


    def __init__(self, backbone = "resnet50"):
        super(iTrackerECModel, self).__init__()
        self.eyeModel = EyeImageModel()
        self.faceModel = FaceImageModel(backbone = backbone)
        self.ecModel = EyeCornerModel()
     
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*128, 128),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+128 , 128),
            nn.ReLU(inplace=True),
            nn.Linear(128 , 2),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(128+128+128 , 128),
            nn.ReLU(inplace=True),
            nn.Linear(128 , 2),
            )

    def forward(self, entry):
        # Eye nets
        if "leye" in entry and "reye" in entry:
            xEyeL = self.eyeModel(entry["leye"])
            xEyeR = self.eyeModel(entry["reye"])
        # Cat and FC
            xEyes = torch.cat((xEyeL, xEyeR), 1)
            xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(entry["face"])
        xECorner = self.ecModel(entry["ec"])

        # Cat all
        if "leye" in entry and "reye" in entry:
            x = torch.cat((xEyes, xFace, xECorner), 1)
            x = self.fc2(x)

        else:
            x = torch.cat((xFace, xECorner), 1)
            x = self.fc(x)

        
        return x

if __name__ == "__main__":
    a = torch.randn(1,3,224,224)
    b = torch.randn(1,3,224,224)
    c = torch.randn(1,3,224,224)
    model = iTrackerMHSAModel()
    e = model(a,b,c)
    print(e.shape)