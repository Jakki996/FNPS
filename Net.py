import torch.nn.functional as F
import torch.nn as nn
import torch
import ImageFeature
import TextFeature
import FuseAllFeature
from LoadData import *
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np
import os
import time
import itertools
from visdom import Visdom
import sklearn.metrics as metrics
import seaborn as sns
import Net
from torch.autograd import Variable, Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)
        
def grad_reverse(x):
    return ReverseLayerF.apply(x)

class FeatureExtractor(torch.nn.Module):
    def __init__(self,lstm_dropout_rate):
        super(FeatureExtractor,self).__init__()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN,lstm_dropout_rate)
        self.text_fuse = FuseAllFeature.ModalityFusion_1(512)
        self.image = ImageFeature.ExtractImageFeature()
        self.image_fuse= FuseAllFeature.ModalityFusion_1(1024)

    def forward(self,text_index,image_feature):
        text_result,text_seq = self.text(text_index,None)
        text_fusion = self.text_fuse(text_result,text_seq.permute(1,0,2))
        image_result,image_seq = self.image(image_feature)
        image_fusion = self.image_fuse(image_result,image_seq)
        return text_fusion,image_fusion

# class Encoder(torch.nn.Module):
#     def __init__(self):
#         super(Encoder,self).__init__()
#         self.encoder=nn.Sequential(
#             nn.Linear(256,128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128,64))
#     def forward(self,feature):
#         output=self.encoder(feature)
#         return output

# class Decoder(torch.nn.Module):
#     def __init__(self):
#         super(Decoder,self).__init__()
#         self.decoder=nn.Sequential(
#             nn.Linear(64,128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128,256))
#     def forward(self,feature):
#         output=self.decoder(feature)
#         return output

class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese,self).__init__()
        self.siamese_text= nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128))
        self.siamese_image= nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128))
    def forward(self,text_feature,image_feature):
        output1 = self.siamese_text(text_feature)
        output2 = self.siamese_image(image_feature)
        return output1,output2

class DomainClassifier(torch.nn.Module):
    def __init__(self):
        super(DomainClassifier,self).__init__()
        self.domain_classifier= nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.LogSoftmax(dim=1))

    def forward(self,feature):
        reverse_feature=grad_reverse(feature)
        domain_pred=self.domain_classifier(reverse_feature)
        return domain_pred

class LableClassifier(torch.nn.Module):
    def __init__(self):
        super(LableClassifier,self).__init__()
        self.lable_classifier= nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid())
    def forward(self,feature):
        lable_pred=self.lable_classifier(feature)
        return lable_pred

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive