# Define model

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature_rgb(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # print('x shape', x.size())
    if idx is None:
        if dim9 == False:
            # idx = knn(x, k=k)   # (batch_size, num_points, k)
            # Use normXYZ + RGB + SWIR + geo for knn search
            idx = knn(x[:,3:], k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    
    # only use RGB for graph feature
    x = x[:,6:9,:] # RGB
    
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

def get_graph_feature_swir(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # print('x shape', x.size())
    if idx is None:
        if dim9 == False:
            # idx = knn(x, k=k)   # (batch_size, num_points, k)
            # Use normXYZ + RGB + SWIR + geo for knn search
            idx = knn(x[:,3:], k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    
    # only use SWIR for graph feature
    x = x[:,9:153,:] # SWIR
    
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

def get_graph_feature_geo(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # print('x shape', x.size())
    if idx is None:
        if dim9 == False:
            # idx = knn(x, k=k)   # (batch_size, num_points, k)
            # Use normXYZ + RGB + SWIR + geo for knn search
            idx = knn(x[:,3:], k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    
    # only use geo for graph feature
    x = x[:,153:181,:] # geo
    
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args_k
        
        self.bn1_rgb = nn.BatchNorm2d(64)
        self.bn2_rgb = nn.BatchNorm2d(64)

        self.bn1_swir = nn.BatchNorm3d(4)
        self.bn2_swir = nn.BatchNorm3d(4)
        self.bn3_swir = nn.BatchNorm2d(64)
        self.bn4_swir = nn.BatchNorm2d(64)
        
        self.bn1_geo = nn.BatchNorm2d(64)
        self.bn2_geo = nn.BatchNorm2d(64)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)        
        self.bn6 = nn.BatchNorm2d(64)


        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        
        
        # RGB
        self.conv1_rgb = nn.Sequential(nn.Conv2d(dim_rgb*2, 64, kernel_size=1, bias=False),
                                   self.bn1_rgb,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_rgb = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2_rgb,
                                   nn.LeakyReLU(negative_slope=0.2))

        # SWIR
        self.conv1_swir = nn.Sequential(nn.Conv3d(1, 4, kernel_size=(32,1,1), bias=False),
                                   self.bn1_swir,
                                   nn.LeakyReLU(negative_slope=0.2))      
        self.conv2_swir = nn.Sequential(nn.Conv3d(4, 4, kernel_size=(32,1,1), bias=False),
                                   self.bn2_swir,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_swir = nn.Sequential(nn.Conv2d(dim_swir*2, 64, kernel_size=1, bias=False),
                                   self.bn3_swir,
                                   nn.LeakyReLU(negative_slope=0.2))      
        self.conv4_swir = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4_swir,
                                   nn.LeakyReLU(negative_slope=0.2))

        
        # geo
        self.conv1_geo = nn.Sequential(nn.Conv2d(dim_geo*2, 64, kernel_size=1, bias=False),
                                   self.bn1_geo,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_geo = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2_geo,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # all
        self.conv1 = nn.Sequential(nn.Conv2d(196*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args_dropout)
        self.conv9 = nn.Conv1d(256, args_num_class, kernel_size=1, bias=False) # CHANGE TO NUMBER OF CLASSES
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # RGB
        x_rgb = get_graph_feature_rgb(x, k=self.k, dim9=False)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k) 
        x_rgb = self.conv1_rgb(x_rgb)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_rgb = self.conv2_rgb(x_rgb)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_rgb = x_rgb.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        # SWIR
        x_swir3d = get_graph_feature_swir(x, k=self.k, dim9=False)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x_swir3d = x_swir3d.unsqueeze(1)
        x_swir3d = self.conv1_swir(x_swir3d)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_swir3d = self.conv2_swir(x_swir3d)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_swir3d = x_swir3d.max(dim=2,keepdim=False)[0]
        x_swir3d = x_swir3d.max(dim=-1,keepdim=False)[0]
        
        x_swir = get_graph_feature_swir(x, k=self.k, dim9=False)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x_swir = self.conv3_swir(x_swir)
        x_swir = self.conv4_swir(x_swir)
        x_swir = x_swir.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        # geo
        x_geo = get_graph_feature_geo(x, k=self.k, dim9=False)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x_geo = self.conv1_geo(x_geo)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_geo = self.conv2_geo(x_geo)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_geo = x_geo.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        x_all_cat = torch.cat((x_rgb, x_swir3d, x_swir, x_geo), dim=1) # RGB + SWIR + geo # 192 + swir3d features
        
        x_all_graph_0 = get_graph_feature(x_all_cat, k=self.k)
        x_all_1 = self.conv1(x_all_graph_0)
        x_all_2 = self.conv2(x_all_1)
        x_all_2_max = x_all_2.max(dim=-1, keepdim=False)[0]
        
        x_all_graph_1 = get_graph_feature(x_all_2_max, k=self.k)
        x_all_3 = self.conv3(x_all_graph_1)
        x_all_4 = self.conv4(x_all_3)
        x_all_4_max = x_all_4.max(dim=-1, keepdim=False)[0]
        
        x_all_graph_2 = get_graph_feature(x_all_4_max, k=self.k)
        x_all_5 = self.conv5(x_all_graph_2)
        x_all_6 = self.conv6(x_all_5)
        x_all_6_max = x_all_6.max(dim=-1, keepdim=False)[0]
        
        x_cat = torch.cat((x_all_2_max, x_all_4_max, x_all_6_max), dim=1)
        
        x_fc1 = self.conv7(x_cat)                       # (batch_size, 64*3, num_points) -> (batch_size, 512, num_points)
        x_fc2 = self.conv8(x_fc1)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x_dp = self.dp1(x_fc2)
        x_end = self.conv9(x_dp)                       # (batch_size, 256, num_points) -> (batch_size, num_class, num_points)
        
        return x_end
