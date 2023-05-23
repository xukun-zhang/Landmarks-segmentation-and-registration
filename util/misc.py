#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:45:48 2022

@author: sharib
"""

import torch
from model import UNet
import os
import argparse
import numpy as np
import torch.nn as nn
import cv2


def loadMHAfile(image_data):
    img = np.zeros((image_data.shape[1], image_data.shape[2], 3), dtype=np.float32)
    img[:,:,0] = image_data[0,:,:]
    img[:,:,1] = image_data[1,:,:]
    img[:,:,2] = image_data[2,:,:]
    ImgOrig = cv2.rotate(img.astype('uint8'), cv2.ROTATE_90_CLOCKWISE)
    ImgOrig = np.fliplr(ImgOrig).astype('uint8')
    
    return ImgOrig

def find_rgb(img, r_query, g_query, b_query):
    coordinates= []
    for x in range(0,img.shape[0]-1):
        for y in range(0,img.shape[1]-1):
            r, g, b = img[x,y]
            if r == r_query and g == g_query and b == b_query:
                # print("{},{} contains {}-{}-{} ".format(x, y, r, g, b))
                coordinates.append((x, y))
    return(coordinates)

def getColors():     # 这三种颜色的顺序分别为： R，L，S，即下缘、镰状韧带、上缘，将我们的输出换为这个顺序。
    """
    List of RGB colors for representing liver classes
    """
    # For MICCAI challenge:
    colors = [
    torch.tensor([0,0,0],dtype=torch.uint8),
    torch.tensor([255,0,0],dtype=torch.uint8),
    torch.tensor([0,0,255],dtype=torch.uint8),
    torch.tensor([255,255,0],dtype=torch.uint8),
    ]
    return colors

def convertFromOneHot(T_one_hot):     # 2D的输出为： torch.Size([4, 272, 480])
    """ T_one_hot [b,c,h,w] with values in {0,1}--> T[b,h,w] with values in {0,...,c-1}"""
    if T_one_hot.dim() == 4:
        return torch.argmax(T_one_hot,dim=1)
    elif T_one_hot.dim() == 3:
        return torch.argmax(T_one_hot,dim=0)
    else:
        return torch.argmax(T_one_hot, dim=0)
    
def createVisibleLabel(label):     # torch.Size([4, 256, 512]) 顺序为B\R\S\L， 但里面的值还是小数；
    """
    Goal: match each class index with 'real' colors

    Label: [H,W] with values in {0, ..., n_class-1}
    return [H,W,C] image tensor uint8
    """

    if label.dim() == 3:
        # case contours is one_hot:
        label = convertFromOneHot(label)     # 转换为了 torch.Size([272, 480]) 里面值为0，1，2，3
        # print("这里label的维度应该为两维？", label.shape)
    M = int(torch.max(label.view(-1)))
    image = torch.zeros((label.shape[0], label.shape[1],3), dtype=torch.uint8)     # 创建了一个空的图像；
    tem_lab = torch.zeros((label.shape[0], label.shape[1], 3), dtype=torch.uint8)  # 创建了一个空的图像；
    if M == 0:
        return image.byte(), tem_lab.byte()     # 全黑的图，什么都没预测得到；
    colors = getColors()
    for i in range(M+1):     # 这里从0遍历到4；

        mask = label == i     # 遍历的第一个是背景，背景的channel中大部分像素的值在这里都应该为0，那么对应mask中为true：即第一个遍历得到的是一个（256，512）的mask，里面背景部分为true；
        # in some case: label has empty class
        if mask.sum() != 0:
            if image[mask].numel() != 0:
                image[mask] = colors[i]     # 只要有预测出对应的label，即0-4（0、1、2、3）分别设置对应的颜色；基于预测的结果，分别设置3通道image对应的颜色；
                if i == 0:
                    tem_lab[mask] = colors[i]
                if i == 1:
                    tem_lab[mask] = torch.tensor([1,1,1],dtype=torch.uint8)
                if i == 2:
                    tem_lab[mask] = torch.tensor([2,2,2],dtype=torch.uint8)
                if i == 3:
                    tem_lab[mask] = torch.tensor([3,3,3],dtype=torch.uint8)
    return image.byte(), tem_lab.byte()


def convertContoursToImage(label, tensor=True):     # 2D的输出为： torch.Size([4, 256, 512]) 顺序为B\R\S\L， 但里面的值还是小数；
    """
    Label: [H,W] with values in {0, ..., n_class-1}
    return [H,W,C] image
    """
    image, tem = createVisibleLabel(label)
    return image.numpy(), tem.numpy()
