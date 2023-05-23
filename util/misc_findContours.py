#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:00:17 2022

@author: sharib
"""

import cv2
import numpy as np

# R = [255,0,0], cType, contoursArray, contourCounter, imageRidge , coordsRidge, label_image, filteredImage
def findCountures(color, cType, contoursArray, contourCounter, imageSilhouette , coordsSilhouette, label_image, filteredImage):
    conn = 4
    import skimage
    from  skimage import measure
    
    for c in coordsSilhouette:
        imageSilhouette[c] = label_image[c]     # 这里是重新将image_Ridge设置了下；
        
    # imageSilhouetteGray = cv2.cvtColor(imageSilhouette, cv2.COLOR_BGR2GRAY) 
    # imageSilhouetteGray = cv2.cvtColor(imageSilhouetteGray, cv2.COLOR_BGR2GRAY) > 0
    # imageSilhouetteGray = morphology.remove_small_objects(imageSilhouetteGray, 1500)
    # imageSilhouetteGray = measure.label(imageSilhouetteGray)
    # Get biggest connected components and insert them in the final filtered image:
    imageSilhouetteGray = cv2.cvtColor(imageSilhouette, cv2.COLOR_BGR2GRAY)
    # imageSilhouetteGray = cv2.cvtColor(imageSilhouette, cv2.COLOR_BGR2GRAY) > 0
    # imageSilhouetteGray = morphology.remove_small_objects(imageSilhouetteGray, 1500)
    # imageSilhouetteGray = morphology.remove_small_holes(imageSilhouetteGray, 50)
    # 该函数输入为一个二值化图像，输出为一个长为4的tuple -
    # 第一个是连通区域的个数，第二个是一整张图的label，第三个是(x, y, width, height, area)，即每个区域的左上角坐标,宽和高，面积；第四个是每个连通区域的中心点
    # labels : labels是一个与image一样大小的矩形（labels.shape = image.shape），其中每一个连通区域会有一个唯一标识，标识从0开始
    _, labels, stats, _ = cv2.connectedComponentsWithStats(imageSilhouetteGray, conn, cv2.CV_32S)
    # print(stats)
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    indexes_group = np.argsort(stats[:, cv2.CC_STAT_AREA])     # cv2.CC_STAT_AREA - 连接组件的总面积(以像素为单位)
    
    # props = skimage.measure.regionprops(labels, imageSilhouetteGray)
    # properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']
    # props[index].label
    # 下面的做法是不要背景的联通区域么？还是说如果连通区域太多的话就减少一些（小的）连通域，因为可能有些连通域明显不属于对的预测？但实际上我们应该不可以减少，尤其对于下缘R和上缘S来说；
    stats = stats[indexes_group]
    if len(indexes_group) > 6:
        rangeval = len(indexes_group)  - 4
    elif len(indexes_group) > 3:
        rangeval = len(indexes_group)  - 2
    else:
        rangeval = len(indexes_group)  - 1
        
    if cType == 'Ligament':
        rangeval = 1
        
    # if cType == 'Silhouette':
    #     rangeval = 2
            
    for componentCount in range(0,rangeval):
        imagePointsX = []
        imagePointsY = []
        # if props[componentCount].area > 1500:
        for x in range(0,labels.shape[1]-1):
            for y in range(0,labels.shape[0]-1):
                if labels[y,x] == indexes_group[(indexes_group.shape[0]-2)-componentCount]:
                    
                    filteredImage[y,x,:] = color 
                    
                    if len(imagePointsX) == 0:
                        imagePointsX.append(x)
                        imagePointsY.append(y)
                    else:
                        imagePointsX.append(x)
                        imagePointsY.append(y)
        contoursArray.append( {"contourType": cType, "imagePoints": {'x':imagePointsX, 'y': imagePointsY}})
        contourCounter += 1
            
    return contoursArray, contourCounter, filteredImage


def find3DCountures(cur_pred, cType, contoursArray, contourCounter):     # # R->1, L->2, background->0；
    verticePoints = []
    for i in range(len(cur_pred)):

        if cType == 'Ridge':
            if cur_pred[i] == 1:
                verticePoints.append(i)
        if cType == 'Ligament':
            if cur_pred[i] == 2:
                verticePoints.append(i)
    contoursArray.append({"contourType": cType, "modelPoints": {'vertices': verticePoints}})
    contourCounter += 1

    return contoursArray, contourCounter