#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:00:17 2022

@author: sharib
"""

import cv2
import numpy as np


def findCountures(color, cType, contoursArray, contourCounter, imageSilhouette , coordsSilhouette, label_image, filteredImage):
    conn = 4
    import skimage
    from  skimage import measure
    
    for c in coordsSilhouette:
        imageSilhouette[c] = label_image[c]
        
    # imageSilhouetteGray = cv2.cvtColor(imageSilhouette, cv2.COLOR_BGR2GRAY) 
    # imageSilhouetteGray = cv2.cvtColor(imageSilhouetteGray, cv2.COLOR_BGR2GRAY) > 0
    # imageSilhouetteGray = morphology.remove_small_objects(imageSilhouetteGray, 1500)
    # imageSilhouetteGray = measure.label(imageSilhouetteGray)
    # Get biggest connected components and insert them in the final filtered image:
    imageSilhouetteGray = cv2.cvtColor(imageSilhouette, cv2.COLOR_BGR2GRAY)
    # imageSilhouetteGray = cv2.cvtColor(imageSilhouette, cv2.COLOR_BGR2GRAY) > 0
    # imageSilhouetteGray = morphology.remove_small_objects(imageSilhouetteGray, 1500)
    # imageSilhouetteGray = morphology.remove_small_holes(imageSilhouetteGray, 50)
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(imageSilhouetteGray, conn, cv2.CV_32S)
    # print(stats)
    indexes_group = np.argsort(stats[:, cv2.CC_STAT_AREA])
    
    # props = skimage.measure.regionprops(labels, imageSilhouetteGray)
    # properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']
    # props[index].label
    
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
      