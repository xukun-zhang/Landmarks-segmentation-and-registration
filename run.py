#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:31:49 2022

@author: sharib
"""

import SimpleITK
from pathlib import Path
import random
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import numpy as np
import json
from model import UNet
from util.misc import convertContoursToImage, find_rgb
import cv2
import shutil
from scipy.ndimage import zoom
import open3d as o3d
import cv2 as cv

def cameraIntrinsicMat(data_params):
    Knew=[]
    Knew.append([float(data_params['fx']), float(data_params['skew']), float(data_params['cx'])])
    Knew.append([ 0, float(data_params['fy']), float(data_params['cy'])])
    Knew.append([float(data_params['p1']), float(data_params['p2']), 1])
    
    return np.asmatrix(Knew)
def tranform_matrix(point_set):
    r_x, r_y, r_z = random.randint(0, 360), random.randint(0, 360), random.randint(0, 360)
    o, p, q = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
    Rx = np.array([[1, 0, 0], [0, np.cos(r_x), np.sin(r_x)], [0, -np.sin(r_x), np.cos(r_x)]])
    Ry = np.array([[np.cos(r_y), 0, -np.sin(r_y)], [0, 1, 0], [np.sin(r_y), 0, np.cos(r_y)]])
    Rz = np.array([[np.cos(r_z), np.sin(r_z), 0], [-np.sin(r_z), np.cos(r_z), 0], [0, 0, 1]])
    Ao = np.array([[o, 0, 0], [0, 1, 0], [0, 0, 1]])
    Ap = np.array([[1, 0, 0], [0, p, 0], [0, 0, 1]])
    Aq = np.array([[1, 0, 0], [0, 1, 0], [0, 0, q]])

    R = np.dot(np.dot(np.dot(np.dot(np.dot(Rx, Ry), Rz), Ao), Ap), Aq)

    point_set = point_set.transpose(1,0)

    point_set = np.dot(R, point_set)
    point_set = point_set.transpose(1, 0)
    return point_set

execute_in_docker = False     # Run this program through python instead of in a docker environment
useOnly2DSeg = 1 # Set flag for 2D segmentation only
useOnly3DSeg = 1 # state:1 for all three (2D seg, 3D seg)
useReg = 0 # state : 0 for 3D to 2D registration

class P2ILFChallenge(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            # Reading all input files 
            input_path = Path("/input/images/laparoscopic-image/") if execute_in_docker else Path("./test/images/laparoscopic-image/"),
            output_file = [Path("/output/2d-liver-contours.json"), Path("/output/3d-liver-contours.json"), Path("/output/transformed-3d-liver-model.obj")]if execute_in_docker else [Path("./output/2d-liver-contours.json"), Path("./output/3d-liver-contours.json"), Path("./output/transformed-3d-liver-model.obj")]
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if execute_in_docker:
            # path_model = "./ckpt/CP50_criteria.pth"     # 提交时注释这句话
            path_model = "/opt/algorithm/ckpt/CP50_criteria.pth"
            # threeD_model = "/opt/algorithm/ckpt/best_model_model.pth"
            threeD_model_R = "/opt/algorithm/ckpt/best_model_R_epoch.pth"
            threeD_model_L = "/opt/algorithm/ckpt/best_model_L_epoch.pth"
            # twoD_model = "/opt/algorithm/ckpt/twoD_best_model.pth"
            twoD_model = "/opt/algorithm/ckpt/twoD_best_model.pth"
        else:
            path_model = "./ckpt/CP50_criteria.pth"
            threeD_model_R = "./ckpt/best_model_R_epoch.pth"
            threeD_model_L = "./ckpt/best_model_L_epoch.pth"
            twoD_model = "./ckpt/twoD_best_model.pth"

            
        self.num_classes = 4
        self.model = UNet(n_channels=3, n_classes=self.num_classes , upsample=False)
        self.model.load_state_dict(torch.load(path_model))
        print("XXXXX   model weight loaded")
        """Load the point cloud segmentation model"""
        self.thrD_model_R = torch.load(threeD_model_R, map_location=torch.device('cpu'))
        self.thrD_model_R = self.thrD_model_R.eval()
        self.thrD_model_L = torch.load(threeD_model_L, map_location=torch.device('cpu'))
        self.thrD_model_L = self.thrD_model_L.train()


        """Load 2D UNet for 2D anatomical curve segmentation"""
        self.thD_model = torch.load(twoD_model, map_location=torch.device('cpu'))
        self.thD_model = self.thD_model.eval()

        
    '''Instruction 1: YOU will need to work here for writing your results as per your task'''
    def save(self):
        if (useOnly2DSeg == 1):
            with open(str(self._output_file[0]), "w") as f:
                json.dump(self._case_results[0][0], f)  

        if useOnly3DSeg==1 or useReg ==1:
            # Hint you can append the results for 2D and 3D segmentation contour dictionaries
            for i in range(1, 2):
                with open(str(self._output_file[i]), "w") as f:
                    json.dump(self._case_results[0][i], f)



        if useReg ==0:
            image_contours = self._case_results[0][0]
            obj_contours = self._case_results[0][1]
            R_2D_x, R_2D_y, L_2D_x, L_2D_y = [], [], [], []
            for i in range(image_contours["numOfContours"]):
                if image_contours["contour"][i]["contourType"] == "Ridge":
                    R_2D_x.extend(image_contours["contour"][i]["imagePoints"]["x"])
                    R_2D_y.extend(image_contours["contour"][i]["imagePoints"]["y"])
                if image_contours["contour"][i]["contourType"] == "Ligament":
                    L_2D_x.extend(image_contours["contour"][i]["imagePoints"]["x"])
                    L_2D_y.extend(image_contours["contour"][i]["imagePoints"]["y"])

            R_3D_v, L_3D_v = [], []
            for i in range(obj_contours["numOfContours"]):
                if obj_contours["contour"][i]["contourType"] == "Ridge":
                    R_3D_v.extend(obj_contours["contour"][i]["modelPoints"]["vertices"])
                if obj_contours["contour"][i]["contourType"] == "Ligament":
                    L_3D_v.extend(obj_contours["contour"][i]["modelPoints"]["vertices"])


            if execute_in_docker:
                input_path_mesh = Path('/input/3D-liver-model.obj')
                input_path_K = Path('/input/acquisition-camera-metadata.json')
            else:
                input_path_mesh = Path('./test/3D-liver-model.obj')
                input_path_K = Path('./test/acquisition-camera-metadata.json')
            f = open(input_path_K)  # camera parameters
            data_params = json.load(f)
            K = cameraIntrinsicMat(data_params)

            """Extract all points in the 3D mesh"""
            with open(input_path_mesh) as file:
                v_points = []
                while 1:
                    line = file.readline()
                    if not line:
                        break
                    strs = line.split(" ")
                    if strs[0] == "v":
                        v_points.append([float(strs[1]), float(strs[2]), float(strs[3])])


            point_3D_R = []     # save Ridge points (3D)
            for i in range(len(R_3D_v)):
                point_3D_R.append(v_points[R_3D_v[i]])
            point_3D_L = []     # save Ligament points (3D)
            for i in range(len(L_3D_v)):
                point_3D_L.append(v_points[L_3D_v[i]])


            """Extract all points in the 2D curve"""
            point_2D_R = []     # save Ridge points (2D, same number with 3D points.)
            min_y, max_y = np.array(R_2D_y).min(), np.array(R_2D_y).max()
            point_2D_R.append([R_2D_x[R_2D_y.index(min_y)], R_2D_y[R_2D_y.index(min_y)]])
            for i in range(1, len(point_3D_R)):
                number = int(min_y + (max_y - np.array(R_2D_y).min()) * (i/len(point_3D_R)))
                if number in R_2D_y:
                    point_2D_R.append([R_2D_x[R_2D_y.index(number)], R_2D_y[R_2D_y.index(number)]])
                    min_y = np.array(R_2D_y).min()
                else:
                    new_R_y = list(np.array(R_2D_y) - number)
                    new_R_y = list(map(abs, new_R_y))
                    number = np.array(new_R_y).min()
                    point_2D_R.append([R_2D_x[new_R_y.index(number)], R_2D_y[new_R_y.index(number)]])


            point_2D_L = []     # save Ligament points (2D, same number with 3D points.)
            min_x, max_x = np.array(R_2D_x).min(), np.array(R_2D_x).max()
            point_2D_L.append([R_2D_x[R_2D_x.index(min_x)], R_2D_y[R_2D_x.index(min_x)]])
            for i in range(1, len(point_3D_L)):
                number = int(min_x + (max_x - np.array(R_2D_x).min()) * (i / len(point_3D_L)))
                if number in R_2D_x:
                    point_2D_L.append([R_2D_x[R_2D_x.index(number)], R_2D_y[R_2D_x.index(number)]])
                    min_x = np.array(R_2D_x).min()
                else:
                    new_R_x = list(np.array(R_2D_x) - number)
                    new_R_x = list(map(abs, new_R_x))
                    number = np.array(new_R_x).min()
                    point_2D_L.append([R_2D_x[new_R_x.index(number)], R_2D_y[new_R_x.index(number)]])

            new_2D, new_3D = [], []
            if len(point_2D_L) > 5:
                for i in range(len(point_2D_L) - 1):
                    if point_2D_L[i + 1] != point_2D_L[i]:
                        new_2D.append(point_2D_L[i])
                        new_3D.append(point_3D_L[i])

            print("L - len(new_2D), len(new_3D)：", len(new_2D), len(new_3D))
            if len(point_2D_R) > 5:
                for i in range(len(point_2D_R) - 1):
                    if point_2D_R[i + 1] != point_2D_R[i]:
                        new_2D.append(point_2D_R[i])
                        new_3D.append(point_3D_R[i])

            print("R - len(new_2D), len(new_3D)：", len(new_2D), len(new_3D))

            """If the previous model does not predict the anatomical points, then the 3D model under the input path is copied to the output path."""
            if len(new_3D) < 3:
                shutil.copyfile('./input/3D-liver-model.obj', self._output_file[2])
                print("The previous model does not predict the anatomical points. The 3D model under the input path is copied to the output path")


            """PnP Solver"""
            point1_3D = np.array(new_3D).astype(np.float64)
            point2_2D = np.array(new_2D).astype(np.float64)

            flag, R, t = cv.solvePnP(point1_3D, point2_2D, K, None)
            R, Jacobian = cv.Rodrigues(R)


            source_mesh = o3d.io.read_triangle_mesh("./input/3D-liver-model.obj")
            source_mesh.rotate(R, center=(0, 0, 0))
            source_mesh.translate((t[0][0], t[1][0], t[2][0]))
            o3d.io.write_triangle_mesh("./output/transformed-3d-liver-model.obj", source_mesh)
            print("Write the transformed 3D mesh successfully!")

            """The second method of writing into the transformed 3D mesh"""
            # import openmesh as om
            # mesh = om.read_trimesh("./input/3D-liver-model.obj")
            # verts = mesh.points()
            # faces = mesh.face_vertex_indices()
            #
            # verts = verts.dot(np.transpose(R))
            # T = [t[0][0], t[1][0], t[2][0]]
            # verts += T
            # mesh_new = om.TriMesh()
            # mesh_new.add_vertices(verts)
            # mesh_new.add_faces(faces)
            # om.write_mesh('./output/mesh_new.obj', mesh_new)




        if useReg:
            print('\n useReg need equal 0!')

        print('\n All files written successfully')



    '''Instruction 2: YOU will need to work here for appending your 2D and 3D contour results'''
    def process_case(self, *, idx, case):
        results_append=[]
        input_image, input_image_file_path = self._load_input_image(case=case)
        results1 = self.predict(input_image=input_image)
        results_append.append(results1)

        if useOnly3DSeg:
            # Write resulting candidates to result.json for this case
            results2 = self.predict2()

            results_append.append(results2)
        return results_append
  
    
  # Sample provided - write your 3D contour prediction ehre
    def predict2(self):

        from util.misc_findContours import find3DCountures
        # Hard coded paths
        if execute_in_docker:
            input_path_mesh = Path('/input/3D-liver-model.obj')

        else:
            input_path_mesh = Path('./test/3D-liver-model.obj')

          

        with open(input_path_mesh) as file:
            v_points = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    v_points.append((float(strs[1]), float(strs[2]), float(strs[3])))

        # normalization
        x, y, z = [], [], []
        points_list = []
        for i in range(len(v_points)):
            x.append(float(v_points[i][0]))
            y.append(float(v_points[i][1]))
            z.append(float(v_points[i][2]))
            points_list.append([float(v_points[i][0]), float(v_points[i][1]), float(v_points[i][2])])
        x, y, z = sorted(x), sorted(y), sorted(z)

        c_p = [x[int(len(x) / 2)], y[int(len(y) / 2)], z[int(len(z) / 2)]]  # point - c_p
        max_d = [x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]]

        save_points = []
        for i in range(len(v_points)):
            new_x = (float(v_points[i][0]) - c_p[0]) / max_d[0]
            new_y = (float(v_points[i][1]) - c_p[1]) / max_d[1]
            new_z = (float(v_points[i][2]) - c_p[2]) / max_d[2]
            save_points.append([new_x, new_y, new_z])



        save_points = torch.Tensor(save_points)
        save_points = save_points.unsqueeze(0)

        save_points = save_points.float()
        save_points = save_points.transpose(2, 1)
        with torch.no_grad():
            output_R, _ = self.thrD_model_R(save_points, torch.eye(1).unsqueeze(0))
            output_L, _ = self.thrD_model_L(save_points, torch.eye(1).unsqueeze(0))

        cur_pred_R = output_R.squeeze(0).cpu().data.numpy()
        cur_pred_L = output_L.squeeze(0).cpu().data.numpy()

        cur_pred_L = np.argmax(cur_pred_L, 1)
        with open("pre_L"+ ".txt", "w", encoding='utf-8') as file:
            for i in range(len(points_list)):
                file.write(str(points_list[i][0]) + ' ' + str(points_list[i][1]) + ' ' + str(
                    points_list[i][2]) + ' ' + str(cur_pred_L[i] + 1) + '\n')
            file.close()

        cur_pred = np.argmax(cur_pred_R, 1)
        with open("pre_R"+ ".txt", "w", encoding='utf-8') as file:
            for i in range(len(points_list)):
                file.write(str(points_list[i][0]) + ' ' + str(points_list[i][1]) + ' ' + str(
                    points_list[i][2]) + ' ' + str(cur_pred[i] + 1) + '\n')
            file.close()

        with open("pre_input"+ ".txt", "w", encoding='utf-8') as file:
            for i in range(len(points_list)):
                file.write(str(points_list[i][0]) + ' ' + str(points_list[i][1]) + ' ' + str(
                    points_list[i][2]) + ' ' + str(1) + '\n')
            file.close()
        cur_pred_L[cur_pred_L==1] = 2
        cur_pred = cur_pred + cur_pred_L
        cur_pred[cur_pred == 3] = 1



        contoursArray = []
        contourCounter3D = 0;
        cType = 'Ridge'
        contoursArray, contourCounter3D = find3DCountures(cur_pred, cType, contoursArray, contourCounter3D)
        cType = 'Ligament'
        contoursArray, contourCounter3D = find3DCountures(cur_pred, cType, contoursArray, contourCounter3D)
        my_dictionary_3D = {"numOfContours": int(contourCounter3D), "contour": contoursArray}

        print("Successful calculation of 3D anatomical curve! ")
        return my_dictionary_3D
        
        
    ''' Instruction 3: YOU will need to write similar functins for your 2D, 3D and registration - 
    these predict functions can be called by process_case for logging in results -> Tuple[SimpleITK.Image, Path]:'''
    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        from util.misc_findContours import findCountures
        
        image = SimpleITK.GetArrayFromImage(input_image)
        new_image = np.zeros(image.shape)
        new_image[:, :, 0] = image[:, :, 2]
        new_image[:, :, 1] = image[:, :, 1]
        new_image[:, :, 2] = image[:, :, 0]
        image = new_image


        image = np.array(image)
        shape = image.shape


        Image_resized = zoom(image, (256/shape[0], 512/shape[1], 1), order=2)
        Image_resized = Image_resized / 255.0
        Image_resized = torch.Tensor(Image_resized)
        Image_resized = Image_resized.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = self.thD_model(Image_resized)

        output_b, output_R, output_S, output_L = output[0, 0, :, :].unsqueeze(0), output[0, 1, :, :].unsqueeze(0), output[0, 2, :, :].unsqueeze(0), output[0, 3, :, :].unsqueeze(0)
        output = torch.cat((output_b, output_R, output_L, output_S), dim=0).unsqueeze(0)

        label_image, tem_lab = convertContoursToImage(output.squeeze())
        label_image = cv2.resize(label_image, (image.shape[1], image.shape[0]))

        contoursArray = []
        contourCounter = 0;


        imageRidge = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)
        coordsRidge = find_rgb(label_image, 255,0,0)
        for c in coordsRidge:
            imageRidge[c] = label_image[c]

        filteredImage = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)

        cType = 'Ridge'
        contoursArray, contourCounter, filteredImage = findCountures([255,0,0], cType, contoursArray, contourCounter, imageRidge , coordsRidge, label_image, filteredImage)

        # Extract ligament masks:
        imageLigament = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)
        coordsLigament = find_rgb(label_image,0,0,255)

        cType = 'Ligament'
        contoursArray, contourCounter, filteredImage = findCountures([0,0,255], cType, contoursArray, contourCounter, imageLigament , coordsLigament, label_image, filteredImage)
        # Extract silhouette masks:
        imageSilhouette = np.zeros(shape=(label_image.shape[0],label_image.shape[1],3),dtype=np.uint8)
        coordsSilhouette = find_rgb(label_image,255,255,0)

        cType = 'Silhouette'
        contoursArray, contourCounter, filteredImage = findCountures([255,255,0], cType, contoursArray, contourCounter, imageSilhouette , coordsSilhouette, label_image, filteredImage)

        
        """
        3: Save your Output : /output/2d-liver-contours.json
        """
        my_dictionary = {"numOfContours": int(contourCounter), "contour": contoursArray}
        print("Successful calculation of 2D anatomical curve! ")
        return my_dictionary
        
        
if __name__ == "__main__":
    P2ILFChallenge().process()
        
