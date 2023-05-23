import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):     # num_classes = 7;
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 8
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])   ### first idx is npoint
        self.sa2 = PointNetSetAbstractionMsg(512, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])   ## (npoint, radius, nsample, in_channel, mlp, group_all)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=135+additional_channel, mlp=[128, 128])   ### 150->135
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # print("----cls_label.shape:", cls_label.shape, cls_label)
        # Set Abstraction layers
        B,C,N = xyz.shape     # 4, 3, 2048;
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  ### l1_points:[batch,320,1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  ### l2_points:[batch,512,512]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  ### l3_points:[batch,1024,1]
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  ### l2_points:[batch,256,512]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)   ### l2_points:[batch,128,1024]
        cls_label_one_hot = cls_label.view(B,1,1).repeat(1,1,N)   # num_classes = 1
        l00_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l00_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):

        # weight_value = torch.Tensor([1,10,20]).float().cuda() #right_weight 用于背景、下缘、镰状韧带的分割，共三类；
        weight_value = torch.Tensor([1, 10]).float().cuda()  # right_weight 两类的任务；
        # weight_CE = weight_value.float().cuda()
        # print("pred[1:5, :]:", pred[1:5, :])
        # print("torch.softmax(pred, dim=1):", torch.softmax(pred, dim=1).shape)
        # print("torch.softmax(pred, dim=1):", torch.log_softmax(pred, dim=1)[1:5,:])
        # print("target:", target[1:5])
        total_loss = F.nll_loss(pred, target, weight = weight_value)


        # total_loss = F.CrossEntropyLoss(pred, target, weight = weight_value)
        # print("total_loss:", total_loss)
        return total_loss