import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist
import torch
# 密集差分会生成一个(n-1)！长度的向量，计算量太大了，先简化用一阶差分试试
# def df_process(vect):
#     ### chafenchuli
#     dflabel = []
#     n = len(vect)
#     for i in range(n):
#         flag = vect[i]
#         df = vect[i+1:]-flag
#         dflabel.append(df)
#
#     dfArr = np.hstack(dflabel)
#     return(dfArr)
#
# def first_order_df(vect):
#     ### 一阶差分处理
#     dflabel = []
#     n = len(vect)
#     for i in range(n - 1):
#         df = vect[i] - vect[i + 1]
#         dflabel.append(df)
#     # dfArr = torch.hstack(dflabel)
#     return (dflabel)

def structloss(pred,target,spoints,num_part):
    points = spoints
    cls = num_part
    batchsize = int(len(target) / points)

    pred_bs = pred.reshape(batchsize,points,cls)
    dfpred = torch.zeros((batchsize,points,cls),dtype = torch.float32).cuda()
    for j in range(batchsize):
        for k in range(points-1):
            dfpred[j,k,:] = torch.abs(pred_bs[j,k,:]-pred_bs[j,k+1,:])
            dfpred[j,k+1,:] = torch.abs(pred_bs[j,k+1,:]-pred_bs[j,0,:])
    c,h,w = np.shape(dfpred)
    df_pred = dfpred.reshape(c*h,w)

    target_bs = target.reshape(batchsize,points)
    dflabel = torch.zeros((batchsize,points),dtype = torch.int64).cuda()
    m, n = dflabel.size()
    for i in range(m):
        rlabel = target_bs[i]
        for j in range(n-1):
            x = torch.eye(num_part,dtype=torch.int)[rlabel[j],:].cuda()   ### one-hot
            y = torch.eye(num_part,dtype=torch.int)[rlabel[j+1],:].cuda()
            flag = torch.bitwise_xor(x,y)
            dflabel[i, j] = flag

        endflag = torch.bitwise_xor(torch.eye(num_part,dtype=torch.int)[rlabel[j+1],:],torch.eye(num_part,dtype=torch.int)[rlabel[0],:])
        dflabel[i,j+1] = endflag
    df_gt = dflabel.reshape(m*n)
    #
    #
    #
    # for i in range(points-1):
    #     dflabel[:,i] = target_bs[:,i] - target_bs[:,i+1]
    #     dflabel[:,i+1] = target_bs[:,i+1] - target_bs[:,0]
    # m,n = dflabel.size()
    # df_gt = dflabel.view(m*n)
    return df_pred,df_gt
# ########################
#
#     target_df = first_order_df(target_bs)
#
#     dlist = []
#     for i in range(batchsize):
#         pred_df = first_order_df(pred_bs[i])
#         target_df = first_order_df(target_bs[i])
#         vectCon = [pred_df,target_df]
#         d = pdist(vectCon,'braycurtis')
#         dlist.append(d)
#     distloss = torch.sum(torch.tensor(dlist))
#     return distloss

