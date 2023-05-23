from __future__ import print_function, division
import torch
from skimage import io, util
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import PIL

from scipy.ndimage.filters import gaussian_filter
from skimage import img_as_ubyte

import cv2

plt.ion()

def show_sample(image,label):
    plt.imshow(image)
    plt.pause(0.001)


def loadDatasetFiles(dataset_file):
    """
    Inputs:
    - dataset.txt
    Outputs:
    - dict('image','label') that contains list of files.
    """
    fd = open(dataset_file)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split("\t")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
    return {'image':image_filenames, 'label': label_filenames}

def loadDatasetWithDTFiles(dataset_file):
    """
    Outputs:
    - dict('image','label','dt') that contains list of files.
    """
    with open(dataset_file,'r') as fd:
        image_filenames, label_filenames, dt_filenames =[],[],[]
        for i in fd:
            i = i.strip().split("\t")
            image_filenames.append(i[0])
            label_filenames.append(i[1])
            dt_filenames.append(i[2])
    return {'image':image_filenames, 'label': label_filenames, 'dt': dt_filenames}


def padNParray(array, padding, fill_value=0):
    S = array.shape
    if len(S) == 1:
        cols = np.zeros(shape=(padding), dtype=array.dtype)
    else:
        new_size = (padding,) + S[1:]
        cols = np.zeros(shape=new_size, dtype=array.dtype)

    if(fill_value is not 0):
        cols.fill(fill_value)

    return np.concatenate((array, cols),0)

# Create custom dataset:
class ContourDataset(Dataset):
    """Contours dataset."""
    def __init__(self, dataset_file, transform=None):
        """
        Args:

        """
        self.dataset_file = dataset_file
        self.files = loadDatasetFiles(dataset_file)
        self.transform = transform

    def __len__(self):
        return len(self.files['image'])

    def __getitem__(self,idx):
        sample = self.loadSampleFromID(idx)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def loadSampleFromID(self,id):
        image_file = self.files['image'][id]
        label_file = self.files['label'][id]
        image = io.imread(image_file)
        label = io.imread(label_file)
        return {'image': image, 'label': label}

class ContourWithDTDataset(Dataset):
    """ Contours dataset with distance transform precomputed"""
    def __init__(self, dataset_file, transform=None):
        self.dataset_file= dataset_file
        self.files = loadDatasetWithDTFiles(dataset_file)
        self.transform = transform

    def __len__(self):
        return len(self.files['image'])

    def __getitem__(self, idx):
        sample = self.loadSampleFromID(idx)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def loadSampleFromID(self,id):
        image_file = self.files['image'][id]
        label_file = self.files['label'][id]
        dt_file = self.files['dt'][id]
        image = io.imread(image_file)
        label = io.imread(label_file)
        dt = io.imread(dt_file)
        return {'image': image, 'label': label, 'dt': dt}


class ContourDatasetAugmented(ContourDataset):
    """ Contour Dataset with Data augmentation
    Original size is left unchanged
    Rest of the dataset is augmented with:
    - rotation
    - scale
    - shear
    - vertical flip

    - add white gaussian noise
    - add gaussian blur
    - elastic transformation
    """
    def __init__(self, dataset_file, augmentation_scale = 5,transform=None, augmentation_transform=None):
        super().__init__(dataset_file,transform)
        self.augmentation_scale = augmentation_scale
        self.augmentation_transforms = augmentation_transform
        # Define index to separate sample from orginal dataset and augmented one
        self.idx_augmentation = len(self.files['image'])

    def __len__(self):
        return self.augmentation_scale * super().__len__()

    def __getitem__(self,idx):
        if idx < self.idx_augmentation:
            return super().__getitem__(idx)
        else:
            # convert idx into real files list
            real_id = idx % self.idx_augmentation
            sample = super().loadSampleFromID(real_id)

            if self.augmentation_transforms:
                # compose original transforms with augmentation ones:
                T = transforms.Compose([
                    self.augmentation_transforms,
                    self.transform
                    ])
                sample = T(sample)
            return sample


class ContourWithDTAugmentedDataset(ContourWithDTDataset):
    def __init__(self, dataset_file, augmentation_scale = 5,transform=None, augmentation_transform=None):
        super().__init__(dataset_file,transform)
        self.augmentation_scale = augmentation_scale
        self.augmentation_transforms = augmentation_transform
        # Define index to separe sample from orginal dataset and augmented one
        self.idx_augmentation = len(self.files['image'])

    def __len__(self):
        return self.augmentation_scale * super().__len__()

    def __getitem__(self,idx):
        if idx < self.idx_augmentation:
            return super().__getitem__(idx)
        else:
            # convert idx into real files list
            real_id = idx % self.idx_augmentation
            sample = super().loadSampleFromID(real_id)

            if self.augmentation_transforms:
                # compose original transforms with augmentation ones:
                T = transforms.Compose([
                    self.augmentation_transforms,
                    self.transform
                    ])
                sample = T(sample)
            return sample


# add transform :

class PadHeight(object):
    """Add pixel to the image"""

    def __init__(self, padding):
        self.padding = int(padding)
        self.fill = 0
        self.mode = 'edge'

    def __call__(self, sample):
        for k in sample.keys():
            s = ((0,self.padding),(0,0)) if sample[k].ndim == 2 else ((0,self.padding),(0,0),(0,0))
            sample[k] = np.pad(sample[k], s, self.mode)
        return sample

class ForceDT2(object):
    """Turn dt to 2 shape"""

    def __call__(self, sample):
        if 'dt' in sample.keys():
            sample['dt'] = sample['dt'][:,:,0:2]
        return sample

class StandardDT(object):
    """ convert groundtruth values in standard values x = (X - mean) / sigma """
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma if sigma != 0 else 1.0

    def __call__(self, sample):
        if 'dt' in sample.keys():
            sample['dt'] = (sample['dt']-self.mean) / self.sigma
        else:
            sample['label'] = (sample['label']-self.mean) / self.sigma
        return sample

class ToTensor(object):
    """ convert ndarrays in sample to Tensors"""

    def __call__(self,sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for k in sample.keys():
            if sample[k].ndim == 3:
                sample[k] = sample[k].transpose((2,0,1))
            sample[k] = torch.from_numpy(sample[k])
        return sample

class Normalize(object):
    def __init__(self, value=255, keys=[]):
        self.value = value
        self.keys = keys

    def __call__(self,sample):
        if self.keys == []:
            for k in sample.keys():
                sample[k] /= self.value
        else:
            for k in self.keys:
                sample[k] /= self.value
        return sample

class PreprocessImage(object):
    def __call__(self, sample):
        image = sample['image']
        m = image.mean()
        s = image.std()
        new_image = (image-m)/s
        new_image_scale = (new_image - new_image.min()) / (new_image.max() - new_image.min())
        sample['image'] = new_image_scale
        return sample


def computePadSize(size_initial, size_final):
    diff_h = size_final[0] - size_initial[0]
    diff_w = size_final[1] - size_initial[1]
    if (diff_h <0) or (diff_w<0):
        raise ValueError('Size after padding [{}] is smaller than original size [{}]'.format(size_final, size_initial))

    # mid_h = diff_h //2
    # mid_w = diff_w //2
    #
    # return ((mid_h, diff_h - mid_h), (mid_w, diff_w - mid_w))
    return ((0,diff_h), (0,diff_w))

def center_crop(image, desired_size):
    diff_h = image.shape[0] - desired_size[0]
    diff_w = image.shape[1] - desired_size[1]

    mid_h = diff_h // 2
    mid_w = diff_w // 2

    return image[mid_h:mid_h + desired_size[0], mid_w:mid_w + desired_size[1]]

class CASENetPreProcess(object):
    def __init__(self,mean, pad_size):
        self.mean = mean
        self.pad_size = pad_size
    def __call__(self,sample):
        image = sample['image']
        label = sample['label']
        # cancel mean
        # crop input image and label
        # crop_size = (image.shape[0], self.pad_size)
        # image_crop = center_crop(image, crop_size)
        # label_crop = center_crop(label,crop_size)
        # apply mean shift
        image -= np.array(self.mean, dtype=image.dtype)
        # pad to have square image
        # pad_params = computePadSize(image_crop.shape[:2], (self.pad_size,)*2)
        # image_pad = np.pad(image_crop, pad_params+((0,0),), mode='constant', constant_values=0)
        # label_pad = np.pad(label_crop, pad_params, mode='constant', constant_values=0)

        return {'image':image, 'label':label}

class NormalizeImage(object):
    def __init__(self, value=255):
        self.value = value

    def __call__(self,sample):
        sample['image'] /= self.value
        return sample

class ToFloat32(object):
    """ Convert ndarrays from uint8 to float32"""

    def __call__(self,sample):
        image, label = sample['image'], sample['label']
        for k in sample.keys():
            sample[k] = sample[k].astype(np.float32)
        return sample

class ConvertLabel3to2Channels(object):
    """ Convert 3-channel arrays into 2-channel array"""
    def __init__(self,ind1=0, ind2=1):
        self.index_first = ind1
        self.index_second = ind2

    def __call__(self,sample):
        label = sample['label']
        new_label = np.concatenate(
            (np.expand_dims(label[:,:,self.index_first],2),
            np.expand_dims(label[:,:,self.index_second],2)),
            axis=2)
        sample['label'] = new_label
        return sample


class ConvertLabelToBinaryClass(object):
    """
    Input: [h,w,channels] with p in [0,255]
    Output: [h,w] with p in [0,1]
    """
    def __call__(self, sample):
        label = sample['label']
        if label.ndim == 3:
            label = label[:,:,0]
        bw = label == 255
        sample['label'] = bw * 1
        return sample

class ConvertHumanToMachineLabel(object):
    """
    Input: Image label loaded from file [H,W,3]
    with class1 = [255,0,0]
    with class2 = [0,255,0]
    with class3 = [0,0,255]
    with class4 = [255,150,0]
    with class5 = [150,0,150]
    with class6 = [0,150,255]
    Output: [H,W] with p in {0,1,2}
    """
    
    def __call__(self, sample):
        label =sample['label']
        assert label.ndim == 3, 'Error : label is not dim 3'
        newLabel = np.zeros(label.shape[:2],dtype=label.dtype)
#        mask1 = label[...,:] == [255,0,0]
#        mask2 = label[...,:] == [0,255,0]
#        mask3 = label[...,:] == [0,0,255]
#        mask4 = label[...,:] == [255,150,0]
#        mask5 = label[...,:] == [150,0,150]
#        mask6 = label[...,:] == [0,150,255]
        
        mask1 = np.logical_and(label[...,0] == 255, np.logical_and(label[...,1] == 0 , label[...,2] == 0))
        mask2 = np.logical_and(label[...,0] == 0, np.logical_and(label[...,1] == 255, label[...,2] == 0))
        mask3 = np.logical_and(label[...,0] == 0, np.logical_and(label[...,1] == 0, label[...,2] == 255))
        mask4 = np.logical_and(label[...,0] == 255, np.logical_and(label[...,1] == 150, label[...,2] == 0))
        mask5 = np.logical_and(label[...,0] == 150, np.logical_and(label[...,1] == 0, label[...,2] == 0))
        mask6 = np.logical_and(label[...,0] == 0, np.logical_and(label[...,1] == 0, label[...,2] == 150))
        mask7 = np.logical_and(label[...,0] == 150, np.logical_and(label[...,1] == 0, label[...,2] == 150))
        mask8 = np.logical_and(label[...,0] == 0, np.logical_and(label[...,1] == 150, label[...,2] == 255))
        mask9 = np.logical_and(label[...,0] == 255, np.logical_and(label[...,1] == 255, label[...,2] == 0))
        mask10 = np.logical_and(label[...,0] == 0, np.logical_and(label[...,1] == 255, label[...,2] == 255))
        mask11 = np.logical_and(label[...,0] == 255, np.logical_and(label[...,1] == 0, label[...,2] == 255))
        mask12 = np.logical_and(label[...,0] == 255, np.logical_and(label[...,1] == 255, label[...,2] == 255))
        
        newLabel[mask1] = 1
        newLabel[mask2] = 2
        newLabel[mask3] = 3
        newLabel[mask4] = 4
        newLabel[mask5] = 5
        newLabel[mask6] = 6
        newLabel[mask7] = 7
        newLabel[mask8] = 8
        newLabel[mask9] = 9
        newLabel[mask10] = 10
        newLabel[mask11] = 11
        newLabel[mask12] = 12
        sample['label'] = newLabel
        return sample
    
    #    def __call__(self, sample):
#        label =sample['label']
#        assert label.ndim == 3, 'Error : label is not dim 3'
#        newLabel = np.zeros(label.shape[:2],dtype=label.dtype)
#        mask1 = label[:,:,0] == 255
#        mask2 = label[:,:,1] == 255
#        newLabel[mask1] = 1
#        newLabel[mask2] = 2
#        sample['label'] = newLabel
#        return sample

class ConvertLabelToMultiClass(object):
    """
    Input: [h,w,n_class] with p in [0,255]
    Output: [h,w] with p in [0, n_class]
    """
    def __init__(self, n_class):
        self.n_class= n_class
    def __call__(self, sample):
        label = sample['label']
        assert label.ndim == 3, 'ERROR: Label is not dim 3'
        newLabel = np.zeros((label.shape[:2]))
        for c in range(0,self.n_class):
            # Note : does nothing for class 0 'trash'but ..
            bw = label[:,:,c] == 255
            newLabel[bw] = c
        sample['label'] = newLabel
        return sample

class ForceLabel2D(object):
    """
    Make sure that labels is 2D image
    """
    def __call__(self, sample):
        label = sample['label']
        if label.ndim != 2:
            sample['label'] = label[:,:,0]
        return sample

class AddNegativeChannel(object):
    """
    Add a last channel which pixel = 1 if other channel are all 0; 0 else

    label should be [H,W,2] numpy array with values = {0,255}
    """
    def __init__(self, in_front=False):
        self.in_front = in_front
    def __call__(self,sample):
        label = sample['label']
        # create zero mask to capture negative of the two channels
        Z = np.zeros((label.shape[0], label.shape[1]),dtype=label.dtype)
        output_mask = np.full((Z.shape[0], Z.shape[1]), False)
        for i in range(0,label.shape[2]):
            mask = Z == label[:,:,i]
            output_mask = np.logical_or(mask,output_mask)

        last_channel = np.expand_dims(output_mask*255, 2)
        if self.in_front:
            # put 'negative' channel in front
            label = np.concatenate((last_channel, label), axis=2)
        else:
            #put 'negative' channel in the back
            label = np.concatenate((label, last_channel), axis=2)
        sample['label'] = label
        return sample

class ConvertSampleNumpyToPIL(object):
    def __call__(self, sample):
        for k in sample.keys():
            sample[k] = PIL.Image.fromarray(np.uint8(sample[k]))
        return sample

class ConvertSamplePILToNumpy(object):
    def __call__(self,sample):
        for k in sample.keys():
            sample[k] = np.asarray(sample[k], np.uint8)
        return sample

class DownScale(object):
    """
    Downscale PIL Image
    """
    def __init__(self, value):
        self.scale = value
    def __call__(self, sample):
        image = sample['image']
        newH = int(float(image.size[0]) / self.scale)
        newW = int(float(image.size[1]) / self.scale)
        for k in sample.keys():
            sample[k] = sample[k].resize((newH,newW), PIL.Image.ANTIALIAS)
        return sample

### Data augmentation:
class RandomRotate(object):
    """
    Apply random rotation for label and label
    """
    def __init__(self, degrees, probability):
        self.max_rot = degrees
        self.probability = probability

    def __call__(self,sample):
        r = round(random.uniform(0,1),1)
        if r <= self.probability:
            return self.do(sample)
        else:
            return sample

    def do(self, sample):
        val_rot = random.randrange(-self.max_rot, +self.max_rot)
        for k in sample.keys():
            sample[k] = transforms.functional.rotate(sample[k], val_rot)
        return sample

class RandomShear(object):
    """
    Apply random Shear to image
    """
    def __init__(self, angle, proba):
        self.max_angle = angle
        self.probability = proba

    def do(self,sample):
        random_angle = random.randrange(-self.max_angle, +self.max_angle)
        for k in sample.keys():
            sample[k] = transforms.functional.affine(sample[k], 0, [0,0], 1.0, random_angle)
        return sample

    def __call__(self,sample):
        r = round(random.uniform(0,1),1)
        if r <= self.probability:
            return self.do(sample)
        else:
            return sample
        
class RadialDistortion(object):
    """
    Apply a radial distortion to both images and labels
    """
    def __init__(self, proba):
        self.probability = proba
            
    def __call__(self,sample):
        r = round(random.uniform(0,1),1)
        if r <= self.probability:
            return self.do(sample)
        else:
            return sample
    
    def do(self,sample):
        positive = False
        r2 = round(random.uniform(0,1),1)
        if r2 <= self.probability:
            distortion_coefficients = np.array([0.1,0.5,0.05,0,0])
            positive = True
        else:
            distortion_coefficients = np.array([-0.1,-0.1,0,0,0])
            positive = False
        
        camera_matrix = np.array([[247.632,0,222.58325],[0,248.1842,151.5792],[0,0,1]])
        image = sample['image']
        label = sample['label']
        
        new_image = np.zeros_like(image)
        new_label = np.zeros_like(label)
        new_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, camera_matrix)
        new_label = cv2.undistort(label, camera_matrix, distortion_coefficients, None, camera_matrix)
#        
#       If distortion is positive, scale and crop image to delete extra black pixels:
        if positive:
            scale_percent = 120 # percent of original size
            width = int(new_label.shape[1] * scale_percent / 100)
            height = int(new_label.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            new_image = cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)
    
            #Crop scaled image to delete added black parts:
            new_image = new_image[(height//2 - image.shape[0]//2):(height//2 + image.shape[0]//2), (width//2 - image.shape[1]//2):(width//2 + image.shape[1]//2)]
           
            # resize image
            new_label = cv2.resize(new_label, dim, interpolation = cv2.INTER_AREA)
    
            #Crop scaled image to delete added black parts:
            new_label = new_label[(height//2 - label.shape[0]//2):(height//2 + label.shape[0]//2), (width//2 - label.shape[1]//2):(width//2 + label.shape[1]//2)]
           

        sample['image'] = new_image
        sample['label'] = new_label
        return sample

class RandomVerticalFlip(object):
    """
    Flip image left to right randomly
    """
    def __init__(self, proba):
        self.proba = proba
    def __call__(self,sample):
        r = round(random.uniform(0,1),1)
        if r <= self.proba:
            return self.do(sample)
        else:
            return sample

    def do(self,sample):
        for k in sample.keys():
            sample[k] = transforms.functional.vflip(sample[k])
        return sample

class RandomRescale(object):
    """
    Apply small scale change
    """
    def __init__(self, proba):
        self.proba = proba

    def __call__(self,sample):
        r = round(random.uniform(0,1),1)
        if r <= self.proba:
            return self.do(sample)
        else:
            return sample
    def do(self,sample):
        initial_shape = sample['image'].shape
        # apply upscale:
        random_scale = round(random.uniform(1.1, 1.3),2)
        random_size = [int(random_scale * x) for x in initial_shape]
        randomResize = transforms.Resize(random_size)
        reshape_back = transforms.CenterCrop(initial_shape)

        for k in sample[k]:
            sample[k] = randomResize(sample[k])
            sample[k] = reshape_back(sample[k])

        return sample

class AddGaussianNoise(object):
    """
    Random gaussian noise;
    sample should be numpy image
    """
    def __init__(self, noise_var = 0.01):
        self.noise_var = noise_var

    def __call__(self, sample):
        seed = random.randint(0,200)
        new_image_float = util.random_noise(sample['image'], 'gaussian',
            seed=seed, clip=True, mean=0, var=self.noise_var)
        # rando_noise converts image (uint8)[0,255] into (float64)
        # as we will converts image to float after all, we just need to put the image back in [0,255]
        sample['image'] = 255 * new_image_float
        return sample

class AddGaussianBlur(object):
    """
    Apply a gaussian blur to image input
    """
    def __init__(self, sigma=2.0):
        self.sigma= sigma
    def __call__(self,sample):
        image = sample['image']
        new_image = np.zeros_like(image)
        if image.ndim == 3:
            for c in range(image.shape[2]):
                new_image[:,:,c] = gaussian_filter(image[:,:,c], sigma=self.sigma)
        else:
            new_image = gaussian_filter(image, sigma=self.sigma)
        sample['image'] = new_image
        return sample

def getLabelsRepartition(label, n_class):
    """
    Label: torch tensor [H,W] wit values in [0, n_class-1]
    """
    repartition = torch.zeros((1,n_class), dtype=torch.int64)
    for i in range(n_class):
        pixel_class = label[:,:] == i
        repartition[0,i] = torch.sum(pixel_class.view(-1))
    return repartition

def CountClassInbalance(dataloader, n_class, normalized=True):
    total = torch.zeros((1,n_class),dtype=torch.int64)
    for i, data in enumerate(dataloader,0):
        labels = data['label']
        #print("Labels shape: {}".format(labels.shape))
        for j in range(labels.shape[0]):
            total += getLabelsRepartition(labels[j,:], n_class)
    if normalized:
        norm = torch.sum(total).to(torch.float32)
        if norm != 0:
            total = total.to(torch.float32) / norm
        else:
            total = 0
    return total

def WeightFromClassRepartition(class_repartition):
    M = torch.median(class_repartition)
    weights = torch.zeros_like(class_repartition)
    for i in range(class_repartition.shape[1]):
        weights[0,i] = M / class_repartition[0,i]
    return weights

def computeMeanData(dataloader, h=272, w=480):
    """
    Compute mean of a whole set.
    """
    D = len(dataloader.dataset)
    b_size = dataloader.batch_size
    C = torch.zeros((D,h,w))

    for i,sample in enumerate(dataloader):
        r = i * b_size
        c = sample['label'].shape[0]
        C[r:r+c, :,:] = sample['label']

    M = torch.mean(C)
    sigma = torch.std(C)
    return M,sigma


def weightsFrom2Class(repartition):
    # weight from "Holistically-Nested Edge Detection"
#    beta = repartition[0,1] / np.sum(repartition)
#    return beta / (1 - beta)

    return repartition[0,1] / repartition[0,0]

def ContoursDetectionTransform(n_class):
    if n_class == 2:
        return transforms.Compose([
                    PadHeight(2),
                    AddNegativeChannel(in_front=True),
                    ConvertLabelToBinaryClass(),
                    ToFloat32(),
                    NormalizeImage(),
                    ToTensor()])
    else:
        return transforms.Compose([
            PadHeight(2),
            ConvertHumanToMachineLabel(),
            ToFloat32(),
            NormalizeImage(255),
            # PreprocessImage(),
            ToTensor()])

def ContoursDetectionTransformCASENet():
    return transforms.Compose([
        ConvertHumanToMachineLabel(),
        ToFloat32(),
        CASENetPreProcess(mean=[129.75, 75.07, 72.96], pad_size=472),
        ToTensor()])
# SGB mean = [104.008, 116.669, 122.675]
#samell+ mean = [129.75, 75.07, 72.96]

def ContoursPretextTransform():
    return transforms.Compose([
        PadHeight(2),
        ConvertLabelToBinaryClass(),
        ToFloat32(),
        PreprocessImage(),
        ToTensor()])

def RegressionOneClassTransform():
    return transforms.Compose([
    ForceLabel2D(),
    PadHeight(2),
    ToFloat32(),
    Normalize(255,['image', 'dt']),
    ToTensor()
    ])

def RegressionStandardTransform(M,sigma):
    return transforms.Compose([
    ForceLabel2D(),
    PadHeight(2),
    ToFloat32(),
    Normalize(255,['image', 'dt']),
    ToTensor(),
    StandardDT(M,sigma)
    ])

def RegressionDoubleClassTransform():
    return transforms.Compose([
    ForceDT2(),
    ConvertHumanToMachineLabel(),
    PadHeight(2),
    ToFloat32(),
    Normalize(255,['image', 'dt']),
    ToTensor()
    ])

def AugmentationTransforms():
    return transforms.Compose([
    ConvertSampleNumpyToPIL(),
    RandomRotate(10,1.0),
    RandomShear(10, 0.5),
#    RandomVerticalFlip(0.3),
    ConvertSamplePILToNumpy(),
    AddGaussianNoise(0.005),
    AddGaussianBlur(2.0),
    RadialDistortion(0.5)
    ])


if __name__ == '__main__':

    import visdom
    # vis = visdom.Visdom()
    # train_dataset = ContourDataset(dataset_file='/home/encov/DATA/all_frames/dataset_contours/train/dataset.txt',
    #     transform=ToTensor())

    dataset = ContourDataset(dataset_file='/home/encov/DATA/all_frames/annotated_contours_uterus/train/dataset.txt',
        transform=ContoursDetectionTransform(3)
        )

    dataset_augmented = ContourDatasetAugmented(dataset_file='/home/encov/DATA/all_frames/annotated_contours_uterus/train/dataset.txt',
    augmentation_scale = 5, transform=ContoursDetectionTransform(3), augmentation_transform=AugmentationTransforms())

    dataset_dt = ContourWithDTAugmentedDataset(dataset_file='/home/encov/DATA/all_frames/annotated_tdt_uterus/train/dataset.txt',
    augmentation_scale= 5, transform=ContoursDetectionTransform(3), augmentation_transform=AugmentationTransforms())

    print("Augmented dataset size: {}".format(len(dataset_augmented)))
    print("DT dataset : {}".format(len(dataset_dt)))

    dataloader = DataLoader(dataset_dt, batch_size=4, shuffle=True, num_workers=4)

    # sample = next(iter(dataloader))

    print("Evaluating class inbalance : ")
    print("Dataset : {}".format(dataset.dataset_file))
    print("Dataset size: {}".format(len(dataset)))
    print("Dataloader batch_size: {}".format(dataloader.batch_size))

    total= CountClassInbalance(dataloader, 3, True)
    print("Class repartition: {}".format(total))

    weight = WeightFromClassRepartition(total)
    print("Weight for each class (normalized): {}".format(weight))

    sample = next(iter(dataloader))
    print(sample['image'].shape)
    print(sample['label'].shape)
    print(sample['dt'].shape)

    import matplotlib.pyplot as plt
    image = sample['image'][0,:]
    label = sample['label'][0,:]
    dt = sample['dt'][0,:]
    # plt.imshow(dt.numpy().transpose(1,2,0))
    # plt.imshow(255*label.numpy())
    # plt.show(10)

    # vis.image(sample['dt'][0])
