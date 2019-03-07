import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os.path as osp
import cv2
import multiprocessing.dummy as multiprocessing
from torchvision import transforms
import random
def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}

    for key in keys:
        val = [item[key] for item in batch_list]

        t = type(batch_list[0][key])
        
        if t is np.ndarray:
            try:
                val = torch.from_numpy(np.stack(val, axis=0))
            except:
                # for items that are not the same shape
                # for eg: orig_poly
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated

class DataProvider(Dataset):
    """
    Class for the data provider
    """
    def __init__(self,opts, split='train', mode='train'):
        """
        split: 'train', 'train_val' or 'val'
        """
        self.opts = opts
        self.mode = mode        
        self.split = split
        self.read_dataset()

    def read_dataset(self):
        dataset = pd.read_csv('/home/jatin/flipkart/'+self.split+'.csv')
        print(len(dataset['image_name']))
        if self.mode!='test':
            #for step,i in enumerate(dataset['image_name']):
            self.image_name = dataset['image_name']
            self.x1 = dataset['x1']
            self.x2 = dataset['x2']
            self.y1 = dataset['y1']
            self.y2 = dataset['y2']
        else:
            #for step,i in enumerate(dataset['image_name']):
            self.image_name = dataset['image_name']
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        if self.mode!='test':
            return self.prepare_instance(idx)
        else:
            return self.prepare_test(idx)
    
    def prepare_instance(self, idx):
        image_name = self.image_name[idx]
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        y1 = self.y1[idx]
        y2 = self.y2[idx]
        results = {}
        img = cv2.imread('/home/jatin/flipkart/images/'+image_name)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        height = img.shape[0]
        width = img.shape[1]
        x_center = float(x1+x2)/2
        y_center = float(y1+y2)/2
        h = abs(y1-y2)/float(height)
        w = abs(x1-x2)/float(width)
        x_center = x_center/float(width)
        y_center = y_center/float(height)
        gt = [x_center,y_center,w,h]
        gt = torch.Tensor(gt)
        gt = gt.view(1,4)
        img = cv2.resize(img,(self.opts["input_height"],self.opts["input_width"]), interpolation = cv2.INTER_AREA)
        #color_jitter = transforms.ColorJitter(brightness=5, contrast=5)
        #grey_scale = transforms.Grayscale(3)
       # pil = transforms.ToPILImage()
        
        #if self.mode=="train":
          #  img = pil(img)
           # if random.randint(1,10)==1:
           #     img = color_jitter(img)
           # if random.randint(1,10)==1:
           #     img = grey_scale(img)
           # img = np.array(img)
           # if random.randint(1,10)==1:
           #     img = cv2.GaussianBlur(img,(3,3),0)
            
        #img = cv2.GaussianBlur(img,(5,5),0)
        img = torch.from_numpy(img)
        #img = img.float()
        results['image_name'] = image_name
        results['img'] = img
        results['gt']=gt
        results['h']=height
        results['w']=width        
        return results
    def prepare_test(self,idx):
        image_name = self.image_name[idx]
        results = {}
        img = cv2.imread('/home/jatin/flipkart/images/'+image_name)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(self.opts["input_height"],self.opts["input_width"]), interpolation = cv2.INTER_AREA)
        #img = cv2.GaussianBlur(img,(5,5),0)
        img = torch.from_numpy(img)
        img = img.float()
        results['img'] = img
        return results
