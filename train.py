import torch
import json
import os
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from model import Model
import image_provider
from tqdm import tqdm
import math
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")
    
    
def iou_from_poly(pred, gt, width, height):
   
    width = int(width)
    height = int(height)
    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]
    masks[0] = draw_poly(masks[0], pred)
    masks[1] = draw_poly(masks[1], gt)

    return iou_from_mask(masks[0], masks[1])

def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou

def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, [poly], 255)

    return mask

def create_folder(path):
   # if os.path.exists(path):
       # resp = input 'Path %s exists. Continue? [y/n]'%path
    #    if resp == 'n' or resp == 'N':
     #       raise RuntimeError()
    
   # else:
     os.system('mkdir -p %s'%(path))
     print('Experiment folder created at: %s'%(path))
        
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')
    dataset_train = DataProvider(opts["train"],split='train',mode='train')
    dataset_val = DataProvider(opts["train_val"],split='train_val',mode='train_val')
    train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],
        shuffle=True, num_workers=opts['train']['num_workers'], collate_fn=image_provider.collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
        shuffle = False, num_workers=opts['train_val']['num_workers'], collate_fn=image_provider.collate_fn)
    
    return train_loader, val_loader

class Trainer(object):
    def __init__(self,args,opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts
        create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints'))

       # Copy experiment file
        os.system('cp %s %s'%(args.exp, self.opts['exp_dir']))

        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)
        self.model = Model(self.opts["input_height"],self.opts["input_width"],self.opts["input_channels"])
        self.model = self.model.to(device)
        self.loss_fn = nn.MSELoss()
        # Allow individual options
        self.optimizer = optim.Adam(self.model.parameters(),lr = self.opts['lr'],weight_decay=self.opts['weight_decay'])
        for name,param in self.model.named_parameters():
            print(name)
       
    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints', 'epoch%d_step%d.pth'\
        %(epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        self.model.reload(path)
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])

        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s'%(self.epoch, self.global_step, path)) 

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            self.save_checkpoint(epoch)
            self.train(epoch)
        
    def train(self, epoch):
        print('Starting training')
        self.model.train()
        losses = []
        accum = defaultdict(float)
        for step, data in enumerate(self.train_loader):     
            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()
        #self.model.train()
                self.save_checkpoint(epoch)
            img = data['img']
            gt = data['gt']            
            img = torch.cat(img)
            gt = torch.cat(gt).to(device)
            img = img.view(-1,self.opts["input_height"],self.opts["input_width"],self.opts["input_channels"])
            img = torch.transpose(img,1,3)
            img = torch.transpose(img,2,3)
            img = img.float()
            output = self.model(img.cuda())
            self.optimizer.zero_grad()
            loss = self.loss_fn(output,gt)           
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            accum['loss'] += float(loss.item())
            accum['length'] += 1
           
            if(step%self.opts['print_freq']==0):
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                print("[%s] Epoch: %d, Step: %d, Loss: %f"%(str(datetime.now()), epoch, self.global_step, accum['loss']))
                accum = defaultdict(float)
            del(output)
            self.global_step += 1
        avg_epoch_loss = 0.0
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
        avg_epoch_loss = avg_epoch_loss/len(losses)
        print("Average Epoch %d loss is : %f"%(epoch,avg_epoch_loss))
        
    def validate(self):
        print('Validating')
        self.model.eval()
        ious = []
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                img = data['img']
                gt = data['gt']
                gt = torch.cat(gt)
                img = torch.cat(img)
                img = img.view(-1,self.opts["input_height"],self.opts["input_width"],self.opts["input_channels"])
                img = torch.transpose(img,1,3)
                img = torch.transpose(img,2,3)
                img = img.float()
                output= self.model(img.cuda())
                gt = gt.numpy()
                output = output.cpu().numpy()
                gt_x2 = math.floor(min(gt[0,0]+gt[0,2]/2,1)*640)
                gt_y2= math.floor(min(gt[0,1]+gt[0,3]/2,1)*480)
                gt_x1 = math.floor(max(gt[0,0]-gt[0,2]/2,0)*640)
                gt_y1= math.floor(max(gt[0,1]-gt[0,3]/2,0)*480)
                pred_x2 = math.floor(min(output[0,0]+output[0,2]/2,1)*640)
                pred_y2 = math.floor(min(output[0,1]+output[0,3]/2,1)*480)
                pred_x1 = math.floor(max(output[0,0]-output[0,2]/2,0)*640)
                pred_y1 = math.floor(max(output[0,1]-output[0,3]/2,0)*480)
                iou = iou_from_poly([[pred_x1,pred_y1],[pred_x2,pred_y1],[pred_x2,pred_y2],[pred_x1,pred_y2]],[[gt_x1,gt_y1],[gt_x2,gt_y1],[gt_x2,gt_y2],[gt_x1,gt_y2]],640,480)
                ious.append(iou)
                del(output)
        avg_iou=0
        for i in ious:
            avg_iou+=i
        avg_iou = avg_iou/len(ious)
        print("Average iou is ",avg_iou)
        self.model.train()
if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    trainer = Trainer(args,opts)
    trainer.loop()
