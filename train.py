from Dataloader import Train_DataLoader
import torch
import sys
import torch.nn as nn
import time
import numpy as np
import cv2
import os
from SSIM import SSIM
from model import Autoencodermodel
from tqdm import tqdm

import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gc
import math
from losses import MMDLoss
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser(description='Autoencoder Training')
parser.add_argument('--base_lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--datasets_dir', type=str, default="SAM")
parser.add_argument('--model_name', type=str, default="AE-SAM")
parser.add_argument('--model_exit', type=str, default="model/")

args = parser.parse_args()

ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
model = Autoencodermodel()
if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)



## Load the dataset
Dataset = Train_DataLoader(args.datasets_dir)
Datasets = Dataset.datasets_names
traindataloader = torch.utils.data.DataLoader(Dataset, batch_size=args.batch_size, shuffle=True, num_workers=8  )
max_iterations = len(traindataloader) * args.epochs
optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.05)
criterion = SSIM()
mmd = MMDLoss(2)
iterations = 0

def adjust_learning_rate(current_iteration, max_iteration, lr_min=0, lr_max=0.001, warmup_iteration=500):
    lr=0.0
    if current_iteration <= warmup_iteration:
        lr = lr_max * current_iteration / warmup_iteration
    else:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((current_iteration - warmup_iteration) / max_iteration * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch):
    global iterations
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    losses = 0
    ssim_losses = 0
    mmd_losses = 0
    print('epoch : {}/{}'.format(epoch + 1, args.epochs))
    for feature, scimg, _, ds in tqdm(traindataloader):
        iterations += 1
        adjust_learning_rate(current_iteration=iterations, max_iteration=max_iterations, lr_min=0, lr_max=args.base_lr)
        feature = feature.float()
        scimg = scimg.float()
        feature, scimg, ds = feature.to(device), scimg.to(device), ds.to(device)
        optimizer.zero_grad()
        _, im_out, z = model(feature)
        ssim_loss = 1 - criterion(im_out,scimg)
        mmd_loss = 5 * mmd(z, ds)
        train_loss = ssim_loss + mmd_loss
        train_loss.backward()
        optimizer.step()

        losses += train_loss.data.cpu()
        ssim_losses += ssim_loss.data.cpu()
        mmd_losses += mmd_loss.data.cpu()


    losses = losses / len(traindataloader)
    ssim_losses = ssim_losses / len(traindataloader)
    mmd_losses = mmd_losses / len(traindataloader)

    print('total_loss : {:.6f}, ssim_loss : {:.6}, mmd_loss : {:.6}'.format(losses, ssim_losses, mmd_losses))
    torch.save(model, args.model_exit + args.model_name +str(epoch+1).zfill(3)+".mdl")


def save_feature():
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()
    print('saving...')
    x = []
    y = []
    dataset = []
    with torch.no_grad():
        for feature, _, label, ds in tqdm(traindataloader):
            feature = feature.float()
            feature = feature.to(device)
            r, _, _ = model(feature)
            r = r.cpu().squeeze().detach().numpy()
            for i in range(len(r)):
                x.append(r[i])
                y.append(label[i])
                dataset.append(ds[i])
    x = np.array(x)
    y = np.array(y)
    dataset = np.array(dataset)
    np.save(os.path.join('./feature', 'X.npy'), x)
    np.save(os.path.join('./feature', 'y.npy'), y)
    np.save(os.path.join('./feature', 'dataset.npy'), dataset)
    

if __name__=='__main__':
    for epoch in range(args.epochs):
        train(epoch)
    save_feature()
        
    

    
    

