import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datasets.dataset_bloodcell import Bloodcell_dataset,RandomGenerator
from importlib import import_module
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import math
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='SAM finetuning')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--rank', type=int, default=4)
parser.add_argument('--base_lr', type=float, default=0.0001)
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--warmup_epoch', type=int, default=10)
parser.add_argument('--vit_name', type=str, default="vit_b")
parser.add_argument('--checkpoint', type=str, default="model/sam_vit_b_01ec64.pth")
parser.add_argument('--datasets_dir', type=str, default="SAM/datasets/Bloodcell")
parser.add_argument('--exit_path', type=str, default="SAM/output")

args = parser.parse_args()
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset = Bloodcell_dataset(base_dir=os.path.join(args.datasets_dir,'Bloodcell_train.dat.gz'),transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[args.img_size//4, args.img_size//4])]))
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
test_dataset = Bloodcell_dataset(base_dir=os.path.join(args.datasets_dir,'Bloodcell_test.dat.gz'),transform=transforms.Compose(
                                    [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[args.img_size//4, args.img_size//4])]))
test_trainloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

train_means,train_stds=train_dataset.get_mean_std()
sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.checkpoint, pixel_mean=train_means,
                                                                pixel_std=train_stds)

model = LoRA_Sam(sam, 4).cuda()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.1)

if args.n_gpu > 1:
    model = nn.DataParallel(model)
model.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes + 1)
history=[]

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.001, warmup_epoch=5):
    lr=0.0
    if current_epoch <= warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((current_epoch - warmup_epoch) / max_epoch * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

if __name__=='__main__':
    for epoch_num in range(args.epochs):
        epoch_loss=0
        model.train()
        adjust_learning_rate(optimizer=optimizer,current_epoch=epoch_num+1,max_epoch=args.epochs,lr_min=0.0,lr_max=args.base_lr,warmup_epoch=args.warmup_epoch)
        print('epoch :',epoch_num+1)
        for sampled_batch in tqdm(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            outputs = model(image_batch, True, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.cpu().detach().numpy()

        model.eval()
        test_loss=0
        for sampled_batch in test_trainloader:
            with torch.no_grad():
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
                low_res_label_batch = sampled_batch['low_res_label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                low_res_label_batch = low_res_label_batch.cuda()
                outputs = model(image_batch, True, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                test_loss+=loss.cpu().detach().numpy()
        print('train_loss : {} , test_loss : {}'.format(epoch_loss/len(trainloader),test_loss/len(test_trainloader)))
        history.append([epoch_loss/len(trainloader),test_loss/len(test_trainloader)])
        np.save(os.path.join(args.exit_path,'sam_train_loss.npy'),history)
        save_interval = 5
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(args.exit_path, 'epoch_' + str(epoch_num + 1) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            print("save model to {}".format(save_mode_path))

