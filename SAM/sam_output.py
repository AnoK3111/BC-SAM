import numpy as np
import torch
import cv2
from tqdm import tqdm
import os
import gzip
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam
from einops import repeat
from torch.nn import functional as F
import argparse

parser = argparse.ArgumentParser(description='SAM Output')
parser.add_argument('--path', type=str, default="/home/anoke/data/images/Matek-19")
parser.add_argument('--checkpoint_sam', type=str, default="model/sam_vit_b_01ec64.pth")
parser.add_argument('--checkpoint_path', type=str, default="SAM/output/epoch_85.pth")
parser.add_argument('--file_ext', type=str, default="SAM/Matek-19.dat.gz")


args = parser.parse_args()
sam ,img_embedding_size= sam_model_registry["vit_b"](image_size=224,num_classes=1,pixel_mean=[215.0322, 187.7592, 200.4668],pixel_std=[41.3047, 56.9191, 21.9522],checkpoint=args.checkpoint_sam)
sam=sam
lora_sam = LoRA_Sam(sam, 4).cuda()
lora_sam.load_lora_parameters(args.checkpoint_path)

features = {}

files = [f for f in os.listdir(args.path) if not f.startswith('.')]

for file in files:
    images = [f for f in os.listdir(os.path.join(args.path,file)) if not f.startswith('.')]
    for image in tqdm(images):
        torch.cuda.empty_cache()
        img = cv2.imread(os.path.join(args.path, file,image))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(224,224))
        inputs = img.transpose(2, 0, 1)
        inputs =  inputs[np.newaxis,:,:,:]
        inputs = torch.FloatTensor(inputs).cuda()
        with torch.no_grad():
            outputs=lora_sam(inputs,multimask_output=True,image_size=224)
            output_masks = outputs['masks'] #(1,2,224,224)
            masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0) #(224,224)
            masks = masks.cpu().detach().numpy().astype(bool)
            low_res_logits=torch.softmax(outputs["low_res_logits"].squeeze(0), dim=0)[1].cpu().detach().numpy()
        features[image] = {
            "label":file,
            "masks": masks,
            "image": img,
            "feature": outputs['encoder_feature'].squeeze(0).cpu().detach().numpy()
        }


print("Saving...")
with gzip.open(args.file_ext, "wb") as f:
    pickle.dump(features, f)
print("Done")