import argparse
import os
import cv2
import torch
import yaml
# from ignite.contrib import metrics
import torchvision
import torchvision.transforms as transforms
from PIL import Image
# import PLT
import matplotlib.pyplot as plt

from torcheval.metrics import BinaryAUROC
from sklearn.metrics import roc_auc_score
import sklearn
import constants as const
import dataset
import fastflow
import utils
import numpy as np
import sys
import time
from torch.utils.data import Dataset, DataLoader
from glob import glob

j = 0
weight = const.WEIGHT_PATH

data_dir = const.DATASET_PATH



def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
   
    
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )

    parser.add_argument(
        "--divide_num", type=int, default=0, help="image devied"
    )
    args = parser.parse_args()
    return args



args = parse_args()

CFG = {
    'input_size': const.INPUT_SIZE,
    'backbone_name': 'resnet18',
    'flow_step': 8,
    'hidden_ratio': 1.0,
    'conv3x3_only': True,
    'batch_size' : 64

}

class CustomDataset(Dataset):
    def __init__(self, root, input_size, is_train=True):
        

  
        self.img_size = input_size

        if is_train:

            self.image_files = glob(
                os.path.join(root, "train", "good", "*.png")
            )


        else:
            self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        
        self.is_train = is_train





    def __getitem__(self, index):
        image_file = self.image_files[index]
        


   
        image = cv2.imread(image_file)
        image = cv2.resize(image, (self.img_size,self.img_size))
    
            

        path = image_file.split('\\')

        path = path[1:]
        
        img = image/255.


        if path[-2] == 'good':
            label = 0
        else:
            label = 1


        if self.is_train:
            return img
        else:
            return img, label, path


    def __len__(self):
        return len(self.image_files)


train_loader = CustomDataset(data_dir, CFG["input_size"], is_train =True)
test_loader = CustomDataset(data_dir, CFG["input_size"], is_train =False)


train_dataloader = DataLoader(train_loader, batch_size = CFG['batch_size'], shuffle = True)
test_dataloader = DataLoader(test_loader, batch_size = CFG['batch_size'], shuffle = True)




def build_model(): 
    model = fastflow.FastFlow(
        backbone_name=CFG["backbone_name"],
        flow_steps=CFG["flow_step"],
        input_size = int(CFG["input_size"]),
        conv3x3_only=CFG["conv3x3_only"],
        hidden_ratio=CFG["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )



import sys



args = parse_args()


model = build_model()
checkpoint = torch.load(weight)
model.load_state_dict(checkpoint["model_state_dict"])

model.cuda()



from sklearn.metrics import accuracy_score


scores = []
labels = []


good_scores = []



def find_theshold(model):
    model.eval()


    for batch_images in train_dataloader:   # images.shape (8배치사이즈)   
        
    
        batch_images = batch_images.permute(0,3,2,1)
        batch_images = batch_images.float()
        batch_images = batch_images.cuda()

        with torch.no_grad():
            ret = model(batch_images)  #divide_num 갯수만큼의 이미지 모델에 넣고 output출력

        
        # outputs = ret["anomaly_map"].detach()
        preds = ret['preds']

        score = torch.mean(torch.stack(preds), 0).cpu().numpy()
        good_scores.extend(score.tolist())

       
        

    threshold  = np.percentile(good_scores, 98)

    return threshold



def eval_one_image(model, threshold, path):

    img  = cv2.imread(path)
    #img = cv2.resize(img, (self.img_size,self.img_size))
    img = cv2.resize(img, (const.INPUT_SIZE,const.INPUT_SIZE))
    img = img/255.
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    img = img.permute(0,3,2,1)
    img = img.float()
    img = img.cuda()

    with torch.no_grad():
        ret = model(img)

    pred = ret['preds']
    score = torch.mean(torch.stack(pred), 0).cpu().numpy()
    print(' score is ======> ',score)
    if score > threshold:
        print('bad')

    else:
        print('good')

        



def eval_once(model):
    model.eval()



    for batch_images in train_dataloader:   # images.shape (8배치사이즈)   
        
    
        batch_images = batch_images.permute(0,3,2,1)
        batch_images = batch_images.float()
        batch_images = batch_images.cuda()

        with torch.no_grad():
            ret = model(batch_images)  #divide_num 갯수만큼의 이미지 모델에 넣고 output출력

        
        # outputs = ret["anomaly_map"].detach()
        preds = ret['preds']

        score = torch.mean(torch.stack(preds), 0).cpu().numpy()
        good_scores.extend(score.tolist())

       
        

    threshold  = np.percentile(good_scores, 98)

    for batch_images, batch_labels ,path in test_dataloader:   # images.shape (8배치사이즈)   
        
        
        # print(batch_images.shape)
    
        batch_images = batch_images.permute(0,3,2,1)
        batch_images = batch_images.float()
        batch_images = batch_images.cuda()
        


        with torch.no_grad():
            ret = model(batch_images)  #divide_num 갯수만큼의 이미지 모델에 넣고 output출력

        
        # outputs = ret["anomaly_map"].detach()
        preds = ret['preds']

       

        score = torch.mean(torch.stack(preds), 0).cpu().numpy()


        scores.extend(score.tolist())
        labels.extend(batch_labels.cpu().numpy().tolist())
        
    

  
        
    pred_labels = [1 if x > threshold else 0 for x in scores]
    print(pred_labels[:20])
    print(labels[:20])



    acc = accuracy_score(labels,pred_labels )

    
    print('accuracy_score : ',acc)



eval_once(model)
        
# threshold = find_theshold(model)
# image_test= r'C:\jaehyuk\FastFlow\window_v3\test\good\20230118_143237_result.png'
# print('threshold : ',threshold)
# eval_one_image(model, threshold, image_test)