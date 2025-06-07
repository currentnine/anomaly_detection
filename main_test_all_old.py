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
from sklearn.metrics import accuracy_score
import sys


j = 0
weight = const.WEIGHT_PATH

data_dir = const.DATASET_PATH
constantPath = const.__file__
CFG = {
    'input_size': const.INPUT_SIZE,
    'backbone_name': 'resnet18',
    'flow_step': 8,
    'hidden_ratio': 1.0,
    'conv3x3_only': True,
    'batch_size' : 64

}
def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
   
    
    parser.add_argument("--eval_All", action="store_true", help="run eval only")
    parser.add_argument("--eval_One", action="store_true", help="run eval only")
    parser.add_argument("--image_path", type=str, help="path to the image file")


    args = parser.parse_args()
    return args



args = parse_args()


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

train_loader = CustomDataset(data_dir, CFG["input_size"], is_train =True)
test_loader = CustomDataset(data_dir, CFG["input_size"], is_train =False)


train_dataloader = DataLoader(train_loader, batch_size = CFG['batch_size'], shuffle = True)
test_dataloader = DataLoader(test_loader, batch_size = CFG['batch_size'], shuffle = True)





model = build_model()
checkpoint = torch.load(weight)
model.load_state_dict(checkpoint["model_state_dict"])

model.cuda()











# def eval_once(model,train_dataloader,test_dataloader):
def eval_once(model,train_dataloader,test_dataloader):
    model.eval()
    scores = []
    labels = []
    good_scores = []


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
    update_constant_in_file(constantPath ,'THRESHOLD',threshold)

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
    # print(pred_labels[:20])
    # print(labels[:20])



    acc = accuracy_score(labels,pred_labels )

    
    # print('accuracy_score : ',acc)
    return acc

my_variable='he re we are => my_variable'
def eval_one_image(model, threshold, path):
    global my_variable

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
        my_variable = 'bad'

    else:
        print('good')
        my_variable = 'good'

def find_theshold(model):
    model.eval()
    good_scores = []

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



print("Path to constants.py:", constantPath)

def update_constant_in_file(file_path, constant_name, new_value):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the desired constant
    for i, line in enumerate(lines):
        if line.startswith(constant_name):
            lines[i] = f"{constant_name} = {repr(new_value)}\n"
            break

    # Write back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

if(args.eval_All):
    acc = eval_once(model,train_dataloader,test_dataloader)
    print('=>accuracy_score : ',acc)
    update_constant_in_file(constantPath,'SCORE_ALL',acc)
    

if(args.eval_One):
    print('here in condition evel_One')
    threshold = find_theshold(model)
    image_test= args.image_path
    print('the image test loaded is : ',image_test)
    print('threshold : ',threshold)
    eval_one_image(model, threshold, image_test)