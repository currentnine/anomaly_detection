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

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument("--eval_All", action="store_true", help="run eval only")
    parser.add_argument("--eval_One", action="store_true", help="run eval only")
    parser.add_argument("--image_path", type=str, help="path to the image file")
    args = parser.parse_args()
    return args

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


weight = const.WEIGHT_PATH

data_dir = const.DATASET_PATH
constantPath = const.__file__


# def eval_once(model,train_dataloader,test_dataloader):
from tqdm import tqdm
def eval_once(model,train_dataloader,test_dataloader):
    model.eval()
    scores = []
    labels = []
    good_scores = []


    for batch_images in tqdm(train_dataloader, desc="Processing training data"):  # images.shape (8배치사이즈)   
        
    
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
    print('accuracy_score : ',acc)
    return acc


def eval_one_image(model, threshold, path):
    global my_variable
    img  = cv2.imread(str(path))
    if img is None:
        print(f"Failed to load image from {path}")
        return  # or handle the error as needed
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
        result = 'bad'
        update_value(constantPath,'PREDECTION_RESUILT',result)

    else:
        print('good')
        result = 'good'
        update_value(constantPath,'PREDECTION_RESUILT',result)

def find_theshold(model,train_dataloader):
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

def update_value(file_path, constant_name, new_value):
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
def load_data(data_dir, is_train):
    dataset = CustomDataset(data_dir, CFG["input_size"], is_train)
    dataloader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True)
    return dataloader   

def main():
    model = build_model()
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.cuda()
    args = parse_args()

    if args.eval_All:
        train_dataloader = load_data(const.DATASET_PATH, True)
        test_dataloader = load_data(const.DATASET_PATH, False)
        acc = eval_once(model,train_dataloader,test_dataloader)
        print('=>accuracy_score : ',acc)
        update_constant_in_file(constantPath,'SCORE_ALL',acc)

    if args.eval_One:
        print('here in condition evel_One')
        #get it from constans.py 
        threshold = const.THRESHOLD
        image_test= args.image_path
        print('the image test loaded is : ',image_test)
        print('threshold : ',threshold)
        eval_one_image(model, threshold, image_test)

if __name__ == "__main__":
    main()