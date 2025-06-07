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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # if is_train:

        #     self.image_files = glob(
        #         os.path.join(root, "train", "good", "*.png")
        #     )
        # else:# self.image_files = glob(os.path.join(root, "test", "*", "*.png"))        
        #     self.image_files = glob(os.path.join(root,  "good", "*.png")) + glob(os.path.join(root, "bad", "*.png"))
        #     print(self.image_files) 
        # self.is_train = is_train

        if is_train:
            self.image_files = glob(os.path.join(root, "train", "good", "*.*"))
        else:
            self.image_files = glob(os.path.join(root, "good", "*.*")) + glob(os.path.join(root, "bad", "*.*"))
        
        # Filter the files to keep only valid image formats
        self.image_files = [file for file in self.image_files if self.is_valid_image(file)]
        
        print(self.image_files)
        self.is_train = is_train


    def is_valid_image(self, file_path):
        try:
            img = Image.open(file_path)
            img.close()
            return True
        except Exception as e:
            return False

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
    
class CustomDataset2(Dataset):
    def __init__(self, root, input_size, is_train=True):

        self.img_size = input_size
        self.image_files = glob(os.path.join(root, "*.png")) 
        self.is_train = is_train
        self.transform = transforms.ToTensor()
        
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
        transform = transforms.ToTensor()
        img = transform(img)
        return img, label
    def __len__(self):
        return len(self.image_files)
def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument("--eval_All", action="store_true", help="run eval only")
    parser.add_argument("--eval_One", action="store_true", help="run eval only")
    parser.add_argument("--image_path", type=str, help="path to the image file")
    parser.add_argument("--threshold", type=str, help="path to the image file")
    parser.add_argument("--train",action="store_true", help="run eval only")
    parser.add_argument("--TH",action="store_true", help="run eval only")
    args = parser.parse_args()
    return args

# def build_model(): 
#     model = fastflow.FastFlow(
#         backbone_name=CFG["backbone_name"],
#         flow_steps=CFG["flow_step"],
#         input_size = int(CFG["input_size"]),
#         conv3x3_only=CFG["conv3x3_only"],
#         hidden_ratio=CFG["hidden_ratio"],
#     )
#     print(
#         "Model A.D. Param#: {}".format(
#             sum(p.numel() for p in model.parameters() if p.requires_grad)
#         )
#     )
#     return model


def build_model(): 
    print(const.OUTPUT_FILE_PATH)
    model = fastflow.FastFlow(
        backbone_name=CFG["backbone_name"],
        flow_steps=CFG["flow_step"],
        input_size = CFG['input_size'],
        conv3x3_only=CFG["conv3x3_only"],
        hidden_ratio=CFG["hidden_ratio"],
    )
    logTraing ="Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    print(logTraing)

    with open(const.OUTPUT_FILE_PATH, 'a') as file:
                file.write(logTraing+'\n')
    return model





# def eval_once(model,train_dataloader,test_dataloader):
from tqdm import tqdm
def eval_once(model,test_dataloader,threshold):
    model.eval()
    scores = []
    labels = []
    good_scores = []


    # for batch_images, batch_labels in test_dataloader:
    #     batch_images = batch_images.permute(0,3,1,2)
    #     batch_images = batch_images.float()
    #     batch_images = batch_images.cuda()

    #     with torch.no_grad():
    #         ret = model(batch_images)  #divide_num 갯수만큼의 이미지 모델에 넣고 output출력

    #     # outputs = ret["anomaly_map"].detach()
    #     preds = ret['preds']

    #     score = torch.mean(torch.stack(preds), 0).cpu().numpy()
    #     good_scores.extend(score.tolist())
    

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

from scipy.special import expit
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
    # anomaly_map =  ret['anomaly_map']


    score = torch.mean(torch.stack(pred), 0).cpu().numpy()

    adjusted_scores = score - threshold

    # similarities = expit(100 / 33455531.475 * adjusted_scores)

    similarities =sigmoid(score , threshold,33455531.475)
    print(similarities)
    print(expit(100 / 33455531.475 * adjusted_scores))
    # similarities = sigmoid(adjusted_scores)

    print(similarities[0])
    print(' score is ======> ',score)
    
    

    if score > threshold:
        print('bad')
        result = similarities
        update_value(constantPath,'PREDECTION_RESUILT','bad')
        return result[0].item()

    else:
        print('good')
        result = similarities
        update_value(constantPath,'PREDECTION_RESUILT','good')
        return result[0].item()
    
# def eval_all_image(model, dataloader, threshold):
#     total_correct = 0
#     total_samples = 0
#     for batch in dataloader:
#         images, labels = batch
#         images = images.cuda()
        
#         with torch.no_grad():
#             ret = model(images)
        
#         preds = ret['preds']
#         scores = torch.mean(torch.stack(preds), 0).cpu().numpy()
#         adjusted_scores = scores - threshold
#         similarities = sigmoid(scores, threshold, 33455531.475)
        
#         for similarity, label in zip(similarities, labels):
#             if similarity > threshold:
#                 if label == 'bad':
#                     total_correct += 1
#             else:
#                 if label == 'good':
#                     total_correct += 1
#             total_samples += 1
    
#     accuracy = total_correct / total_samples
#     return accuracy

    
def sigmoid1(x):
    return 1 / (1 + np.exp(-x))   
def sigmoid(x, x0, k):
    y = 1 / (1+ np.exp(-k*(x-x0)))
    return y     

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
    if(is_train):
        dataset = CustomDataset(data_dir, CFG["input_size"], is_train = True)
        dataloader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True)
        return dataloader   
    else :
        dataset = CustomDataset(data_dir, CFG["input_size"], is_train =False)
        dataloader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True)
        return dataloader 
    
def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )

def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    
    for step, batch_images in enumerate(dataloader):

        images = []
        

        batch_images = batch_images.permute(0,3,2,1)
        batch_images = batch_images.float()
        data = batch_images
        data = data.to(device)

        ret = model(data)

        loss = ret["loss"]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        
    # loss_meter.update(loss.item())
    # if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
    #     logTraing ="Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
    #             epoch + 1, step + 1, loss_meter.val, loss_meter.avg
    #         )
    #     print(logTraing)
    #     with open(file_path, 'a') as file:
    #         file.write(logTraing+'\n')
    loss_meter.update(loss.item())
    # if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
    logTraing ="Epoch {} - Step {}: loss = {:.2f}({:.2f})".format(
            epoch + 1, step + 1, loss_meter.val, loss_meter.avg
        )
    print(logTraing)
    
    with open(const.OUTPUT_FILE_PATH, 'a') as file:
        file.write(logTraing+'\n')
    

def train(model,train_dataloader):

    # model = build_model()
    optimizer = build_optimizer(model)


    model.to(device)
    

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        # if (epoch + 1) % const.EVAL_INTERVAL == 0:
        #     # eval_once(test_dataloader, model)
            
        #     j = 0

        # if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(const.CHECKPOINT_PATH, "%d.pt"%epoch),
        )

constantPath = const.__file__
weight = const.WEIGHT_PATH
import importlib
def reload_constants():
    importlib.reload(const)
def main():
    

    # data_dir = const.DATASET_PATH
   
    
    model = build_model()
    

    model.cuda()
    args = parse_args()

    if args.TH:
        reload_constants()
        if not os.path.exists(weight):
            print(f"Model file not found at {weight}")
        else:
            print(f"Model file found. Proceeding to load.")
        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint["model_state_dict"])
        train_dataloader = load_data(const.DATASET_PATH, True)
        theshold_value=find_theshold(model,train_dataloader)
        print('the new trushold is ',theshold_value)
        update_constant_in_file(constantPath ,'THRESHOLD',theshold_value)

    if args.eval_All:
        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint["model_state_dict"])
        # train_dataloader = load_data(const.DATASET_PATH, True)
        reload_constants()
        print('const.MULTIPLE_TEST_PATH ---> ',const.MULTIPLE_TEST_PATH)
        test_dataloader = load_data(const.MULTIPLE_TEST_PATH, False)
        # acc = eval_once(model,train_dataloader,test_dataloader)
        
        if args.threshold is not None:
            threshold  = float(args.threshold)
        else :
            threshold=const.THRESHOLD
        acc = eval_once(model,test_dataloader,threshold)
        print('=>accuracy_score : ',acc)
        update_constant_in_file(constantPath,'SCORE_ALL',acc)

    if args.eval_One:       
        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('here in condition evel_One')
        #get it from constans.py 
        # threshold = float(args.threshold)
        if args.threshold is not None:
            threshold  = float(args.threshold)
        else :
            threshold=const.THRESHOLD
        image_test= args.image_path
        print('the image test loaded is : ',image_test)
        print('threshold : ',threshold)
        acc= eval_one_image(model, threshold, image_test)
        update_constant_in_file(constantPath,'SCORE_TEST',acc)

    if args.train:
        print('const.DATASET_PATH :',const.DATASET_PATH)
        train_dataloader = load_data(const.DATASET_PATH, True)
        train(model,train_dataloader)

if __name__ == "__main__":
    main()