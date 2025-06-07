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
from glob import glob


from torch.utils.data import Dataset, DataLoader


j = 0

data_dir = 'C:/Users/wjdgu/OneDrive/Desktop/project/original_data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
   
    
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
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
    'batch_size' : 16

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
        image = cv2.resize(image, (CFG['input_size'],CFG['input_size']))
    
            

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




train_loader = CustomDataset(data_dir, CFG["input_size"],  is_train =True)
test_loader = CustomDataset(data_dir, CFG["input_size"],is_train =False)


train_dataloader = DataLoader(train_loader, batch_size = CFG['batch_size'], shuffle = True)
test_dataloader = DataLoader(test_loader, batch_size = CFG['batch_size'], shuffle = True)




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
    



def eval_once(dataloader, model):
    model.eval()
    global j
    

    for batch_images, label ,path in dataloader:   # images.shape (8배치사이즈)   
        
        labels = []   #정답값
        predicts = []   #예측값 good or bad
        
        
            
         #list에 담겨져 있는 torch tensor에 옮겨담기
        batch_images = batch_images.permute(0,3,2,1)
        batch_images = batch_images.float()
        batch_images = batch_images.to(device)

        with torch.no_grad():
            ret = model(batch_images)  

        
        outputs = ret["anomaly_map"].detach()
        preds = ret['preds']



        mean = torch.mean(torch.stack(preds), 0)
    

        for k in range(outputs.shape[0]):

            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            ax.imshow(batch_images[k].permute(2,1,0).cpu().numpy())

            
            ax = fig.add_subplot(1,2,2)
            img = outputs[k].permute(2,1,0).cpu().numpy()
            
            score = mean[k]

            
            # if score <-19000000:
            #     predicts.append('good')

            # else:
            #     predicts.append('bad')

            

def train(args):
    # global file_path
    # #creat new file for eash training
    # os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    # checkpoint_dir = os.path.join(
    #     const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    # )
    # # os.makedirs(checkpoint_dir, exist_ok=True)

    # print('-------------')
    # print(checkpoint_dir)
    # print('-------------')

    
    # os.makedirs(checkpoint_dir, exist_ok=True)

    # printprint('---------------- creat the file for ---------------- ',checkpoint_dir) 
    # file_path = os.path.join(checkpoint_dir, 'output.txt')
    
    # update_output_file_path_in_constants(file_path)
    # print(const.OUTPUT_FILE_PATH)
    #  # Create the file
    # with open(file_path, 'w') as file:
    #     file.write("")
    # print(file_path)
    # print('----------------------------------------------------')
 
    model = build_model()
    optimizer = build_optimizer(model)


    model.to(device)
    

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        # if (epoch + 1) % const.EVAL_INTERVAL == 0:
        #     # eval_once(test_dataloader, model)
            
        #     j = 0

        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(const.CHECKPOINT_PATH, "%d.pt" % epoch),
            )

        

def evaluate(args):

    model = build_model()
    optimizer = build_optimizer(model)
    model.to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(checkpoint)
    # print(model)
    

    model.to(device)
    eval_once(test_dataloader, model)

# def update_output_file_path_in_constants(new_path, constants_file='constants.py'):
#     try:
#         new_path= os.path.abspath(new_path)
#         print('we update ',new_path)
#         # Read the current contents of the file
#         with open(constants_file, 'r') as file:
#             lines = file.readlines()

#         # Modify the specific constant line
#         updated = False
#         for i, line in enumerate(lines):
#             if line.startswith('OUTPUT_FILE_PATH'):
#                 lines[i] = f'OUTPUT_FILE_PATH = "{new_path}"\n'
#                 updated = True
#                 break

#         if not updated:
#             # If the constant was not found, add it to the file
#             lines.append(f'OUTPUT_FILE_PATH = "{new_path}"\n')

#         # Write the updated lines back to the file
#         with open(constants_file, 'w') as file:
#             file.writelines(lines)

#     except Exception as e:
#         print(f"An error occurred while updating constants file: {e}")


if __name__ == "__main__":
    
    train(args)
