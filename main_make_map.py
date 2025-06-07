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

data_dir = "E:/fastflow_dataset"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
   
    parser.add_argument(
        "--img_size", type=int, default=256, help="image size"
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )


    args = parser.parse_args()
    return args



args = parse_args()



CFG = {
    'input_size': args.img_size,
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
                os.path.join(root, "train", "good", "*")
            )

        else:
            self.image_files = glob(os.path.join(root, "test", "*", "*"))
        
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
    model = fastflow.FastFlow(
        backbone_name=CFG["backbone_name"],
        flow_steps=CFG["flow_step"],
        input_size = CFG['input_size'],
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

        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )





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

       
        

    threshold  = np.percentile(good_scores, 99)

    return threshold

def calculate_accuracy(y_true, y_pred):
    # 올바르게 예측한 레이블의 개수를 세기 위해 True/False 값을 저장하는 리스트 생성
    correct_predictions = [true == pred for true, pred in zip(y_true, y_pred)]
    
    # 정확도 계산: 올바르게 예측한 레이블의 개수를 전체 레이블의 개수로 나눔
    accuracy = sum(correct_predictions) / len(y_true)
    
    return accuracy




from sklearn.metrics import roc_curve, accuracy_score
def eval_once(dataloader, model):
    model.eval()
    global j
    
    labels = []   #정답값
    predicts = []   #예측값 good or bad
    scores = []

    for batch_images, label ,path in dataloader:   # images.shape (8배치사이즈)   
        
        
        
            
         #list에 담겨져 있는 torch tensor에 옮겨담기
        batch_images = batch_images.permute(0,3,2,1)
        batch_images = batch_images.float()
        batch_images = batch_images.to(device)

        with torch.no_grad():
            ret = model(batch_images)  

        
        # outputs = ret["anomaly_map"].detach()
    
        preds = ret['preds']
        anomaly_maps = ret['anomaly_map']

        mean = torch.mean(torch.stack(preds), 0)
        
        for k in range(mean.shape[0]):
        
            score = mean[k]

            scores.append(score.item())

            labels.append(label[k].item())
            save_path = os.path.join('results/anomaly_map', path[1][k])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            anomaly_map = anomaly_maps[k].permute(1, 2, 0).cpu().numpy()
            anomaly_map = cv2.normalize(anomaly_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            anomaly_map = np.uint8(anomaly_map)

            # 색상 맵 적용
            anomaly_map_colored = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)

            # 원본 이미지를 회색조로 변환합니다.
            original_image = batch_images[k].permute(1, 2, 0).cpu().numpy() * 255
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(cv2.cvtColor(np.uint8(original_image), cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original Image')
            axs[1].imshow(anomaly_map_colored)
            axs[1].set_title('Anomaly Map')
            # anomaly map과 원본 이미지를 수평으로 결합합니다.
            filename = f'{path[-1][k]}_combined.png'
            file_path = os.path.join(save_path, filename)
            plt.savefig(file_path)
            plt.close(fig)


        # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = roc_auc_score(labels,scores)

    # 최적의 임계값 찾기 (예: Youden's J statistic을 사용하는 경우)
    j_scores = tpr - fpr
    max_j_index = np.argmax(j_scores)  # Youden's J statistic을 최대화하는 임계값의 인덱스
    optimal_threshold = thresholds[max_j_index]

    # threshold = find_theshold(model)

    # 이 임계값을 사용하여 예측 레이블을 결정
    y_pred = [1 if score >= optimal_threshold else 0 for score in scores]


    # print(labels)
    # print('-----------------------------------')
    # print(y_pred)
    # 정확도 계산
    accuracy = accuracy_score(labels, y_pred)

    print(f'Optimal threshold: {optimal_threshold}')
    print(f'Accuracy: {accuracy}')
    print(f'AUC: {auc_score}')

        


        

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



if __name__ == "__main__":
    
    evaluate(args)
