import os
from glob import glob
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import random
import imutils
import sys



class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, divide_num, is_train=True):
        

        self.divide_num = divide_num
        
        self.img_size = input_size
        self.image_transform = transforms.Compose(
            [
                
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
               

            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
        
        self.is_train = is_train





    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        image = cv2.imread(image_file)
        image = cv2.resize(image, (self.img_size,self.img_size))


        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, gray_img = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
        _, gray_img = cv2.threshold(gray_img,245, 255, cv2.THRESH_TRUNC)



        contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        cnt = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(image)


        mask = cv2.drawContours(mask, [cnt], -1, (255,255,255), thickness=cv2.FILLED)

        image = cv2.bitwise_and(image, mask)


        cv2.imwrite("./train_image_confirm/{}.jpg".format(index), image)


        flip_random = random.randint(0,3)

        if flip_random == 0:
            image = cv2.flip(image, 0)

        elif flip_random == 1:
            image = cv2.flip(image, 1)
        
        elif flip_random == 2:
            image = cv2.flip(image, -1)
      
            

        path = image_file.split('\\')
        path = [path[3], path[4].split('.')[0]]
      



        origin_img = image/255.

        img = image/255.




        if self.is_train:
            return img
        else:
            return img, origin_img, path
        


    def __len__(self):
        return len(self.image_files)

