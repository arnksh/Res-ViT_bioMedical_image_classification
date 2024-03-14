import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
# import math
# import scipy.io as sio
import os
           
img_size=128
batch_size=16
transform = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Resize(size=(img_size, img_size), antialias=True),
          transforms.Normalize((1,1,1),(1, 1,1))
          ])
          
def loaddata(root, targetName):
    root = root + targetName + '/'
    classNames = [f for f in os.listdir(root + 'train/') if not f.endswith('.ini')]
    numClass = len(classNames)
    label_map = {i : className for i, className in enumerate(classNames)}
    
    
    Train = pd.DataFrame(columns = ['img', 'label'])
    j=0
    for i in label_map:
      img_path = os.path.join(root, 'train', label_map[i])
      for img in os.listdir(img_path):
          if not img.endswith('.ini'):
              image = os.path.join(img_path, img)
              Train  = pd.concat((Train, pd.DataFrame({'img':[image], 'label': i})), ignore_index = True)
              j+=1
    
    
    Test = pd.DataFrame(columns = ['img', 'label'])
    j=0
    for i in label_map:
      img_path = os.path.join(root, 'test', label_map[i])
      for img in os.listdir(img_path):
          if not img.endswith('.ini'):
              image = os.path.join(img_path, img)
              Test  = pd.concat((Test, pd.DataFrame({'img':[image], 'label': i})), ignore_index = True)
              j+=1
    
    
    
    Val = pd.DataFrame(columns = ['img', 'label'])
    j=0
    for i in label_map:
      img_path = os.path.join(root, 'val', label_map[i])
      for img in os.listdir(img_path):
          if not img.endswith('.ini'):
              image = os.path.join(img_path, img)
              Val  = pd.concat((Val, pd.DataFrame({'img':[image], 'label': i})), ignore_index = True)
              j+=1
    
    
    TrainLoader = DataLoader(loadData(Train), batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(loadData(Test), batch_size=batch_size, shuffle=False)
    ValLoader = DataLoader(loadData(Val), batch_size=batch_size, shuffle=False)
    return TrainLoader, TestLoader, ValLoader, numClass

def loadNoisydata(root, targetName):
    root = root + targetName + '/'
    classNames = [f for f in os.listdir(root + 'train_noise/') if not f.endswith('.ini')]
    numClass = len(classNames)
    label_map = {i : className for i, className in enumerate(classNames)}
    
    
    Train = pd.DataFrame(columns = ['img', 'label'])
    j=0
    for i in label_map:
      img_path = os.path.join(root, 'train_noise', label_map[i])
      for img in os.listdir(img_path):
          if not img.endswith('.ini'):
              image = os.path.join(img_path, img)
              Train  = pd.concat((Train, pd.DataFrame({'img':[image], 'label': i})), ignore_index = True)
              j+=1
    
    
    Test = pd.DataFrame(columns = ['img', 'label'])
    j=0
    for i in label_map:
      img_path = os.path.join(root, 'test_noise', label_map[i])
      for img in os.listdir(img_path):
          if not img.endswith('.ini'):
              image = os.path.join(img_path, img)
              Test  = pd.concat((Test, pd.DataFrame({'img':[image], 'label': i})), ignore_index = True)
              j+=1
    
    
    
    Val = pd.DataFrame(columns = ['img', 'label'])
    j=0
    for i in label_map:
      img_path = os.path.join(root, 'val_noise', label_map[i])
      for img in os.listdir(img_path):
          if not img.endswith('.ini'):
              image = os.path.join(img_path, img)
              Val  = pd.concat((Val, pd.DataFrame({'img':[image], 'label': i})), ignore_index = True)
              j+=1
    
    
    TrainLoader = DataLoader(loadData(Train), batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(loadData(Test), batch_size=batch_size, shuffle=False)
    ValLoader = DataLoader(loadData(Val), batch_size=batch_size, shuffle=False)
    return TrainLoader, TestLoader, ValLoader, numClass


class loadData(Dataset):
    def __init__(self, df):
      self.imgPath = df

    def __len__(self):
        return self.imgPath.shape[0]

    def __getitem__(self, idx):
        image_id = self.imgPath.iloc[idx,0]
        image = Image.open(image_id).convert("RGB")
        label = self.imgPath.iloc[idx,1]
        image = transform(image)
        return image, label