import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
from PIL import Image
from PIL import ExifTags
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

class dataAdaptor:
    def __init__(self, dir_path, df):
        self.dir_path = Path(dir_path)
        self.df = df
        self.img_names = self.df.name.unique().tolist()
        
    def __len__(self) -> int:
        return len(self.img_names)
        
    def get_image_and_labels_by_idx(self, index):
        image_name = self.img_names[index]
        image = Image.open(self.dir_path / image_name)
        label = self.df.loc[self.df.name==image_name, 'class_lb'].values
        return image, label, index

class ImgDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (image, class_label, image_id) = self.ds.get_image_and_labels_by_idx(index)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if 274 in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        sample = {"image":np.array(image, dtype=np.float32), "labels":torch.Tensor(class_label).long()}
        sample = self.transforms(**sample)
        return sample

def get_train_transforms(target_img_size=512):
    return A.Compose([A.HorizontalFlip(p=0.5), A.Resize(height=target_img_size, width=target_img_size, p=1), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2(p=1)], p=1.0)

def get_valid_transforms(target_img_size=512):
    return A.Compose([A.Resize(height=target_img_size, width=target_img_size, p=1), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2(p=1)], p=1.0)

def prepare_data(df_path, te_sz, dir_path, batch_sz):
    df = pd.read_csv(df_path)
    tr_df, te_df = train_test_split(df, test_size=te_sz, random_state=42)
    train_ds, test_ds = dataAdaptor(dir_path, tr_df), dataAdaptor(dir_path, te_df)
    train = ImgDataset(dataset_adaptor=train_ds, transforms=get_train_transforms())
    test = ImgDataset(dataset_adaptor=test_ds, transforms=get_valid_transforms())
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_sz, shuffle=True, num_workers=0, worker_init_fn=random.seed(1))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sz, shuffle=False, num_workers=0, worker_init_fn=random.seed(1))
    return train_loader, test_loader

def train_nn(mdl, train_loader, test_loader, criterion, optimizer, device, it=100):
    tr_loss, te_loss, acc, pre, rec = [], [], [], [], []
    for epoch in range(it):
        y_true, y_pred = [], []
        mdl.train()
        for i, (tr_d, tr_lb) in enumerate(train_loader):
            tr_d, tr_lb = tr_d.to(device), tr_lb.to(device)
            out = mdl(tr_d)
            loss1 = criterion(out, tr_lb)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
        mdl.eval()
        with torch.no_grad():
            for i, (te_d, te_lb) in enumerate(test_loader):
                te_d, te_lb = te_d.to(device), te_lb.to(device)
                out = mdl(te_d)
                loss2 = criterion(out, te_lb)
                y_pred.append(torch.argmax(out, dim=1))
                y_true.append(te_lb.cpu().numpy())
        
        acc.append(metrics.accuracy_score(y_true, y_pred))
        pre.append(metrics.precision_score(y_true, y_pred))
        rec.append(metrics.recall_score(y_true, y_pred))
        tr_loss.append(loss1.item())
        te_loss.append(loss2.item())
    return acc, pre, rec, tr_loss, te_loss

def eval_nn(test_loader, model, device):
    pred_y, true_y = [], []
    model.eval()
    with torch.no_grad():
        for i, (te_d, te_lb) in enumerate(test_loader):
            te_d, te_lb = te_d.to(device), te_lb.to(device)
            out = model(te_d)
            pred_y.append(torch.argmax(out, dim=1))
            true_y.append(te_lb.cpu().numpy())
    
    acc = metrics.accuracy_score(true_y, pred_y)
    pre = metrics.precision_score(true_y, pred_y)
    rec = metrics.recall_score(true_y, pred_y)
    return acc, pre, rec

df_path = 'FOCUS_box_coord.csv'
dir_path = 'FOCUS_crop'
te_sz = 0.3
batch_sz = 8
train_loader, test_loader = prepare_data(df_path, te_sz, dir_path, batch_sz)
mdl = resnet50(weights=ResNet50_Weights.DEFAULT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mdl.to(device)

a0, p0, r0 = eval_nn(test_loader, mdl, device)
print('No train:', a0, p0, r0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mdl.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
acc, pre, rec, tr_l, te_l = train_nn(mdl, train_loader, test_loader, criterion, optimizer, device, it=100)
print('Accuracy:', acc[-1], max(acc))
print('Precision:', pre[-1])
print('Recall:', rec[-1])
