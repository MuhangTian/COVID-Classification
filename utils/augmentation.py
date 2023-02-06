'''
For generating augmented images from original images for training
'''
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import pandas as pd
import numpy as np
import cv2
import os
from image_helper import DataAdapter, show_image
from tqdm import tqdm
from PIL import Image

def augment(func: A.Compose, image: np.array, bbox: list, label: int):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # this is IMPORTANT
    transformed = func(image=img, bboxes=bbox, class_labels=label)
    
    return transformed['image'], transformed['bboxes'], transformed['class_labels']


def augment_save(func: A.Compose, da: DataAdapter, path: str, start: int, num: int, save_name: str):
    assert type(num) == int
    map = {'label': [],
           'filename': [],
           'xmin': [],
           'ymin': [],
           'xmax': [],
           'ymax': []}
    if num == 0:
        iterator = iter(range(start, start+len(da)))
        for id in tqdm(range(len(da)), desc='Augmenting...'):
            id, img, bbox, label = da.i_get(id)
            img, bbox, label = augment(func, img, bbox, label)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # NOTE: important, as colo is flipped during saving
            id = next(iterator)
            cv2.imwrite(f"{path}/{id}.jpg", img)
            map['label'].append(label[0])
            map['filename'].append(f"{id}.jpg")
            bbox = bbox[0]
            map['xmin'].append(bbox[0])
            map['ymin'].append(bbox[1])
            map['xmax'].append(bbox[2])
            map['ymax'].append(bbox[3])
    else:
        iterator = iter(range(start, start+num))
        np.random.seed(1)       # TODO: remove this
        for _ in tqdm(range(num), desc='Augmenting...'):
            id = np.random.randint(0, len(da))
            id, img, bbox, label = da.i_get(id)
            img, bbox, label = augment(func, img, bbox, label)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # NOTE: important, as color is flipped during saving
            cv2.imwrite(f"{path}/{next(iterator)}.jpg", img)

    df = pd.DataFrame.from_dict(map)
    df.set_index('label')
    df.to_csv(f'{save_name}.csv')
    
    return print('COMPLETE')

def random_check(func: A.Compose, da: DataAdapter, num: int=100):
    for _ in range(num):
        idx = np.random.randint(0, len(da))
        id, image, bbox, label = da.i_get(idx)
        image, bbox, label = augment(func, image, bbox, label)
        show_image(image, bbox, label)


def run_all():
    da = DataAdapter('data/self-data-TRAIN.csv', data_path='data/self-data/images')
    transform1 = A.Compose(
    [
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, 
                               p=1, brightness_by_max=False),
    A.RandomGamma((50, 300), p=1)
    ],
    p=1.0,
    bbox_params=A.BboxParams(format="pascal_voc", 
                                label_fields=["class_labels"]),
    )
    
    transform2 = A.Compose(
    [
    A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40, p=1),
    ],
    p=1.0,
    bbox_params=A.BboxParams(format="pascal_voc", 
                                label_fields=["class_labels"]),
    )
    
    transform3 = A.Compose(
    [
    A.AdvancedBlur(sigmaX_limit=(15,18), sigmaY_limit=(15,18), p=1),
    A.GaussNoise(var_limit=(30,100), mean=10, p=1)
    ],
    p=1.0,
    bbox_params=A.BboxParams(format="pascal_voc", 
                                label_fields=["class_labels"]),
    )
    
    transform4 = A.Compose(
    [
    A.ChannelDropout(p=1)
    ],
    p=1.0,
    bbox_params=A.BboxParams(format="pascal_voc", 
                                label_fields=["class_labels"]),
    )
    
    transform5 = A.Compose(
    [
    A.Downscale(p=1),
    ],
    p=1.0,
    bbox_params=A.BboxParams(format="pascal_voc", 
                                label_fields=["class_labels"]),
    )
    
    transform6 = A.Compose(
    [
    A.RandomSunFlare(p=1),
    ],
    p=1.0,
    bbox_params=A.BboxParams(format="pascal_voc", 
                                label_fields=["class_labels"]),
    )
    
    transform7 = A.Compose(
    [
    A.Defocus(p=1),
    ],
    p=1.0,
    bbox_params=A.BboxParams(format="pascal_voc", 
                                label_fields=["class_labels"]),
    )
    
    # random_check(transform7, da)
    augment_save(transform1, da, 'data/contrast-gamma-aug', 1731, 0, save_name='contrast-gamma')
    augment_save(transform2, da, 'data/hue-satur-aug', 3033, 0, save_name='hue-satur')
    augment_save(transform3, da, 'data/blur-noise-aug', 4335, 0, save_name='blur-noise')
    augment_save(transform4, da, 'data/dropout-aug', 5637, 0, save_name='dropout')
    augment_save(transform5, da, 'data/downscale-aug', 6939, 0, save_name='downscale')
    augment_save(transform6, da, 'data/sun-aug', 8241, 0, save_name='sun')
    augment_save(transform7, da, 'data/defocus-aug', 9543, 0, save_name='defocus')
    
    return print(f"{'='*8} FINISH {'='*8}")


def merge_csv(csvs: list):
    a = pd.read_csv('data/self-data-TRAIN.csv')     # want to merge with this one
    a = a.drop(['width', 'height', 'depth'], axis=1)
    df_arr = [a]
    for csv in tqdm(csvs, desc='Merging...'):
        df = pd.read_csv(csv)
        df = df[['label', 'filename', 'xmin', 'ymin', 'xmax', 'ymax']]
        df_arr.append(df)
    all = pd.concat(df_arr)
    all = all[['label', 'filename', 'xmin', 'ymin', 'xmax', 'ymax']]
    all.to_csv('augmented-TRAIN.csv')
    
    return print(f"{'='*8} FINISH {'='*8}")

if __name__ == '__main__':
    
    # da = DataAdapter('augmented.csv', data_path='data/contrast-aug')
    # for _ in range(100):
    #     idx = np.random.randint(0, len(da))
    #     id, image, bbox, label = da.i_get(idx)
    #     show_image(image, bbox, label)
    merge_csv(['blur-noise.csv', 'contrast-gamma.csv', 'defocus.csv', 'downscale.csv', 'dropout.csv', 'hue-satur.csv', 'sun.csv'])