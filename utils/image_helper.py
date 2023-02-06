''' Helpers for image data '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
import cv2

def get_path(name): return 'data/self-data/images/{}'.format(name)
def get_img(name): return Image.open(get_path(name))
def get_idx(df, name): return df.index[df['filename'] == name].tolist()[0]
def get_name(df, idx): return df.iloc[idx]['filename']

def get_bbox(df, name):
    ''' get coordinates of bounding box 
    Returns: xmin, ymin, xmax, ymax'''
    idx = get_idx(df, name)
    return [df.loc[idx]['xmin'], df.loc[idx]['ymin'], df.loc[idx]['xmax'], df.loc[idx]['ymax']]


def get_edges(bbox):
    xmin, ymin, xmax, ymax = bbox
    bottom_left = (xmin, ymax)
    width = xmax - xmin
    height = ymin - ymax
    return bottom_left, width, height


def plot_bbox(ax, bbox, color):
    bottom_left, width, height = get_edges(bbox)
    rect = patches.Rectangle(bottom_left, width, height, 
                               linewidth=1, edgecolor=color, color=color, fill=True, alpha=0.2)
    ax.add_patch(rect)


def plot_image(df, img_name, figsize=(8,8)):
    """
    plot image of a file along with its bounding box in dataset
    Args:
        df (pd.DataFrame): dataframe of the dataset
        img_name (str): name of the image to be plotted
        figsize (tuple, optional): size of the visualization. Defaults to (8,8).
    """
    bbox = get_bbox(df, img_name)
    idx = df.index[df['filename'] == img_name].tolist()[0]
    color = 'red' if df.loc[idx]['label'] == 1 else 'tab:blue'
    img = get_img(img_name)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    plot_bbox(ax, bbox, color)
    plt.title(img_name)
    plt.axis('off')
    plt.show()


def show_image(img, bbox, label, img_name=None, figsize=(8,8)):
    ''' Plot image along with bbox without dataframe '''
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    color = 'red' if label[0] == 1 else 'tab:blue'
    plot_bbox(ax, bbox[0], color)
    
    # print(img)
    plt.title(img_name)
    plt.axis('off')
    plt.show()


def show_transform_image(func, DA, val):
    """
    Plot transformed image
    """
    if type(val) == str:
       idx, img, bbox, label = DA.n_get(val)
       name = DA.get_name(idx)
    elif type(val) == int:
        name, img, bbox, label = DA.i_get(val)
    else: raise ValueError('Only filename or index allowed')
    img = np.array(img, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # this is IMPORTANT
    transformed = func(image=img, bboxes=bbox, class_labels=label)
    
    return show_image(transformed['image'], transformed['bboxes'], transformed['class_labels'], name)

class DataAdapter:
    
    def __init__(self, df) -> None:
        self.df = df
    
    def __len__(self): return self.df.shape[0]
    
    
    def get(self, val):
        ''' return PIL image '''
        if type(val) == str: return get_img(val)
        elif type(val) == int: return get_img(get_name(self.df, val))
        else: raise ValueError('Only image name or index please')
    
    
    def get_name(self, idx): return self.df.iloc[idx]['filename']
    
    
    def i_get(self, idx: int):
        ''' Given index of the dataframe
        Returns: name, image, bbox, label '''
        name = get_name(self.df, idx)
        id = int(name.replace('.jpg', ''))
        img = get_img(name)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # this is IMPORTANT
        bbox = get_bbox(self.df, name)
        label = self.df.iloc[idx]['label']
        
        return id, img, [bbox], [label]
    
    
    def n_get(self, name: str):
        ''' Given image name
        Returns: dataframe index, image, bbox, label '''
        idx = get_idx(self.df, name)
        img = get_img(name)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # this is IMPORTANT
        bbox = get_bbox(self.df, name)
        label = self.df.iloc[idx]['label']
        
        return idx, img, [bbox], [label]
    
    
    def plot_img(self, val: int):
        ''' 
        Given dataframe index or image name, plot image along with bounding box colored by label
        '''
        if type(val) == int: return plot_image(self.df, get_name(self.df, val))
        elif type(val) == str: return plot_image(self.df, val)
        else: raise ValueError('Only index or image name allowed')
        
if __name__ == '__main__':
    df = pd.read_csv('data/augmented-TRAIN.csv')
    da = DataAdapter(df)
    for _ in range(10000):
        id = np.random.randint(0, 10467)
        idx, img, bbox, label = da.i_get(id)
        show_image(img, bbox, label)
    
    
    