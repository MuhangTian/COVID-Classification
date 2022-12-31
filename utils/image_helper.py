''' helpers for image data '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

def get_img(name): return 'data/self-data/images/{}'.format(name)

def get_bbox(df, name):
    ''' get coordinates of bounding box '''
    idx = df.index[df['filename'] == name].tolist()[0]
    return df.loc[idx]['xmin'], df.loc[idx]['ymin'], df.loc[idx]['xmax'], df.loc[idx]['ymax']

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
    color = 'red' if df.loc[idx]['label'] == 'Positive' else 'tab:blue'
    img = Image.open(get_img(img_name))
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    plot_bbox(ax, bbox, color)
    plt.title(img_name)
    plt.show()
    
if __name__ == '__main__':
    df = pd.read_csv('data/Tony_annotated.csv')
    names = df['filename'].tolist()
    for e in names: plot_image(df, e)
    