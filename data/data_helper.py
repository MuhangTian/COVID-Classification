'''To help with image data and label annotation files'''
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
import json

def cont_int(path, num):
    """ Check whether files in a directory has consecutive integer names starting from num """
    files = os.listdir(path)
    try: files.remove('.DS_Store')
    except: files
    files = sorted([int(e.replace('.jpg', '')) for e in files])
    for i in range(num, num+len(files)):
        if i == files.pop(0): continue
        else: return False
    return True

def get_inter(arr1, arr2):
    """ Obtain intersection of two arrays """
    arr2, arr1 = set(arr2), set(arr1)
    arr = []
    for e in arr1:
        if e in arr2: arr.append(e)
    return arr

def get_comp(A, B):
    '''Obtain A not in B'''
    B, A = set(B), set(A)
    arr = []
    for e in A:
        if e not in B: arr.append(e)
    return arr

def files_rename(path, start):
    """
    Rename all files in a directory according to consecutive integers

    Args:
        path (str): path of directory
        start (int): start of the numbering
    """
    files = os.listdir(path)
    try: files.remove('.DS_Store')
    except: files
    int_arr = ['{}.jpg'.format(i) for i in range(start, start+len(files))]
    inter_files = get_inter(int_arr, files)
    inter_int = [int(e.replace('.jpg', '')) for e in inter_files]
    comp_files = get_comp(files, inter_files)
    comp_int = get_comp(np.arange(start, start+len(files)).tolist(), inter_int)
    
    before = len(os.listdir(path)) - 1
    c = 0
    for file in comp_files: 
        os.rename('{}/{}'.format(path, file), '{}/{}.jpg'.format(path, comp_int[c]))
        c += 1
    after = len(os.listdir(path)) - 1
    
    if before == after: 
        print('No files lost :)')
        print('Number of images: {}'.format(before))
    else: print('Files lost!!! :(')
    print('File names are consecutive integers from {}: {}'.format(start, cont_int(path, start)))
    if before == after and cont_int(path,start) == True:
        ph = 'SUCCESS'
    else: ph = 'FAILED'
    print('---------------------- {} ----------------------'.format(ph))

def find_rotated(path, start=None, num=None):
    '''Output IDs of all images that have rotated bounding box'''
    f = open(path)
    j = json.load(f)
    id = []
    c = 0
    for e in j:
        try:
            rotation = e['annotations'][0]['result'][0]['value']['rotation']
            c += 1
            if rotation != 0: 
                id.append(e['id'])
        except: continue
    if len(id) == 0: print('All bounding boxese are NOT rotated')
    elif num == None and start == None: print('Total Images: {}\nRotated IDs: {}'.format(c, id))
    elif num == None: print('Total Images: {}\nRotated IDs: {}'.format(c, id[id.index(start):]))
    elif start == None: print('Total Images: {}\nRotated IDs: {}'.format(c, id[:num]))
    else: print('Total Images: {}\nRotated IDs: {}'.format(c, id[id.index(start):id.index(start)+num]))
    
def xml_csv(path): # TODO: Implement this
    files = os.listdir(path)
    files.remove('.DS_Store')
    # filename, size, object
    for file in files:
        df = pd.read_xml(file)
        

if __name__ == '__main__':
    # xtree = et.parse('data/Tony_cropped_annotated/Annotations/0a6ae273-458.xml')
    # root = xtree.getroot()
    # print(root.findall('filename')[0].text) # file name
    # for child in root.findall('size'):
    #     for child2 in child:
    #         print(child2.tag, child2.text)  # image size information
    # for child in root.findall('object'):
    #     for child2 in child:
    #         if child2.tag == 'name':
    #             print(child2.text)  # label
    #         elif child2.tag == 'bndbox':
    #             for child3 in child2:
    #                 print(child3.tag, child3.text)  # bounding box coordinates
    
    find_rotated('data/project.json', start=494, num=None)