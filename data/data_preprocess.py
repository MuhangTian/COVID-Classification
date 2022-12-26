'''File to preprocess image data and extract information'''
import os
import numpy as np

def cont_int(path, num):
    """ Check whether files in a directory has consecutive integer names starting from num """
    files = os.listdir(path)
    files.remove('.DS_Store')
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
    files.remove('.DS_Store')
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
    
if __name__ == '__main__':
    files_rename('data/Tony_cropped', 501)
    # print(get_inter([1,2,3,4], [1,3]))
    # os.rename('data/Tony_cropped/501.jpg', 'data/Tony_cropped/513.jpg')